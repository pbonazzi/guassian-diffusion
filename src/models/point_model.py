from typing import List
import open3d as o3d
import math
import pdb
import torch
import imageio.v2 as imageio
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from matplotlib import cm
import matplotlib.colors as mpc
from pytorch3d.renderer.lighting import PointLights,AmbientLights
from pytorch3d.ops import knn_points

from . import BaseGenerator
from .. import get_debugging_mode, get_debugging_tensor
from ..utils import get_tensor_values
from src.utils.mathHelper import estimate_pointcloud_normals
from src.core.cloud import PointClouds3D, PointCloudsFilters
from utils.common import get_point_colors_from_verts
from src.models.sh import eval_sh

def save_grad_with_name(name):
    def _save_grad(grad):
        dbg_tensor = get_debugging_tensor()
        # a dict of list of tensors
        dbg_tensor.pts_world_grad[name] = [grad[b].detach().cpu() for b in range(grad.shape[0])]

    return _save_grad

def save_mask_grad():
    def _save_grad(grad):
        dbg_tensor = get_debugging_tensor()
        dbg_tensor.img_mask_grad = grad.detach().cpu()
    return _save_grad


class Model(nn.Module):
    def __init__(self, points, normals, colors, alphas, renderer, texture=None, colors_init = "grey", size_neigh_for_normals=3,
                 learn_points=True, learn_normals=True, learn_colors=True, learn_alphas=True, learn_sh=True, use_light=True, 
                 light_source="camera_flash", device='cpu', **kwargs):
        """
        points (1,N,3)
        normals (1,N,3)
        colors (1,N,3)
        points_activation (1,N,1)
        points_visibility (1,N,1)
        renderer
        texture
        """
        super().__init__()
        self.device = device
        self.cameras = None  # will be set in forward pass
        self.hooks = []
        
        # points
        self.points = nn.Parameter(points.to(device=device)).requires_grad_(learn_points)
        self.learn_points = learn_points
        self.n_points_per_cloud = self.points.shape[1]
        self.register_buffer('points_activation', torch.full(self.points.shape[:2], True, device=device, dtype=torch.bool)) # filters
        self.points_filter = PointCloudsFilters(activation=self.points_activation).to(device) # filters
        
        # normals
        self.normals = nn.Parameter(normals.to(device=device)).requires_grad_(learn_normals)    
        self.learn_normals = learn_normals
        self.size_neigh_for_normals = size_neigh_for_normals # for normal loss and in case normals are not learned

        # colors
        self.colors_init = colors_init # color of the pc in case it didn't have one
        self.colors = nn.Parameter(colors.to(device=device)).requires_grad_(learn_colors)       
        self.alphas = nn.Parameter(alphas.to(device=device)).requires_grad_(learn_alphas)
        self.learn_alphas = learn_alphas
        self.learn_colors = learn_colors
    
        # spherical-harmonics
        self.sh_n, sh_param = 2, [self.colors]
        for i in range((self.sh_n + 1) ** 2):
            sh_param.append(torch.rand_like((self.colors)))
        sh_param = torch.cat(sh_param, -1)
        self.sh_param = nn.Parameter(sh_param.to(device=device)).requires_grad_(learn_sh)
        self.learn_sh = learn_sh
        
        # renderer 
        self.renderer = renderer.to(device=device)
        self.use_light = use_light
        self.light_source = light_source
        self.texture = texture.to(device=device)


    def get_point_clouds(self, points=None, with_colors=False, with_lights=True, filter_inactive=True, verbose=False, **kwargs):
        """
        Create point clouds using points parameter, normals from the implicit function gradients, colors from the color decoder.
        Pointclouds contains additional features: activation and visibility (if available)
        """
        points = self.points
        normals =  F.normalize(self.normals, dim=-1)

        # features
        if self.learn_sh:
            features =  self.sh_param
        elif self.learn_colors:
            features =  torch.clamp(self.colors, min=0, max=1)
            if self.learn_alphas:
                features = torch.cat([features, torch.clamp(self.alphas, min=0, max=1)], dim=2)
        else :
            features = get_point_colors_from_verts(points, self.colors_init)
            if self.learn_alphas:
                features = torch.cat([features, torch.clamp(self.alphas, min=0, max=1)], dim=2)
            assert features.min() >= 0 and  features.max() <= 1 

        if not self.learn_normals:
            normals = estimate_pointcloud_normals(PointClouds3D(points=points, features=features), neighborhood_size=self.size_neigh_for_normals)
        
        pointclouds = PointClouds3D(points=points, normals=normals, features=features)

        # filter inactive points
        self.points_filter.set_filter(activation=self.points_activation)
        self.points_filter.to(pointclouds.device)
        if filter_inactive:
            self.points_filter.to(pointclouds.device)
            pointclouds = self.points_filter.filter_with(pointclouds, ('activation',))


        # apply lighting
        if self.use_light and with_lights:
            lights = AmbientLights(ambient_color = torch.tensor([[1, 1, 1]])).to(self.device)
            if self.light_source == "camera_flash":
                lights=PointLights(location=kwargs["cameras"].get_camera_center()).to(self.device)
            pointclouds = self.texture(pointclouds, lights=lights, **kwargs)

        elif len(kwargs["cameras"]) != len(pointclouds) and len(pointclouds) == 1:
            pointclouds = pointclouds.extend(len(kwargs["cameras"]))

        return pointclouds


    def forward(self, n_of_GT, view=None, mask_img=None, locations=None, filter_outliers=False, verbose=False, only_base=True, **kwargs):
        """
        Returns:
            rgb (tensor): (N, H, W, 3)
            mask (tensor): (N, H, W, 1)
        """
        self.cameras = kwargs.get('cameras', self.cameras)
        assert(self.cameras is not None), 'cameras wasn\'t set.'

        batch_size = self.cameras.R.shape[0]

        if batch_size != self.points.shape[0]:
            assert(batch_size == 1 or self.points.shape[0] == 1),'Cameras batchsize and points batchsize are incompatible.'
        
        # do not filter inactive here, because it will be filtered in renderer
        colored_pc = self.get_point_clouds(with_colors=True, locations=locations, filter_inactive=False, verbose=verbose, **kwargs)
        rgba = self.renderer(colored_pc, point_clouds_filter=self.points_filter, cameras=self.cameras)
        # shfeat = rgba[..., 3:-1]
        harmonics = None
        # shfeat = torch.stack(shfeat.split(3, 3), -1)
        # harmonics = eval_sh(self.sh_n, shfeat, view)
        #mask = rgba[..., -1:]  * (harmonics[...,:1] + harmonics[...,1:2] + harmonics[...,2:])
        mask = rgba[..., -1:]
    
        # spherical harmonics
        if only_base :
            rgb = rgba[..., :3]
        else:
            shfeat = rgba[..., 3:-1]
            shfeat = torch.stack(shfeat.split(3, 3), -1)
            rgb =  rgba[..., :3] + eval_sh(self.sh_n, shfeat, view)

        rgb = torch.clamp(rgb, min = 0 , max = 1)
        mask = torch.clamp(mask, min = 0 , max = 1)

        # the activation is expanded when creating visibility filter
        self.points_filter.visibility = self.points_filter.visibility.any(dim=0, keepdim=True)
        self.points_filter.activation = self.points_filter.activation[:1]
        self.points_filter.inmask = self.points_filter.inmask[:1]

        # Gets point clouds repulsion and projection losses
        point_clouds = self.get_point_clouds(with_colors=True, locations=locations, **kwargs)

        # compute inmask filter, which is used for projection and repulsion
        with autograd.no_grad():
            self.points_filter.visibility = self.points_filter.visibility[self.points_filter.activation].unsqueeze(0)
            if mask_img is not None:
                with autograd.enable_grad():
                    p_screen_hat = self.cameras.transform_points(point_clouds.points_padded())

                p = -p_screen_hat[..., :2]
                mask_pred = get_tensor_values(mask_img.float(),p[:n_of_GT, ...].clamp(-1.0, 1.0),squeeze_channel_dim=True).bool()
                mask_pred = mask_pred.any(dim=0, keepdim=True)
                mask_pred = mask_pred & self.points_filter.visibility
                self.points_filter.set_filter(inmask=mask_pred)

        return {'iso_pcl': point_clouds, 'img_pred': rgb, 'mask_img_pred': mask, "harmonics": harmonics}

    def render(self, p_world=None, only_base=True, locations=None, cameras=None, **kwargs) -> torch.Tensor:
        """ Render point clouds to RGBA (N, H, W, 4) images"""
        cameras = cameras or self.cameras
        batch_size = cameras.R.shape[0]

        pointclouds = self.get_point_clouds(p_world, with_colors=True, with_normals=True, locations=locations, cameras=cameras, filter_inactive=False)
        
        if len(cameras) != len(pointclouds):
            pointclouds = pointclouds[:len(cameras)]

        if batch_size != len(pointclouds) and len(pointclouds) == 1:
            pointclouds = pointclouds.extend(batch_size)

        # render
        rgba = self.renderer(pointclouds, point_clouds_filter=self.points_filter, cameras=cameras)
        # shfeat = rgba[..., 3:-1]
        # shfeat = torch.stack(shfeat.split(3, 3), -1)
        # harmonics = eval_sh(self.sh_n, shfeat, kwargs["data"]["view"])
        # mask = rgba[..., -1:]  + (harmonics[...,:1] * harmonics[...,1:2] * harmonics[...,2:]).clamp(0,1)
        mask = rgba[..., -1:]
        
        # spherical harmonics
        if only_base :
            rgb = rgba[..., :3]
        else:
            shfeat = rgba[..., 3:-1]
            shfeat = torch.stack(shfeat.split(3, 3), -1)
            rgb =  rgba[..., :3] + eval_sh(self.sh_n, shfeat, kwargs["data"]["view"])

        rgba = torch.cat([rgb, mask], dim=-1)
        rgba = torch.clamp(rgba, min = 0 , max = 1)

        self.points_filter.activation = self.points_filter.activation[:1]
        self.points_filter.visibility = self.points_filter.visibility.any(dim=0, keepdim=True)
        self.points_filter.inmask = self.points_filter.inmask[:1]

        return rgba

    def debug(self, is_debug, **kwargs):
        if is_debug:
            # nothing to do
            pass
        else:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()

    def detach_normals(self) :
        self.learn_normals = False
        self.normals = nn.Parameter(self.normals.detach()).requires_grad_(self.learn_normals)
        self.use_light = False
    
    def detach_sh(self) :
        self.learn_sh = False
        self.sh_param = nn.Parameter(self.sh_param.detach()).requires_grad_(self.learn_sh)

    def attach_sh(self) :
        self.learn_sh = True
        self.sh_param = nn.Parameter(self.sh_param.detach()).requires_grad_(self.learn_sh)
  
    def attach_normals(self) :
        self.learn_normals = True
        self.use_light = True

    def remove_and_repeat(self):

        # find outliers
        pts_all = self.points.data[0]
        pts_in = remove_outlier(pts_all.cpu().data.numpy())
        pts_in = torch.tensor(pts_in).to(self.device).float()
        idx = knn_points(pts_in[None,...], pts_all[None,...], None, None, 1).idx[0,:,0]

        # save gradients
        if self.learn_points:
            saved_points_grad = self.points.grad[0][idx].unsqueeze(0).detach()
        if self.learn_sh:
            saved_shparams_grad = self.sh_param.grad[0][idx].unsqueeze(0).detach()
        if self.learn_colors:
            saved_colors_grad = self.colors.grad[0][idx].unsqueeze(0).detach()
        if self.learn_alphas:
            saved_alphas_grad = self.alphas.grad[0][idx].unsqueeze(0).detach()
        if self.normals.grad is not None:
            saved_normals_grad = self.normals.grad[0][idx].unsqueeze(0).detach()

        # slightly shift the added points
        pc = o3d.cuda.pybind.geometry.PointCloud(points=o3d.cuda.pybind.utility.Vector3dVector(self.points[0].detach().cpu()) )
        bb = pc.get_oriented_bounding_box()
        diag = math.dist(bb.get_max_bound(),  bb.get_min_bound())
        offset_points = torch.from_numpy(np.random.uniform(low=-diag*0.01, high=diag*0.01, size=(1, len(idx), 3))).float().to(self.device)

        # duplicate inliers
        new_points = torch.cat([self.points[0][idx].unsqueeze(0).detach(), self.points[0][idx].unsqueeze(0).detach()+ offset_points], dim=1)
        self.points = nn.Parameter(new_points).requires_grad_(self.learn_points)
        self.sh_param = nn.Parameter(self.sh_param[0][idx].unsqueeze(0).detach().repeat(1, 2, 1)).requires_grad_(self.learn_sh)
        self.normals = nn.Parameter(self.normals[0][idx].unsqueeze(0).detach().repeat(1, 2, 1)).requires_grad_(self.learn_normals)
        self.colors = nn.Parameter(self.colors[0][idx].unsqueeze(0).detach().repeat(1, 2, 1)).requires_grad_(self.learn_colors)
        self.alphas = nn.Parameter(self.alphas[0][idx].unsqueeze(0).detach().repeat(1, 2, 1)).requires_grad_(self.learn_alphas)

        if self.learn_points:
            self.points.grad = saved_points_grad.repeat(1, 2, 1)
        if self.learn_sh:
            self.sh_param.grad = saved_shparams_grad.repeat(1, 2, 1)
        if self.normals.grad is not None:
            self.normals.grad = saved_normals_grad.repeat(1, 2, 1)
        if self.learn_colors:
            self.colors.grad = saved_colors_grad.repeat(1, 2, 1)
        if self.learn_alphas:
            self.alphas.grad = saved_alphas_grad.repeat(1, 2, 1)

        self.points_activation = self.points_activation[:,idx].repeat(1, 2)
        self.points_filter = PointCloudsFilters(
            device=self.device, 
            activation=self.points_filter.activation[:,idx].repeat(1, 2),
            visibility=self.points_filter.visibility[:,idx].repeat(1, 2),
            inmask = self.points_filter.inmask[:,idx].repeat(1, 2))


def remove_outlier(pts):
    import open3d as o3d
    import numpy as np
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(voxel_size=0.010)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    return np.array(pcd.points)[np.array(ind)]

class Generator(BaseGenerator):
    def __init__(self, model, device='cpu', **kwargs):
        super().__init__(model, device=device)

    def generate_mesh(self, *args, **kwargs) -> o3d.geometry.TriangleMesh:
        """
        Generage mesh via poisson reconstruction
        """
        pdd.set_trace()
        pcl = self.model.get_point_clouds(**kwargs)
        points = pcl.points_list()
        normals = pcl.normals_list()
        colors = pcl.features_list()[:3]

        # logger_py.info('Running poisson reconstruction')
        meshes = []
        for b in range(len(points)):

            pc = o3d.cuda.pybind.geometry.PointCloud()
            pc.points=o3d.cuda.pybind.utility.Vector3dVector(points[b].detach().cpu().numpy()) 
            pc.normals=o3d.cuda.pybind.utility.Vector3dVector(normals[b].detach().cpu().numpy()) 

            color_dim = colors[b].shape[-1]
            cmap = cm.get_cmap('jet')
            if color_dim == 1:
                cvalue = colors[b].squeeze(1)
                normalizer = mpc.Normalize(vmin=cvalue.min(), vmax=cvalue.max())
                cvalue = normalizer(cvalue)
                colors[b] = cmap(cvalue)
            if color_dim == 4:
                colors[b] = colors[b][:,:3]*np.expand_dims(colors[b][:,3], 1)
            if color_dim == 3:
                normalizer = mpc.Normalize(vmin=colors[b].min(), vmax=colors[b].max())
                colors[b] = normalizer(colors[b])

            pc.colors=o3d.cuda.pybind.utility.Vector3dVector(colors[b].detach().cpu().numpy())

            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=8)
            meshes.append(mesh)

        if len(meshes) == 1:
            return meshes.pop()
        return meshes

    def generate_pointclouds(self, *args, **kwargs) -> List[o3d.geometry.PointCloud]:
        outputs = super().generate_pointclouds(*args, **kwargs)
        self.model.eval()
        pcl = self.model.get_point_clouds(require_normals_grad=False, **kwargs)
        points = pcl.points_list()
        normals = pcl.normals_list()
        colors = pcl.features_list()
        points = [x[:, :3].detach().cpu().numpy() for x in points]

        if normals is None:
            normals = [None] * len(points)
        else:
            normals = [x[:, :3].detach().cpu().numpy() for x in normals]
            
        if colors is None:
            colors = [None] * len(points)
        else:
            colors = [x.detach().cpu().numpy() for x in colors]

        for b in range(len(points)):
            if points[b].size != 0:

                pc = o3d.cuda.pybind.geometry.PointCloud()
                pc.points=o3d.cuda.pybind.utility.Vector3dVector(points[b]) 
                pc.normals=o3d.cuda.pybind.utility.Vector3dVector(normals[b]) 

                if colors[b] is not None:
                    color_dim = colors[b].shape[-1]
                    if color_dim == 1:
                        cmap = cm.get_cmap('jet')
                        cvalue = colors[b].squeeze(1)
                        normalizer = mpc.Normalize(vmin=cvalue.min(), vmax=cvalue.max())
                        cvalue = normalizer(cvalue)
                        mapped_colors = cmap(cvalue)
                    if color_dim == 4:
                        mapped_colors = colors[b][:,:3] + np.expand_dims(colors[b][:,3], 1)
                    else:
                        #TODO : handle sh
                        normalizer = mpc.Normalize(vmin=colors[b][..., :3].min(), vmax=colors[b][..., :3].max())
                        mapped_colors = normalizer(colors[b][..., :3]).data

                pc.colors=o3d.cuda.pybind.utility.Vector3dVector(mapped_colors)
                outputs.append(pc)

        return outputs

    def generate_meshes(self, *args, **kwargs) -> List[o3d.geometry.TriangleMesh]:
        outputs = super().generate_meshes(*args, **kwargs)
        self.model.eval()
        meshes = Generator.generate_mesh(self, *args, **kwargs)
        outputs.extend(meshes)
        return outputs

    def generate_images(self, data, **kwargs) -> List[np.array]:
        """Return list of rendered point clouds (H,W,3)"""
        outputs = super().generate_images(data, **kwargs)
        with torch.autograd.no_grad():
            self.model.eval()
            rgba = self.model.render(data=data, **kwargs)
            assert rgba.min() >= 0 and rgba.max() >= 1
            if rgba is not None:
                rgba = rgba.detach().cpu().numpy()
                rgba = [np.clip(img, 0, 1).squeeze(0) for img in np.vsplit(rgba, rgba.shape[0])]
                rgba = [rgba2rgb(rgba[0])]
                outputs.extend(rgba)
            return outputs


def rgba2rgb( rgba, background=(1,1,1)):
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0]*255, rgba[:,:,1]*255, rgba[:,:,2]*255, rgba[:,:,3]

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )