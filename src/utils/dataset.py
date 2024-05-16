from typing import Union
import torch
import torch.utils.data as data
import os
import imageio.v2 as imageio
import numpy as np
import pdb
import open3d  as o3d
from tqdm import tqdm
from itertools import chain
from glob import glob
import math

import pytorch3d
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    MeshRendererWithFragments,
)
from pytorch3d.renderer.lighting import PointLights, DirectionalLights
from pytorch3d.io import load_obj
from pytorch3d.ops import eyes, sample_points_from_meshes
from pytorch3d.structures import Pointclouds

# custom
from .. import logger_py
from . import get_class_from_string
from .mathHelper import decompose_to_R_and_t
from src.core.shader import HardFlatShaderwithoutSpecular
from src.core.camera import CameraSampler
from src.utils import convert_tensor_property_to_value_dict
from src.misc.imageFilters import L0Smooth, SuperPixel, Pix2PixDenoising, ImgL0Smooth
from src.training.clip_encoder import get_embed_fn, resize_img_for_clip
from src.training.trainer import rgba2rgb
from sklearn.neighbors import KDTree
from src.core.cloud import PointClouds3D
from utils.config import create_texture, create_lights, create_renderer
from utils.common import normalize_unit_sphere, get_rays, load_structure_from_config
from pytorch3d.transforms import Rotate, Transform3d, Translate

class GenericDataset(data.Dataset):
    """
    Dataset for MVR
    loads RGB, camera_mat, mask, points, lights, cameras

    Attributes:
        image_files
        mask_files
        data_dict
        point_clouds: point positions N,3 in object coordinates
        lights:
        cameras:
    """

    def __init__(self, name, device, settings):


        # settings
        self.name = name
        self.device = device
        self.settings = settings
        self.out_dir = os.path.join(self.settings["generic"]["output_dir"], "vis", "gt", name)
        os.makedirs(self.out_dir, exist_ok=True)

        # deterministic results
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(2147483647)

        # load from images or 3D object
        if self.settings["data"]["dataset_name"] == "any_obj":
            self.load_from_object()

    def load_from_images(self, images, poses, intrinsic, pts, locations, noisy_points=None):

        
        assert self.settings["data"]["val_images"] + self.settings["data"]["train_images"] < 400

        # Cast intrinsics to right types
        focal, H, W = 1, 512, 512
        self.texture = create_texture(self.settings, self.device)
        
        # self.settings["data"]["camera_fov"] =  2 * math.atan(W / (2 * focal))
        camera_params = {"fov": self.settings["data"]["camera_fov"], "znear":  self.settings["data"]["camera_znear"], "zfar":  self.settings["data"]["camera_zfar"]}
        self.data_dict = {"cameras_type": '.'.join([FoVPerspectiveCameras.__module__,  FoVPerspectiveCameras.__name__]),
                "cameras_min_dist": self.settings["data"]["camera_min_dist"],  "cameras_max_dist": self.settings["data"]["camera_max_dist"], 
                "cameras_params": camera_params,}


        if self.settings["data"]["dataset_name"] != "blender":
            if not torch.is_tensor(pts):
                pts = torch.tensor(pts)

            self.data_dict['points']  = pts[...,:3].float().to(self.device)
            self.data_dict['colors']  = pts[...,3:6].float().to(self.device)
            self.data_dict['normals']  = pts[...,6:].float().to(self.device)

            if noisy_points is not None:
                if not torch.is_tensor(noisy_points):
                    noisy_points = torch.tensor(noisy_points)

                self.data_dict['noisy_points']  = noisy_points[...,:3].float().to(self.device)
                self.data_dict['noisy_colors']  = noisy_points[...,3:6].float().to(self.device)
                self.data_dict['noisy_normals']  = noisy_points[...,6:].float().to(self.device)

        self.image_size = self.settings["data"]["camera_image_size"]

        masks = images[..., -1:]
        # images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        images = images[..., :3]

        self.rgb_images = images
        self.masks_images = masks
        self.poses = poses
        self.target_emb, self.clip_images = [], []

        if not torch.is_tensor(self.poses):
            self.poses = torch.tensor(self.poses)

        if not torch.is_tensor(self.masks_images):
            self.masks_images = torch.tensor(self.masks_images)

        if not torch.is_tensor(self.rgb_images):
            self.rgb_images = torch.tensor(self.rgb_images)

        if self.name == "val":
            self.n_imgs = self.settings["data"]["val_images"]
            self.rgb_images = self.rgb_images[-self.n_imgs:].float().to(self.device)
            self.masks_images = self.masks_images[-self.n_imgs:].float().to(self.device)
            self.data_dict['camera_mat'] = self.poses[-self.n_imgs:].float().to(self.device)
            self.data_dict['camera_loc'] = locations[-self.n_imgs:]

        elif self.name == "train":
            self.n_imgs = self.settings["data"]["train_images"]
            self.rgb_images =  self.rgb_images[:self.n_imgs].float().to(self.device)
            self.masks_images =  self.masks_images[:self.n_imgs].float().to(self.device)
            self.data_dict['camera_mat'] = self.poses[:self.n_imgs].float().to(self.device)
            self.data_dict['camera_loc'] = locations[:self.n_imgs]

        cams = FoVPerspectiveCameras( R=self.data_dict['camera_mat'][:, :3, :3], T=self.data_dict['camera_mat'][:, :3, -1]).to(self.device)

        self.data_dict['camera_cal'] = cams.get_projection_transform().get_matrix()[:,:3,:3].to(self.device)
         
        dense_depths = cams.zfar.view(-1, 1, 1, 1).clone().expand_as(self.masks_images)
        self.depth_images = torch.where(self.masks_images > 0, self.masks_images, dense_depths)
        self.data_dict['view_dir'] = torch.empty(self.n_imgs, self.image_size, self.image_size, 3).to(self.device)

        for i in tqdm(range(self.n_imgs)):
            rays_o, rays_d = get_rays(self.image_size, self.image_size, self.data_dict['camera_cal'][0], self.data_dict['camera_mat'][i])
            self.data_dict['view_dir'][i, ...] =  torch.nn.functional.normalize(rays_d, dim=2)
            
            if self.settings["training"]["lambda_sem_cons_loss"] > 0 and self.name == "train":
                with torch.no_grad():
                    target_image_i_rgb = rgba2rgb(torch.cat([ self.rgb_images[i][..., :3],  self.masks_images[i]], dim=2), device=self.device) # [512, 512, 3]
                    target_image_i_norm = resize_img_for_clip(target_image_i_rgb)  # [1, 3, 224, 224]
                    
                    self.clip_images.append((target_image_i_norm*255).detach().cpu().numpy().astype("uint8"))
                    self.target_emb.append(self.settings["training"]["init_clip"](target_image_i_norm))

            if self.settings["data"]["save_views"]:
                imageio.imwrite(os.path.join(self.out_dir, "%06d.png" % i),(self.rgb_images[i].detach().cpu().numpy() * 255.0).astype('uint8'))
        
        self.rgb_images = self.rgb_images.permute(0,3,1,2)
        self.masks_images = self.masks_images.permute(0,3,1,2)
        
        self.sem_cons_dict = {}
        if self.name == "train" and self.settings["training"]["lambda_sem_cons_loss"] > 0:
            self.sem_cons_dict["target_emb"] = self.target_emb
            self.sem_cons_dict["tree"] = KDTree(self.data_dict['camera_loc'], leaf_size=2)
            self.sem_cons_dict["clip_images"] =  self.clip_images

        self.data_dict['camera_loc'] = torch.tensor(self.data_dict['camera_loc']).float().to(self.device)

    def load_from_object(self):
        device = self.device

        if self.name == "val":
            num_cams = self.settings["data"]["val_images"]
            continuous_views = False
        elif self.name == "train":
            num_cams = self.settings["data"]["train_images"]
            continuous_views = True

        # sample camera positions
        camera_params = {"fov": self.settings["data"]["camera_fov"], "znear":  self.settings["data"]["camera_znear"], "zfar":  self.settings["data"]["camera_zfar"]}
        self.camera_sampler = CameraSampler(num_cams_total=num_cams, num_cams_batch=1, continuous_views=continuous_views,
                                    distance_range=torch.tensor(((self.settings["data"]["camera_min_dist"],  self.settings["data"]["camera_max_dist"]),)),
                                    sort_distance=True,camera_type=FoVPerspectiveCameras, camera_params=camera_params)
        
        
        self.texture = create_texture(self.settings, device)

        # load input mesh
        self.structure, self.raster_settings, verts, normals, colors =  load_structure_from_config(
            path = self.settings["data"]["input_path"], 
            input_type = self.settings["data"]["input_type"], 
            input_format = self.settings["data"]["input_format"], 
            num_of_points= self.settings["model"]["n_points_per_cloud"],
            xyz_rotation = self.settings["data"]["input_xyz_rotation"], 
            color = self.settings["data"]["input_color"], 
            image_size=self.settings["dss"]["raster_settings"]["image_size"],
            device = self.device)

        self.data_dict = {"cameras_type": '.'.join([self.camera_sampler.camera_type.__module__,  self.camera_sampler.camera_type.__name__]),
                        "cameras_min_dist": self.settings["data"]["camera_min_dist"], 
                        "cameras_max_dist": self.settings["data"]["camera_max_dist"], 
                        "cameras_params": camera_params,}

            # load init mesh
        if self.settings["data"]["init_another_mesh"]:
            init_verts, init_normals, init_colors = load_structure_from_config(
                path = self.settings["data"]["init_path"], 
                input_type = self.settings["data"]["init_type"], 
                input_format = self.settings["data"]["init_format"], 
                xyz_rotation = self.settings["data"]["init_xyz_rotation"], 
                num_of_points= self.settings["model"]["n_points_per_cloud"],
                color = self.settings["data"]["init_color"], 
                image_size=self.settings["dss"]["raster_settings"]["image_size"],
                device = device, 
                special_init=True
                )
            self.data_dict["init_verts"] = init_verts
            self.data_dict["init_normals"] = init_normals
            self.data_dict["init_colors"] = init_colors

        if self.texture is not None :
            self.data_dict["lights_type"]= '.'.join([self.texture.lights.__module__, self.texture.lights.__class__.__name__])

        self.image_size = self.settings["dss"]["raster_settings"]["image_size"]

        self.data_dict['camera_mat'] = torch.zeros(num_cams, 4, 4)
        self.data_dict['camera_loc'] = torch.zeros(num_cams, 3)
        self.data_dict['camera_cal'] = torch.zeros(num_cams, 4, 4)
        self.data_dict['view_dir'] = torch.zeros(num_cams, self.image_size, self.image_size, 3)

        self.renderer = create_renderer(self.settings["dss"]["raster_settings"], background_color=[1]*3)

        self.data_dict['points']  = verts
        self.data_dict['normals'] = normals
        self.data_dict['colors'] = colors


        # save original point cloud
        if self.settings["data"]["save_pointcloud"] and self.name == "val":
            pc = o3d.cuda.pybind.geometry.PointCloud()
            pc.points=o3d.cuda.pybind.utility.Vector3dVector(verts.cpu()) 
            pc.normals=o3d.cuda.pybind.utility.Vector3dVector(normals.cpu()) 
            pc.colors=o3d.cuda.pybind.utility.Vector3dVector(colors.cpu())
            o3d.io.write_point_cloud(os.path.join(self.settings["generic"]["output_dir"], "vis", "gt", "original.ply"), pc)

        self._create_images()

    def _create_images(self):

        device = self.device
        idx = 0
        self.rgb_images, self.masks_images, self.depth_images, self.target_emb, self.clip_images = [], [], [], [], []

        for cams in tqdm(self.camera_sampler, desc="Image from Mesh"):

            # camera
            cams = cams.to(device)
            camera_loc = cams.get_camera_center()
            cam_RT = cams.get_world_to_view_transform().get_matrix().to(self.device)
            cam_K = cams.get_projection_transform().get_matrix().to(self.device)

            # lights
            location = None if self.settings["model"]["light_source"] != "camera_flash" else cams.get_camera_center()
            lights = create_lights(light_type=self.settings["model"]["light_type"], device=device, location=location)

            # renderer
            if self.settings["data"]["input_type"] == "mesh":
                shader = HardFlatShaderwithoutSpecular( device=device,  cameras=cams, lights=lights,)
                pyd_renderer = MeshRendererWithFragments(
                    rasterizer=MeshRasterizer(cameras=cams, raster_settings=self.raster_settings),
                    shader=shader)

                images, fragments = pyd_renderer(self.structure, cameras=cams, lights=lights)
                images = torch.clamp(images, min=0, max=1)
                
                mask = fragments.pix_to_face[..., :1] >= 0
                mask_imgs = torch.clamp(mask.to(dtype=torch.uint8), min=0, max=1)

                images = torch.cat([images[..., :3], mask.float()], dim=-1)

            else:
                # POINT CLOUD update point colors with light
                structure = self.structure
                if self.texture is not None:
                    structure = self.texture(self.structure, cameras=cams, lights=lights)
                rgba, fragments = self.renderer(structure, cameras=cams, verbose=True)
            
                mask = fragments.idx[..., :1] >= 0
                mask_imgs = torch.clamp(rgba[..., -1:], min=0, max=1)
                images = torch.clamp(rgba[..., :3], min=0, max=1)
        
            dense_depths = cams.zfar.view(-1, 1, 1, 1).clone().expand_as(mask_imgs)
            dense_depths = torch.where(mask, fragments.zbuf[..., :1], dense_depths)/255

            # apply diffusion
            if self.settings["data"]["input_filter"] == "StableDiffusion":
                pdb.set_trace()
                prepare_img = images.permute(0,3,2,1).permute(0,1,3,2)

                images = self.settings["data"]["init_diffusion"](prepare_img)

            # apply filters
            elif self.settings["data"]["input_filter"] == "L0Smooth":
                images = ImgL0Smooth(images)
                images = torch.cat([images[..., :3], mask.float()], dim=-1)
            elif self.settings["data"]["input_filter"] == "SuperPixel":
                images = SuperPixel(images)      
                images = torch.cat([images[..., :3], mask.float()], dim=-1)    

            # apply GAN denoising     
            elif self.settings["data"]["input_filter"] == "Pix2Pix":
                images = Pix2PixDenoising(images[:,:,:, :3], model=self.settings["data"]["init_pix2pix"])
                mask_imgs = (((0.2126* images[...,0] + 0.7152* images[...,1] + 0.0722 * images[...,2])*255 < 127 ) * 1).unsqueeze(3)
                images = rgba2rgb(torch.cat([rgba[..., :3], mask_imgs], dim=-1)[0], device=self.device).unsqueeze(0)
            
            images = torch.clamp(images, min=0, max=1)
            mask_imgs = torch.clamp(mask_imgs, min=0, max=1)

            # save images
            rays_o, rays_d = get_rays(self.image_size, self.image_size, cam_K[0][:3, :3], cam_RT[0])
            self.data_dict['view_dir'][idx, ...] =  torch.nn.functional.normalize(rays_d, dim=2)
            self.data_dict['camera_mat'][idx, ...] = cam_RT[0]
            self.data_dict['camera_cal'][idx, ...] = cam_K[0]
            self.data_dict['camera_loc'][idx, ...] = camera_loc[0]
            if self.texture is not None:
                self.data_dict['lights_%d' % idx] = convert_tensor_property_to_value_dict(self.texture.lights)

            self.rgb_images.append(images[0][..., :3].permute(2,0,1))
            self.masks_images.append(mask_imgs[0].permute(2,0,1))
            self.depth_images.append(dense_depths[0].permute(2,0,1))

            # clip
            if self.settings["training"]["lambda_sem_cons_loss"] > 0 and self.name == "train":
                with torch.no_grad():
                    target_image_i_rgb = rgba2rgb(torch.cat([images[0][..., :3], mask_imgs[0]], dim=2), device=self.device) # [512, 512, 3]
                    target_image_i_norm = resize_img_for_clip(target_image_i_rgb)  # [1, 3, 224, 224]
                    
                    self.clip_images.append((target_image_i_norm*255).detach().cpu().numpy().astype("uint8"))
                    self.target_emb.append(self.settings["training"]["init_clip"](target_image_i_norm))

            if self.settings["data"]["save_views"]:
                imageio.imwrite(os.path.join(self.out_dir, "%06d.png" % idx),(images[0].detach().cpu().numpy() * 255.0).astype('uint8'))

            idx += 1

        self.sem_cons_dict = {}
        if self.name == "train" and self.settings["training"]["lambda_sem_cons_loss"] > 0:
            self.sem_cons_dict["target_emb"] = self.target_emb
            self.sem_cons_dict["tree"] = KDTree(self.data_dict['camera_loc'], leaf_size=2)
            self.sem_cons_dict["clip_images"] =  self.clip_images

        self.n_imgs = len(self.rgb_images)


    def get_pointclouds(self, num_points=None) -> PointClouds3D:
        """ Returns points, normals and color in object coordinate """
        if hasattr(self, 'point_clouds'):
            if num_points is None or (self.point_clouds.points_packed().shape[0] == num_points):
                return self.point_clouds

        points = self.data_dict["points"]
        normals = self.data_dict["normals"]
        colors = self.data_dict["colors"]
        self.point_clouds = PointClouds3D([points], [normals], [colors])
        return self.point_clouds

    def get_cameras(self, camera_mat=None):
        """ Returns a cameras instance """
        if not hasattr(self, 'cameras'):
            Camera = get_class_from_string(self.data_dict["cameras_type"])
            self.cameras = Camera(**self.data_dict["cameras_params"])
        if camera_mat is not None:
            # set camera R and T
            self.cameras.R, self.cameras.T = decompose_to_R_and_t(camera_mat)
            self.cameras._N = self.cameras.R.shape[0]

        if not torch.is_tensor(self.cameras.R):
            self.cameras.R = torch.tensor(self.cameras.R, dtype=torch.float32)
            self.cameras.T = torch.tensor(self.cameras.T, dtype=torch.float32)
        return self.cameras

    def __len__(self):
        ''' Returns the length of the dataset.'''
        return self.n_imgs

    def __getitem__(self, idx):
        """
        Returns:
            data dict {"img.rgb": rgb (C,H,W),
                       "img.mask": mask (1,H,W),
                       "camera_mat": camera_mat (4,4),
                       "img.depth: depth (1,H,W)}
        """
        idx = idx % self.__len__()

        rgb = self.rgb_images[idx]
        mask = self.masks_images[idx]
        depth = self.depth_images[idx]
        assert(rgb.min() >= 0 and rgb.max() <=1), "Values outside range"
        assert(mask.min() >= 0 and mask.max() <=1), "Values outside range"

        assert(rgb.shape[-2:] == mask.shape[-2:]), "rgb {} and mask {} images must have the same dimensions.".format(rgb.shape, mask.shape)
        assert(rgb.shape[0] == 3 and rgb.ndim ==3), "Invalid RGB image shape {}".format(rgb.shape)
        assert(mask.shape[0] == 1 and mask.ndim ==3), "Invalid Mask image shape {}".format(mask.shape)

        # camera
        camera_mat = self.data_dict['camera_mat'][idx].to(self.device)
        camera_cal = self.data_dict['camera_cal'][idx].to(self.device)
        camera_loc = self.data_dict['camera_loc'][idx]
        view = self.data_dict['view_dir'][idx].to(self.device)

        out_data = {"img.rgb": rgb, "img.mask": mask, "img.depth": depth, "img.view": view, "camera_mat": camera_mat,  "camera_cal": camera_cal, "camera_loc": camera_loc}

        # light
        if light_properties := self.data_dict.get('lights_%d' % idx, None):
            out_data['lights'] = {}
            for k in light_properties.keys() :
                if isinstance(light_properties[k], (list, np.ndarray)): 
                    out_data['lights'][k] = np.array(light_properties[k][0], dtype=np.float32)

        return out_data, idx