import yaml
import pdb
import open3d as o3d
import numpy as np
import os, argparse
from easydict import EasyDict as edict
import torch
import torch.optim as optim
import pytorch3d
from pytorch3d.renderer import (TexturesVertex,)
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.renderer.lighting import PointLights, DirectionalLights, AmbientLights
from tqdm import tqdm
from src.utils import get_class_from_string
from src import logger_py
from utils.common import get_point_colors_from_verts, get_tri_color_lights_for_view

import torch.nn.functional as F
from src.utils.mathHelper import estimate_pointcloud_normals

# parser
def file_choices(choices,fname):
    ext = os.path.splitext(fname)[1][1:]
    if ext not in choices:
       argparse.ArgumentTypeError("file doesn't end with one of {}".format(choices))
    return fname

def normalize_unit_sphere(verts):

    # normalize to unit sphere
    vert_center = torch.mean(verts, dim=0)
    vert_scale = torch.norm(verts, dim=1).max()
    verts_clone = (verts - vert_center)/vert_scale
    return verts_clone

# terminal colors
class bcolors:
    # colors for terminal logs
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    ERROR = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# device
def select_device(device_id: str) -> torch.device:
    if torch.cuda.is_available():
        print(f"{bcolors.OKGREEN} GPU available {bcolors.ENDC}")
        # device_id = os.environ.get("CUDA_VISIBLE_DEVICES")
        print('GPU description :', torch.cuda.get_device_name(0), device_id)
        device = torch.device("cuda")
    else:
        print(f"{bcolors.OKGREEN} GPU not available, using CPU{bcolors.ENDC}")
        device = torch.device("cpu")
    return device


# General config
def load_config(path):
    ''' Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.Loader))

    return cfg


def save_config(path, config):
    """
    Save config dictionary as json file
    """
    out_dir = os.path.dirname(path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if os.path.isfile(path):
        logger_py.warn(
            "Found file existing in {}, overwriting the existing file.".format(out_dir))

    with open(path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    logger_py.info("Saved config to {}".format(path))


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = edict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def _get_tensor_with_default(opt, key, size, fill_value=0.0):
    if key not in opt:
        return torch.zeros(*size).fill_(fill_value)
    else:
        return torch.FloatTensor(opt[key])


def create_optimizer(cfg, model):
    # Optimizer
    param_groups = []
    if cfg["model"]["learn_normals"]:
        param_groups.append({"name": "normals", "params": [model.normals], "lr": cfg['training']['lr_normals'], 'betas': cfg['training']['betas']})
    if cfg["model"]["learn_sh"]:
        param_groups.append({"name": "sh_param", "params": [model.sh_param], "lr": cfg['training']['lr_sh'], 'betas': cfg['training']['betas']})
    if cfg["model"]["learn_points"]:
        param_groups.append({"name": "points", "params": [model.points], "lr": cfg['training']['lr_points'], 'betas': cfg['training']['betas']})
    if cfg["model"]["learn_colors"]:
        param_groups.append({"name": "colors", "params": [model.colors], "lr": cfg['training']['lr_colors'], 'betas': cfg['training']['betas']})
    if cfg["model"]["learn_alphas"]:
        param_groups.append({"name": "alphas", "params": [model.alphas], "lr": cfg['training']['lr_alphas'], 'betas': cfg['training']['betas']})
    return optim.Adam(param_groups, betas=(0.5, 0.9))

def create_scaler():
    # Scaler
    return torch.cuda.amp.GradScaler()
    
def create_scheduler(cfg, optimizer, last_epoch=0):
    # Scheduler
    if cfg["scheduler_name"] == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, cfg['scheduler_milestones'], gamma=cfg['scheduler_gamma'], last_epoch=last_epoch-1)
    if cfg["scheduler_name"] == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    if cfg["scheduler_name"] == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, cfg['scheduler_decay'], -1)
    return scheduler


def create_texture(cfg, device):
    from src.core.texture import SurfaceSplattingShader
    if cfg["model"]["light_source"] == "camera_flash" or cfg["model"]["light_type"] == "point":
        texture = SurfaceSplattingShader(lights=PointLights().to(device))
    elif cfg["model"]["light_type"] == "ambient":
        texture = SurfaceSplattingShader(lights=AmbientLights(ambient_color= torch.tensor([[1, 1, 1]])).to(device))
    elif cfg["model"]["light_type"] == "directional":
        texture = SurfaceSplattingShader(lights=DirectionalLights().to(device))
    return texture


def create_model(cfg: dict, device: torch.device, data_dict=None, texture=None, images=None, **kwargs):
    ''' Returns model

    Args:
        cfg (edict): imported yaml config
        device (device): pytorch device
    '''

    from src.models.point_model import Model
    from src.core.cloud import PointClouds3D

    # initialize parameters
    if  cfg["data"]["init_from_noisy"]:
        points : torch.float32 =data_dict["noisy_points"].unsqueeze(0)
        normals : torch.float32 = data_dict["noisy_normals"].unsqueeze(0)
        colors : torch.float32 = data_dict["noisy_colors"].unsqueeze(0)

    elif cfg["data"]["init_from_images"] and len(images) >= 3:
        pdb.set_trace()
        pointcloud = create_pointcloud(cfg["data"]["camera_image_size"],  data_dict["camera_cal"][0, :3, :3], data_dict["camera_mat"], images, cfg["model"]["n_points_per_cloud"])
        if pointcloud.shape[0] < 1000 :
            sphere_mesh = ico_sphere(level=4)
            sphere_mesh.scale_verts_(0.5)
            points, normals = sample_points_from_meshes(sphere_mesh, return_normals=True,  num_samples=cfg["model"]["n_points_per_cloud"])
            colors = get_point_colors_from_verts(points , cfg["data"]["init_color"])
        else:
            points : torch.float32 = pointcloud[:,:3].unsqueeze(0)
            pc = PointClouds3D(points=pointcloud[:,:3].unsqueeze(0))
            normals = estimate_pointcloud_normals(pc, 8)
            colors : torch.float32 = pointcloud[:,3:].unsqueeze(0)
    elif cfg["data"]["init_another_mesh"]:
        points : torch.float32 =data_dict["init_verts"].unsqueeze(0)
        normals : torch.float32 = data_dict["init_normals"].unsqueeze(0)
        colors : torch.float32 = data_dict["init_colors"].unsqueeze(0)

    elif cfg["data"]["init_cone"]:
        mesh = o3d.geometry.TriangleMesh.create_cone(resolution=cfg["model"]["n_points_per_cloud"], radius=0.5)
        verts = normalize_unit_sphere(torch.tensor(np.asarray(mesh.vertices)).float().to(device))
        faces = torch.tensor(np.asarray(mesh.triangles)).to(device)
        colors = get_point_colors_from_verts(verts, color_type=cfg["data"]["init_color"]).float().to(device)
        textures = TexturesVertex(verts_features=colors.unsqueeze(0))
        structure = pytorch3d.structures.Meshes(verts=[verts], faces=[faces], textures=textures).to(device=device)         
        points, normals, colors = sample_points_from_meshes( structure, num_samples=cfg["model"]["n_points_per_cloud"], return_normals=True, return_textures=True)
    
    elif cfg["data"]["init_cylinder"]:
        mesh = o3d.geometry.TriangleMesh.create_cylinder(resolution=cfg["model"]["n_points_per_cloud"])
        verts = normalize_unit_sphere(torch.tensor(np.asarray(mesh.vertices)).float().to(device))
        faces = torch.tensor(np.asarray(mesh.triangles)).to(device)
        colors = get_point_colors_from_verts(verts, color_type=cfg["data"]["init_color"]).float().to(device)
        textures = TexturesVertex(verts_features=colors.unsqueeze(0))
        structure = pytorch3d.structures.Meshes(verts=[verts], faces=[faces], textures=textures).to(device=device)         
        points, normals, colors = sample_points_from_meshes( structure, num_samples=cfg["model"]["n_points_per_cloud"], return_normals=True, return_textures=True)
    elif cfg["data"]["init_icosphere"]:
        sphere_mesh = ico_sphere(level=4)
        sphere_mesh.scale_verts_(0.5)
        points, normals = sample_points_from_meshes(sphere_mesh, return_normals=True,  num_samples=cfg["model"]["n_points_per_cloud"])
        colors = get_point_colors_from_verts(points , cfg["data"]["init_color"])
    else:
        points : torch.float32 =data_dict["points"].unsqueeze(0)
        normals : torch.float32 = data_dict["normals"].unsqueeze(0)
        colors : torch.float32 = data_dict["colors"].unsqueeze(0)

    alphas : torch.float32 = torch.ones_like(normals[:,:,:1])

    if cfg["model"]["learn_sh"] :
        background_color = [1] * 30
    elif cfg["model"]["learn_alphas"] :
        background_color = [1] * 4
    else :
        background_color = [1] * 3


    renderer = create_renderer(raster_settings=cfg["dss"]["raster_settings"], background_color=background_color).to(device)
    model = Model(  points=points, normals=normals, colors=colors, alphas=alphas, colors_init=cfg["data"]["init_color"],
                    renderer=renderer, device=device, texture=texture, **cfg["model"])

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    return model.to(device=device)


def create_lights(light_type, device, cams=None, location=((0, 1, 0),)):

    if light_type == "point":
        lights = PointLights(location = location).to(device)
    elif light_type == "ambient":
        lights = AmbientLights(ambient_color = torch.tensor([[1, 1, 1]])).to(device)
    elif light_type == "directional":
        lights = DirectionalLights().to(device)
    elif light_type == "tri_color":
        lights = get_tri_color_lights_for_view(cams).to(device)
    return lights

def create_generator(cfg, model, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    from src.models.point_model import Generator
    generator = Generator(model, device)
    return generator


def create_renderer(raster_settings, background_color=None):
    """ Create renderer """
    from src.core.renderer import SurfaceSplattingRenderer
    from src.core.rasterizer import SurfaceSplattingRasterizer
    from pytorch3d.renderer import NormWeightedCompositor, AlphaCompositor


    compositor = NormWeightedCompositor(background_color=background_color)
    #compositor = AlphaCompositor(background_color=background_color)
    
    rasterizer = SurfaceSplattingRasterizer(raster_settings=raster_settings)
    renderer = SurfaceSplattingRenderer(rasterizer=rasterizer, compositor=compositor)

    return renderer


def create_pointcloud(image_size, K, poses, images, num_points):
    with torch.no_grad():
        pc,color,N = [],[], image_size
        if isinstance(images, list):
            images = torch.stack(images)

        train_n = len(images)

        [xs,ys,zs],[xe,ye,ze] = [-2,-2,-2],[2,2,2]
        pts_all = []
        for h_id in tqdm(range(N)):
            i, j = torch.meshgrid(torch.linspace(xs, xe, N).cuda(), torch.linspace(ys, ye, N).cuda())  # pytorch's meshgrid has indexing='ij'
            i, j = i.t(), j.t()
            pts = torch.stack([i, j, torch.ones_like(i).cuda()], -1)
            pts[...,2] = h_id / N * (ze - zs) + zs
            pts_all.append(pts.clone())
            
            uv = batch_get_uv_from_ray(image_size,image_size, K, poses.to(pts.device),pts)
            if images.shape != (len(images), 3, image_size, image_size):
                images = images.permute(0, 3, 1, 2)
            result = F.grid_sample(images, uv, align_corners=False).permute(0,2,3,1)

            margin = 0.05
            result[(uv[..., 0] >= 1.0) * (uv[..., 0] <= 1.0 + margin)] = 1
            result[(uv[..., 0] >= -1.0 - margin) * (uv[..., 0] <= -1.0)] = 1
            result[(uv[..., 1] >= 1.0) * (uv[..., 1] <= 1.0 + margin)] = 1
            result[(uv[..., 1] >= -1.0 - margin) * (uv[..., 1] <= -1.0)] = 1
            result[(uv[..., 0] <= -1.0 - margin) + (uv[..., 0] >= 1.0 + margin)] = 0
            result[(uv[..., 1] <= -1.0 - margin) + (uv[..., 1] >= 1.0 + margin)] = 0

            img = ((result>0.).sum(0)[...,0]> train_n -1).float()
            pc.append(img)
            color.append(result.mean(0))
            torch.cuda.empty_cache()


        pc = torch.stack(pc,-1)
        color = torch.stack(color,-1)
        r, g, b = color[:, :, 0], color[:, :, 1], color[:, :, 2]
        idx = torch.where(pc > 0)
        color = torch.stack((r[idx],g[idx],b[idx]),-1)
        idx = (idx[1],idx[0],idx[2])
        pts = torch.stack(idx,-1).float()/N
        pts[:,0] = pts[:,0]*(xe-xs)+xs
        pts[:,1] = pts[:,1]*(ye-ys)+ys
        pts[:,2] = pts[:,2]*(ze-zs)+zs

        pts = torch.cat((pts,color),-1)
        samples = pts[:num_points]

        print('Initialization, Found {} points , Selected {} points'.format(pts.shape, samples.shape))

        return pts

def batch_get_uv_from_ray(H,W,K,poses,pts):
    RT = (poses[:, :3, :3].transpose(1, 2))
    pts_local = torch.sum((pts[..., None, :] - poses[:, :3, -1])[..., None, :] * RT, -1)
    pts_local = pts_local / (-pts_local[..., -1][..., None] + 1e-7)
    u = pts_local[..., 0] * K[0][0] + K[0][2]
    v = -pts_local[..., 1] * K[1][1] + K[1][2]
    uv0 = torch.stack((u, v), -1)
    uv0[...,0] = uv0[...,0]/W*2-1
    uv0[...,1] = uv0[...,1]/H*2-1
    uv0 = uv0.permute(2,0,1,3)
    return uv0