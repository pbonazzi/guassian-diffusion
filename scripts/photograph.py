import fire

from typing import Union
import torch
import torch.utils.data as data
import os
import random
import math
import imageio
import numpy as np
import pdb, copy
from bisect import bisect
from pathlib import Path

from src.core.cloud import PointClouds3D
from utils.common import load_structure_from_config, normalize_unit_sphere
from utils.config import create_lights
from src.utils.mathHelper import decompose_to_R_and_t

from src.core.shader import HardFlatShaderwithoutSpecular
from pytorch3d.io.obj_io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.io.obj_io import load_objs_as_meshes
import numpy as np
import imageio
import os
import open3d  as o3d
from tqdm import tqdm
from itertools import chain
from glob import glob
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    
    MeshRenderer,
    MeshRasterizer,
    MeshRendererWithFragments,
    HardGouraudShader,
    HardPhongShader,
    SoftPhongShader,

    PointsRasterizationSettings,
    PointsRenderer,
    TexturesVertex,
    TexturesAtlas,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)

# add these imports
from pytorch3d.renderer.blending import BlendParams, hard_rgb_blend
from pytorch3d.renderer.mesh.shading import flat_shading

from pytorch3d.ops import eyes, sample_points_from_meshes
from pytorch3d.io import load_obj, load_ply, save_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes,Pointclouds
from src.core.camera import CameraSampler
from src.core.lighting import PointLights, DirectionalLights
from pytorch3d.renderer import Textures
from src.utils import convert_tensor_property_to_value_dict
from utils.common import get_tri_color_lights_for_view, get_light_for_view
from utils.config import create_renderer, load_config
from src.training.trainer import resize_img_for_clip, rgba2rgb
from src.core.texture import SurfaceSplattingShader

from sklearn.neighbors import KDTree

def normalize_unit_sphere(verts):

    # normalize to unit sphere
    vert_center = torch.mean(verts, dim=0)
    vert_scale = torch.norm(verts, dim=1).max()

    verts_clone = (verts - vert_center)/vert_scale

    return verts_clone

def main(idx):

    initpath = "/data/storage/bpietro/datasets/scut/Uniform/"


    inst_list = []
    # difficulty_level = ["ordinary", "complex"]
    # for level in difficulty_level:
    #     inst_list= [*inst_list, *list(Path(os.path.join(initpath, level)).iterdir())]

    initpath = "/data/storage/bpietro/datasets/sketchfab/train_mesh/"
    inst_list = [*inst_list, *list(Path(initpath).iterdir())]
    inst_list = inst_list[77:]
    #inst_list = list(Path(initpath).iterdir())
    
    # generic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raster_settings = {
                "backface_culling": False,
                "cutoff_threshold": 1.0,
                "depth_merging_threshold": 0.05,
                "Vrk_invariant": True,
                "Vrk_isotropic": False,
                "knn_points": 12,
                "radii_backward_scaler": 5,
                "image_size": 512,
                "points_per_pixel": 10,
                "bin_size": None,
                "max_points_per_bin": None,
                "clip_pts_grad": 0.05,
                "antialiasing_sigma" : 1.0,
                "frnn_radius": -1 , 
            }
    
    dss_renderer = create_renderer(raster_settings, background_color=[1,1,1])
    texture = SurfaceSplattingShader()
    num_points = 100000
    num_cams_total= 30

    # inst_list = ["simple/chair23.txt", "complex/chair36.txt", "ordinary/Chair2.txt", "ordinary/chair3.txt"]

    # inst_list = ["/data/storage/bpietro/datasets/example_data/mesh/teapot/teapot.obj"]

    num = 77
    for i in tqdm(range(0, len(inst_list)), desc="Mesh"):

        # if ("/chair" not in str(inst_list[i])) or ("train_mesh" in str(inst_list[i])):
        #     continue

        num += 1
        mesh = inst_list[i]
        idx = 0

        name = "/data/storage/bpietro/datasets/sd_data/train_"+str(num)
        #path = os.path.join(initpath, mesh) # str(mesh)
        path = str(mesh)
        
        sub_dir = "Noise"
        output_noise = os.path.join(name, sub_dir)
        os.makedirs(output_noise, exist_ok=True)


        sub_dir = "Clean"
        output_normal = os.path.join(name, sub_dir)
        os.makedirs(output_normal, exist_ok=True)


        if "test_mesh" in path or "train_mesh" in path:
            mesh_type = "mesh"
            mesh_format = "off"
        elif True:
            mesh_type = "point"
            mesh_format = "xyzn"
        else:
            mesh_type = "mesh"
            mesh_format = "obj"            

        # light
        rand = random.randint(1,1)  # no brackets necessary
        #if rand == 1:
        
        color = "grey"
        # elif rand == 2:
        #     color= "from_verts"


    
        try:
            structure, raster_settings, verts, normals, colors = load_structure_from_config(path, mesh_type, mesh_format, num_points, np.array([0,0,0]), color, 512, device, special_init=False)
        except:
            pdb.set_trace()

        # camera
        camera_params = {"znear":  1.0, "zfar":  100}
        camera_sampler = CameraSampler(
                                    num_cams_total=num_cams_total, 
                                    num_cams_batch=1, 
                                    continuous_views=False,
                                    distance_range=torch.tensor(((1.7, 2.2),)),
                                    sort_distance=True,
                                    camera_type=FoVPerspectiveCameras, #FoVOrthographicCameras
                                    camera_params=camera_params)


        pc_structure = PointClouds3D(points=[verts], normals=[normals], features=[colors])


        # get diagonal 
        pc = o3d.cuda.pybind.geometry.PointCloud(points=o3d.cuda.pybind.utility.Vector3dVector(verts.cpu()) )
        bb = pc.get_oriented_bounding_box()
        diag = math.dist(bb.get_max_bound(),  bb.get_min_bound())

        texture = SurfaceSplattingShader()

        noise_dir_type = ["Misalignment1", "Misalignment2", "Misalignment3", "Noise1", "Noise2", "Noise3",
                 "Outlier1", "Outlier2"]

        #noise_dir_type = [ "Missing_Data1",  "Missing_Data2",  "Missing_Data3"]


        for cams in camera_sampler:

            # camera
            cams = cams.to(device)
            camera_loc = cams.get_camera_center()
            cam_RT = cams.get_world_to_view_transform().get_matrix().to(device)
            cam_K = cams.get_projection_transform().get_matrix().to(device)

            # # lightsW
            location = cams.get_camera_center()
            rand = random.randint(1,2)  # no brackets necessary

            #if rand == 1:
            lights = create_lights(light_type="tri_color",  cams=cams, device=device, location=location)
            #else:
            #    lights = create_lights(light_type="point",  cams=cams, device=device, location=location)

            # # no noise
            render_structure = texture(copy.deepcopy(pc_structure), cameras=cams, lights=lights)
            images_clean = dss_renderer(render_structure, cameras=cams, lights=lights)
            images_clean = torch.clamp(images_clean, min=0, max=1)

            noise_type = random.randint(0, len(noise_dir_type)-1)
            if "train_mesh" not in path and "test_mesh" not in path and os.path.exists(path.replace("Uniform", noise_dir_type[noise_type])):
                structure, raster_settings, verts, normals, colors = load_structure_from_config(path.replace("Uniform", noise_dir_type[noise_type]), mesh_type, mesh_format, num_points, np.array([0,0,0]), color, 512, device, special_init=False)
                new_structure = PointClouds3D(points=[verts], normals=[normals], features=[colors])
            else:
                new_structure = copy.deepcopy(pc_structure)


            new_structure = copy.deepcopy(pc_structure)

            # copy and create noise to structure
            coeff = np.random.uniform(low=0.003, high=0.01, size=(1,1)).item()

            #levels = [0.003, 0.005, 0.007, 0.01]

            #for n in levels:
            #coeff = n
            offset_points = torch.from_numpy(np.random.uniform(low=-diag*coeff, high=diag*coeff, size=(new_structure.points_packed().shape))).to(device)
            new_structure = new_structure.offset(offset_points.float())
            render_structure = texture(copy.deepcopy(new_structure), cameras=cams, lights=lights)

            images_noise = dss_renderer(render_structure, cameras=cams, lights=lights)
            images_noise = torch.clamp(images_noise, min=0, max=1)
            # os.makedirs(os.path.join(output_noise, str(n)), exist_ok=True)
            # os.makedirs(os.path.join(output_normal, str(n)), exist_ok=True)
            imageio.imwrite(os.path.join(output_noise, "NOS_%06d.png" % idx),(images_noise[0][..., :3].detach().cpu().numpy() * 255.0).astype('uint8'))
            imageio.imwrite(os.path.join(output_normal,"CLN_%06d.png" % idx),(images_clean[0][..., :3].detach().cpu().numpy() * 255.0).astype('uint8'))
            idx += 1


if __name__ == '__main__':
  fire.Fire(main)