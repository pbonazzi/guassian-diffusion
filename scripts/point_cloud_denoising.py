"""
Pipeline to denoise point clouds from their renderings
Training Shapes : https://polybox.ethz.ch/index.php/s/sCC6pYAngLa9d8Y
Test Shapes : https://polybox.ethz.ch/index.php/s/wPxFlfxKbgVkfNb

Noise from a range between +- 1% of the bounding box diagonal of the shapes is added to training pointcloud.
PointClouds are then rendered using DSS rasterizer : "https://github.com/yifita/DSS"
"""

import os
from tqdm import tqdm
import numpy as np
import imageio.v2 as imageio
import torch
import pdb
import matplotlib.pyplot as plt

# custom 
from dependencies.SD.pipe_denoising import StableDiffusionImg2ImgDenoisingPipeline
from src.misc.imageFilters import Pix2PixDenoising, Pix2PixInitialize

import torch
import os
import glob
import pdb
import math
import copy
import cv2 as cv
import open3d as o3d
import numpy as np
import pytorch3d
from tqdm import tqdm
from pytorch3d.renderer import TexturesVertex, RasterizationSettings, FoVPerspectiveCameras, MeshRendererWithFragments, MeshRasterizer, BlendParams
from pytorch3d.renderer.lighting import PointLights, DirectionalLights, AmbientLights
from src.core.shader import HardFlatShaderwithoutSpecular
from src.core.camera import CameraSampler
from utils.common import load_structure_from_config
import imageio.v2 as imageio
from PIL import Image
from src.misc.imageFilters import Pix2PixDenoising, Pix2PixInitialize

from pytorch3d.ops import sample_points_from_meshes
from utils.config import create_renderer, create_lights


from src.training.clip_encoder import get_embed_fn, resize_img_for_clip
from src.training.losses import SemanticConsistencyLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/data/storage/bpietro/huggingface/diffusers/model-stable-diffusion-2-1-denoising-01"


#pix2pix = Pix2PixInitialize()

def load_noise_data(path, znear, zfar, image_size, num_cams, num_of_points, input_type, input_format, noise, save_path):
    from src.core.texture import SurfaceSplattingShader
    from src.core.cloud import PointClouds3D

    diffusion = StableDiffusionImg2ImgDenoisingPipeline.from_pretrained(model_path, revision=None, torch_dtype=torch.float32,).to(device)

    sem_cons_loss = SemanticConsistencyLoss(reduction='mean')
    clip_embed = get_embed_fn( model_type="clip_vit", device=device, num_layers=-1, clip_cache_root=os.path.expanduser("~/.cache/clip/"))


    raster_settings = {
                "backface_culling": False,
                "cutoff_threshold": 1.0,
                "depth_merging_threshold": 0.05,
                "Vrk_invariant": True,
                "Vrk_isotropic": False,
                "knn_points": 5,
                "radii_backward_scaler": 10,
                "image_size": image_size,
                "points_per_pixel": 12,
                "bin_size": None,
                "max_points_per_bin": None,
                "clip_pts_grad": 0.05,
                "antialiasing_sigma" : 1.0,
                "frnn_radius": -1 , 
            }
    
    dss_renderer = create_renderer(raster_settings, background_color=[0,0,0])
    texture = SurfaceSplattingShader()
    H = W = image_size

    # clean 
    _, _, verts, normals, colors = load_structure_from_config(path, input_type, input_format, num_of_points, np.array([0,0,0]), "grey", image_size, device, special_init=False)
    pts = torch.cat([verts, colors, normals], dim=1) # clean
    pc_structure = PointClouds3D(points=[verts], normals=[normals], features=[colors])
    pc = o3d.cuda.pybind.geometry.PointCloud(points=o3d.cuda.pybind.utility.Vector3dVector(verts.cpu()) )
    bb = pc.get_oriented_bounding_box()
    diag = math.dist(bb.get_max_bound(),  bb.get_min_bound())

    # noise 
    level = 0
    level = 0.01 if noise == "Noise001" else  level
    level = 0.02 if noise == "Noise0007" else  level
    level = 0.03 if noise == "Noise0005" else   level
    level = 0.04 if noise == "Noise0003" else   level
    # _, _, verts, normals, colors = load_structure_from_config(path.replace("Uniform", noise), input_type, input_format, num_of_points, np.array([0,0,0]), "grey", image_size, device, special_init=False)
    # noise_structure = PointClouds3D(points=[verts], normals=[normals], features=[colors])
    new_structure = copy.deepcopy(pc_structure)
    offset_points = torch.from_numpy(np.random.uniform(low=-diag*level, high=diag*level, size=(new_structure.points_packed().shape))).to(device)
    noise_structure = new_structure.offset(offset_points.float())
    pts_noise = torch.cat([verts+offset_points, colors, normals], dim=1)

    # sample camera positions
    intrinsic = [[1, 0, 0.5 * W],[0, 1, 0.5 * H],[0, 0, 1]]
    camera_sampler = CameraSampler(continuous_views=False, num_cams_total=num_cams, num_cams_batch=1, distance_range=torch.tensor(((znear,  zfar),)),  sort_distance=True,camera_type=FoVPerspectiveCameras)

    poses = torch.zeros(num_cams, 4, 4).to(device)
    imgs, locations = [], []

    noise_sem, denoise_sem = 0, 0

    for i, cams in tqdm(enumerate(camera_sampler), desc="Image from Mesh"):
        # camera
        cams = cams.to(device)
        poses[i] = cams.get_world_to_view_transform().get_matrix()[0].to(device)

        # lights
        location = cams.get_camera_center()
        locations.append(location)
        lights = create_lights(light_type="point" , location=location, device=device)

        # noise
        render_structure = texture(copy.deepcopy(pc_structure), cameras=cams, lights=lights)
        images = dss_renderer(render_structure, cameras=cams, verbose=False)
        gt_sem_ = clip_embed(resize_img_for_clip(images[0][..., :3]))[0]

        if save_path is not None:
            os.makedirs(os.path.join(save_path, "clean"), exist_ok=True)
            src = (images[0][..., :3]*255).cpu().numpy().astype("uint8")
            # tmp = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
            # _, alpha = cv.threshold(tmp, 0, 255, cv.THRESH_BINARY)
            # r, g, b = cv.split(src)
            # rgba = [r, g, b, alpha]
            # dst = cv.merge(rgba, 4)
            cv.imwrite(os.path.join(save_path, "clean", "%06d.png"%i), src)


        # noise
        render_structure = texture(copy.deepcopy(noise_structure), cameras=cams, lights=lights)
        images = dss_renderer(render_structure, cameras=cams, verbose=False)

        noise_sem_ = clip_embed(resize_img_for_clip(images[0][..., :3]))[0]
        noise_sem += torch.nn.CosineSimilarity()(gt_sem_[:1], noise_sem_[:1]).item()
                
        if save_path is not None:
            os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
            src = (images[0][..., :3]*255).cpu().numpy().astype("uint8")
            # tmp = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
            # _, alpha = cv.threshold(tmp, 0, 255, cv.THRESH_BINARY)
            # r, g, b = cv.split(src)
            # rgba = [r, g, b, alpha]
            # dst = cv.merge(rgba, 4)
            cv.imwrite(os.path.join(save_path, "images", "%06d.png"%i), src)

        # pix2pix
        # image_pix2pix = Pix2PixDenoising(images[..., :3], pix2pix)[0]
        # if save_path is not None:
        #     os.makedirs(os.path.join(save_path, "pix2pix"), exist_ok=True)
        #     src = (image_pix2pix*255).cpu().numpy().astype("uint8")
        #     cv.imwrite(os.path.join(save_path, "pix2pix", "%06d.png"%i), src)

        # apply diffusion
        # torch.cuda.empty_cache()
        prepare_img = images[..., :3].permute(0,3,2,1).permute(0,1,3,2).to(device)
        images = diffusion(prepare_img).permute(0, 2, 3, 1)
        images = torch.clamp(images, min=0, max=1)

        denoise_sem_ = clip_embed(resize_img_for_clip(images[0][..., :3]))[0]
        denoise_sem += torch.nn.CosineSimilarity()(gt_sem_[:1], denoise_sem_[:1]).item()
              
        if save_path is not None:
            os.makedirs(os.path.join(save_path, "denoised"), exist_ok=True)
            src = (images[0][..., :3]*255).cpu().numpy().astype("uint8")
            # tmp = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
            # _, alpha = cv.threshold(tmp, 0, 255, cv.THRESH_BINARY)
            # r, g, b = cv.split(src)
            # rgba = [r, g, b, alpha]
            # dst = cv.merge(rgba, 4)
            cv.imwrite(os.path.join(save_path, "denoised", "%06d.png"%i), src)

            # src[..., :3] = alpha*src[..., :3]
            # imageio.imwrite(os.path.join(save_path, "denoised", "%06d.png"%i), src)
        
        imgs.append(images[0])


    denoise_sem /= num_cams
    noise_sem /= num_cams

    if save_path is not None:
        numpy_poses = poses.cpu().numpy()
        poses_2d = numpy_poses.reshape( numpy_poses.shape[0], -1)
        numpy_locations = torch.stack(locations).squeeze(1).cpu().numpy()

        np.savetxt(os.path.join(save_path,'locations.txt'), numpy_locations.astype("float"))
        np.savetxt(os.path.join(save_path,'poses.txt'), poses_2d.astype("float"))
        np.savetxt(os.path.join(save_path,'intrinsic.txt'),  intrinsic)
        np.savetxt(os.path.join(save_path,'pts.txt'), pts.cpu().numpy())
        np.savetxt(os.path.join(save_path,'noise_pts.txt'), pts_noise.cpu().numpy())


    #return pts, pts_noise, torch.stack(imgs), poses, intrinsic, locations
    return denoise_sem, noise_sem


fig, ax = plt.subplots()
x, y_den, y_noi = ["1.0%", "2.0%", "3.0%", "4.0%"], [], []
for n in ["Noise001", "Noise0007", "Noise0005", "Noise0003"]:
    denoise_sem, noise_sem = load_noise_data("/data/storage/bpietro/datasets/scut/Uniform/complex/aphrodite.txt", znear=1.5, zfar=1.7, image_size=512, num_cams=20, num_of_points=0, input_type="point", input_format="xyzn", noise=n, save_path="/data/storage/bpietro/datasets/mesh/aphrodite_"+(n))
    y_den.append(denoise_sem)
    y_noi.append(noise_sem)

ax.plot(x, y_den, label="Denoised")
ax.plot(x, y_noi, label="Noise")
ax.legend()
plt.xlabel('Point Noise', fontsize=18)
plt.ylabel('Semantic Similarity', fontsize=16)
fig.savefig('poses.png')
plt.close(fig) 