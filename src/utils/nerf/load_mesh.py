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


from pytorch3d.ops import sample_points_from_meshes
from utils.config import create_renderer, create_lights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def normalize_unit_sphere(verts):

    # normalize to unit sphere
    vert_center = torch.mean(verts, dim=0)
    vert_scale = torch.norm(verts, dim=1).max()
    verts_clone = (verts - vert_center)/vert_scale
    return verts_clone


def load_mesh_data(path, znear, zfar, image_size, num_cams, num_of_points, input_type, input_format, save_path="/data/storage/bpietro/datasets/mesh/kangaroo_dss"):

    H = W = image_size

    # clean 
    structure, raster_settings, verts, normals, colors = load_structure_from_config(path, input_type, input_format, num_of_points, torch.tensor(np.array([0,0,0])).float(), "grey", image_size, device, special_init=False)
    pts = torch.cat([verts, colors, normals], dim=1)

    # sample camera positions
    intrinsic = [[1, 0, 0.5 * W],[0, 1, 0.5 * H],[0, 0, 1]]
    camera_sampler = CameraSampler(continuous_views=True, num_cams_total=num_cams, num_cams_batch=1, distance_range=torch.tensor(((znear,  zfar),)),  sort_distance=True,camera_type=FoVPerspectiveCameras)

    poses = torch.zeros(num_cams, 4, 4).to(device)
    imgs, locations = [], []

    for i, cams in tqdm(enumerate(camera_sampler), desc="Image from Mesh"):
        # camera
        cams = cams.to(device)
        poses[i] = cams.get_world_to_view_transform().get_matrix()[0].to(device)

        # lights
        location = cams.get_camera_center()
        locations.append(location)
        lights = create_lights(light_type="point", location=location, device=device, cams=cams)

        # renderer
        shader = HardFlatShaderwithoutSpecular( device=device,  cameras=cams, lights=lights, blend_params= BlendParams(background_color = (1.0, 1.0, 1.0)))
        pyd_renderer = MeshRendererWithFragments(rasterizer=MeshRasterizer(cameras=cams, raster_settings=raster_settings),shader=shader)

        images, fragments = pyd_renderer(structure, cameras=cams, lights=lights)
        images = torch.clamp(images, min=0, max=1)
        if save_path is not None:
            os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
            imageio.imwrite(os.path.join(save_path, "images", "%06d.png"%i), (images[0][..., :3]*255).cpu().numpy().astype("uint8"))
        imgs.append(images[0])

    if save_path is not None:
        numpy_poses = poses.cpu().numpy()
        poses_2d = numpy_poses.reshape( numpy_poses.shape[0], -1)
        numpy_locations = torch.stack(locations).squeeze(1).cpu().numpy()

        np.savetxt(os.path.join(save_path,'locations.txt'), numpy_locations.astype("float"))
        np.savetxt(os.path.join(save_path,'poses.txt'), poses_2d.astype("float"))
        np.savetxt(os.path.join(save_path,'intrinsic.txt'),  intrinsic)
        np.savetxt(os.path.join(save_path,'pts.txt'), pts.cpu().numpy())


    return pts, torch.stack(imgs), poses, intrinsic, numpy_locations



def load_noise_data(path, znear, zfar, image_size, num_cams, num_of_points, input_type, input_format, diffusion, save_path="/data/storage/bpietro/datasets/mesh/armadillo_noise"):
    from src.core.texture import SurfaceSplattingShader
    from src.core.cloud import PointClouds3D

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
    
    dss_renderer = create_renderer(raster_settings, background_color=[1,1,1])
    texture = SurfaceSplattingShader()
    H = W = image_size

    # clean 
    _, _, verts, normals, colors = load_structure_from_config(path, input_type, input_format, num_of_points, np.array([0,0,0]), "grey", image_size, device, special_init=False)
    pts = torch.cat([verts, colors, normals], dim=1) # clean
    structure = PointClouds3D(points=[verts], normals=[normals], features=[colors])
    
    # noise 
    pc = o3d.cuda.pybind.geometry.PointCloud(points=o3d.cuda.pybind.utility.Vector3dVector(verts.cpu()) )
    bb = pc.get_oriented_bounding_box()
    diag = math.dist(bb.get_max_bound(),  bb.get_min_bound())
    noise_structure = copy.deepcopy(structure).to(device)
    offset_points = torch.from_numpy(np.random.uniform(low=-diag*0.01, high=diag*0.01, size=(noise_structure.points_packed().shape))).to(device)
    noise_structure = noise_structure.offset(offset_points.float())
    pts_noise = torch.cat([verts+offset_points, colors, normals], dim=1)

    # sample camera positions
    intrinsic = [[1, 0, 0.5 * W],[0, 1, 0.5 * H],[0, 0, 1]]
    camera_sampler = CameraSampler(continuous_views=False, num_cams_total=num_cams, num_cams_batch=1, distance_range=torch.tensor(((znear,  zfar),)),  sort_distance=True,camera_type=FoVPerspectiveCameras)

    poses = torch.zeros(num_cams, 4, 4).to(device)
    imgs, locations = [], []

    for i, cams in tqdm(enumerate(camera_sampler), desc="Image from Mesh"):
        # camera
        cams = cams.to(device)
        poses[i] = cams.get_world_to_view_transform().get_matrix()[0].to(device)

        # lights
        location = cams.get_camera_center()
        locations.append(location)
        lights = create_lights(light_type="point" , location=location, device=device)

        # noise
        render_structure = texture(copy.deepcopy(noise_structure), cameras=cams, lights=lights)
        images = dss_renderer(render_structure, cameras=cams, verbose=False)

        if save_path is not None:
            os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
            imageio.imwrite(os.path.join(save_path, "images", "%06d.png"%i), (images[0][..., :3]*255).cpu().numpy().astype("uint8"))

        # apply diffusion
        prepare_img = images[..., :3].permute(0,3,2,1).permute(0,1,3,2).to(device)
        images = diffusion(prepare_img).permute(0, 2, 3, 1)

        images = torch.clamp(images, min=0, max=1)
        if save_path is not None:
            os.makedirs(os.path.join(save_path, "denoised"), exist_ok=True)
            imageio.imwrite(os.path.join(save_path, "denoised", "%06d.png"%i), (images[0][..., :3]*255).cpu().numpy().astype("uint8"))
        
        imgs.append(images[0])


    if save_path is not None:
        numpy_poses = poses.cpu().numpy()
        poses_2d = numpy_poses.reshape( numpy_poses.shape[0], -1)
        numpy_locations = torch.stack(locations).squeeze(1).cpu().numpy()

        np.savetxt(os.path.join(save_path,'locations.txt'), numpy_locations.astype("float"))
        np.savetxt(os.path.join(save_path,'poses.txt'), poses_2d.astype("float"))
        np.savetxt(os.path.join(save_path,'intrinsic.txt'),  intrinsic)
        np.savetxt(os.path.join(save_path,'pts.txt'), pts.cpu().numpy())
        np.savetxt(os.path.join(save_path,'noise_pts.txt'), pts_noise.cpu().numpy())


    return pts, pts_noise, torch.stack(imgs), poses, intrinsic, locations







def load_image_data(path, load_noise=False):

    import timeit

    start = timeit.timeit()
    print("Start Loading Data")

    noise_pts, name = None, "images"
    if load_noise :
        noise_pts= np.loadtxt(os.path.join(path,'noise_pts.txt'))
        name = "denoised"

    imgs = []
    for file in tqdm(sorted(glob.glob(os.path.join(path, name, "*.png")))):
        im = imageio.imread(file,pilmode="RGBA")
        tmp = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        _, alpha = cv.threshold(tmp, 10, 255, cv.THRESH_BINARY)
        im[...,3] = alpha
        imgs.append(im)

    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.loadtxt(os.path.join(path,'poses.txt'))
    poses = poses.reshape(poses.shape[0], poses.shape[1] // 4, 4)
  
    intrinsic = np.loadtxt(os.path.join(path,'intrinsic.txt'))
    locations = np.loadtxt(os.path.join(path,'locations.txt'))

    pts= np.loadtxt(os.path.join(path,'pts.txt'))
    
    end = timeit.timeit()
    print("End Loading Data")
    print(end - start)

    return pts, noise_pts, imgs, poses, intrinsic, locations

