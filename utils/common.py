import torch, pdb
import torch.nn.functional as F
import os
import math
import numpy as np
import open3d as o3d
import random
from src.misc.visualize import animate_points, animate_mesh, figures_to_html
from pytorch3d.renderer.lighting import PointLights, DirectionalLights
from src import logger_py
from src.core.camera import look_at_rotation
from src.core.cloud import PointClouds3D

import pytorch3d
from pytorch3d.transforms import Rotate
from pytorch3d.renderer import (
    RasterizationSettings,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    MeshRendererWithFragments,
    
    HardFlatShader,
    HardGouraudShader,
    HardPhongShader,
    SoftPhongShader,

    PointsRasterizationSettings,
    PointsRenderer,
    TexturesVertex,
    TexturesAtlas,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
from pytorch3d.renderer.lighting import PointLights, DirectionalLights
from pytorch3d.renderer import Textures
from pytorch3d.renderer.blending import BlendParams, hard_rgb_blend
from pytorch3d.renderer.mesh.shading import flat_shading


from pytorch3d.io import load_obj
from pytorch3d.ops import eyes, sample_points_from_meshes
from pytorch3d.structures import Pointclouds

def normalize_unit_sphere(verts):

    # normalize to unit sphere
    vert_center = torch.mean(verts, dim=0)
    vert_scale = torch.norm(verts, dim=1).max()
    verts_clone = (verts - vert_center)/vert_scale
    return verts_clone


def load_structure_from_config(path, input_type, input_format, num_of_points, xyz_rotation, color, image_size, device, special_init=False):

        if path in ["bunny", "armadillo"]:

            if path == "bunny":
                mesh_path = o3d.data.BunnyMesh()

            elif path == "armadillo":
                mesh_path = o3d.data.ArmadilloMesh()
            
            mesh = o3d.io.read_triangle_mesh(mesh_path.path)

            verts = torch.tensor(np.asarray(mesh.vertices)).float().to(device)

            if xyz_rotation.sum() > 0:
                rotat_mat = pytorch3d.transforms.axis_angle_to_matrix(torch.tensor(xyz_rotation).to(device))
                transform = pytorch3d.transforms.Rotate(rotat_mat)
                verts = transform.transform_points(verts)
                
            faces = torch.tensor(np.asarray(mesh.triangles)).to(device)
            
            colors = get_point_colors_from_verts(verts, color_type=color).float().to(device)
            textures = TexturesVertex(verts_features=colors.unsqueeze(0))
            
            structure = pytorch3d.structures.Meshes(verts=[normalize_unit_sphere(verts)], faces=[faces], textures=textures).to(device=device)         
            verts, normals, colors = sample_points_from_meshes( structure, num_samples=num_of_points, return_normals=True, return_textures=True)
            verts, normals, colors= verts[0], normals[0], colors[0]
            
            raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=5, bin_size=0,  max_faces_per_bin=0)


        elif  input_type == "point":

            verts, normals, colors = load_pointcloud_att(path, input_format, xyz_rotation, color, device=device)
            structure = PointClouds3D(points=[verts], normals=[normals], features=[colors])
            raster_settings = PointsRasterizationSettings( image_size=image_size, radius = 0.003,   bin_size=None, points_per_pixel = 10 )
            

        elif  input_type == "mesh" and input_format in ["ply", "off"]:
            
            verts, verts_idx, colors = load_mesh_att(path, xyz_rotation, color, device=device)
            structure = pytorch3d.structures.Meshes(verts=[verts], faces=[verts_idx], textures= Textures(verts_rgb=colors)).to(device=device)
            
            verts, normals, colors = sample_points_from_meshes( structure, num_samples=num_of_points, return_normals=True, return_textures=True)
            verts, normals, colors= verts[0], normals[0], colors[0]
            
            raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=5, bin_size=0,  max_faces_per_bin=0)

        elif  input_type == "mesh" and input_format == "obj":

            if False:
                verts, faces, aux = load_obj(path, device=device, load_textures=True, create_texture_atlas=True,texture_atlas_size=8, texture_wrap=None, )
                textures = TexturesAtlas(atlas=[aux.texture_atlas])
            else:
                verts, faces, aux = load_obj(path, device=device, load_textures=False, create_texture_atlas=False)
                colors : torch.float32 = get_point_colors_from_verts(verts, color_type=color).to(device)
                textures = TexturesVertex(verts_features=colors.unsqueeze(0))

            if xyz_rotation.sum() > 0:
                rotat_mat = pytorch3d.transforms.axis_angle_to_matrix(torch.tensor(xyz_rotation).to(device))
                transform = pytorch3d.transforms.Rotate(rotat_mat)
                verts = transform.transform_points(verts)
            
            structure = pytorch3d.structures.Meshes(verts=[normalize_unit_sphere(verts)], faces=[faces.verts_idx], textures=textures, )
            verts, normals, colors = sample_points_from_meshes( structure, num_samples=num_of_points, return_normals=True, return_textures=True)
            verts, normals, colors= verts[0], normals[0], colors[0]

            # renderer settings
            raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=5, bin_size=0,  max_faces_per_bin=0)
        else:

            pdb.set_trace()
            raise NotImplementedError


        if special_init:
            return verts, normals, colors
        else:
            return structure, raster_settings, verts, normals, colors


def load_pointcloud_att(path, input_format, xyz_rotation, color, device):
    o3d_point_cloud = o3d.io.read_point_cloud(path, format=input_format)
    # verts  = normalize_unit_sphere(torch.tensor(np.asarray(o3d_point_cloud.points), dtype=torch.float32).to(device))
    verts  = torch.tensor(np.asarray(o3d_point_cloud.points), dtype=torch.float32).to(device)
    if xyz_rotation.sum() > 0:
        o3d_point_cloud.rotate(o3d_point_cloud.get_rotation_matrix_from_xyz( xyz_rotation), center=o3d_point_cloud.get_center())

    if o3d_point_cloud.has_normals() :
        o3d_point_cloud.estimate_normals()
        normals = torch.tensor(np.asarray(o3d_point_cloud.normals), dtype=torch.float32).to(device)
    else:
        normals  = torch.tensor(np.asarray(o3d_point_cloud.normals), dtype=torch.float32).to(device)

    if o3d_point_cloud.has_colors() :
        colors = torch.tensor(np.asarray(o3d_point_cloud.points), dtype=torch.float32).to(device)
    else:
        colors = get_point_colors_from_verts(verts, color_type=color).to(device)

    return verts, normals, colors


def load_mesh_att(path, xyz_rotation, color, device):

    mesh = o3d.io.read_triangle_mesh(path)

    mesh.compute_vertex_normals()
    
    if xyz_rotation.sum() > 0:
        mesh.rotate(mesh.get_rotation_matrix_from_xyz(xyz_rotation), center=(0, 0, 0))
    
    verts = normalize_unit_sphere(torch.tensor(mesh.vertices, dtype=torch.float32)).to(device)    
    verts_idx = torch.tensor(mesh.triangles, dtype=torch.int).to(device)
    
    if mesh.has_vertex_colors():
        colors  = torch.tensor(mesh.vertex_colors).unsqueeze(0).to(device).float()
    else:
        colors = get_point_colors_from_verts(verts, color_type=color).unsqueeze(0).to(device)

    return verts, verts_idx, colors


def get_rays(H, W, K, c2w):
    device = c2w.device
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i, device=device)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def create_animation(pts_dir, show_max=-1):
    figs = []
    # points
    pts_files = [f for f in os.listdir(pts_dir) if 'pts' in f and f[-4:].lower() in ('.ply', 'obj')]
    if len(pts_files) == 0:
        logger_py.info("Couldn't find '*pts*' files in {}".format(pts_dir))
    else:
        pts_files.sort()
        if show_max > 0:
            pts_files = pts_files[::max(len(pts_files) // show_max, 1)]
        pts_names = list(map(lambda x: os.path.basename(x)
                             [:-4].split('_')[0], pts_files))
        pts_paths = [os.path.join(pts_dir, fname) for fname in pts_files]
        fig = animate_points(pts_paths, pts_names)
        figs.append(fig)
    # mesh
    mesh_files = [f for f in os.listdir(pts_dir) if 'mesh' in f and f[-4:].lower() in ('.ply', '.obj')]

    if len(mesh_files) == 0:
        logger_py.info("Couldn't find '*mesh*' files in {}".format(pts_dir))
    else:
        mesh_files.sort()
        if show_max > 0:
            mesh_files = mesh_files[::max(len(mesh_files) // show_max, 1)]
        mesh_names = list(map(lambda x: os.path.basename(x)
                              [:-4].split('_')[0], mesh_files))
        mesh_paths = [os.path.join(pts_dir, fname) for fname in mesh_files]
        fig = animate_mesh(mesh_paths, mesh_names)
        figs.append(fig)

    save_html = os.path.join(pts_dir, 'animation.html')
    os.makedirs(os.path.dirname(save_html), exist_ok=True)
    figures_to_html(figs, save_html)

def get_point_colors_from_verts(verts, color_type=""):
    assert color_type in ["from_verts", "from_verts_n", "black", "grey", "blue"]

    if color_type == "from_verts":
        colors = (verts - verts.min())/(verts.max()- verts.min())
    elif color_type == "black":
        colors = torch.zeros_like(verts)
    elif color_type == "grey":
        colors = torch.zeros_like(verts)+0.5
    elif color_type == "blue":
        colors = torch.zeros_like(verts)
        colors[:,2] = torch.ones_like(colors[:,2])
    elif color_type == "from_verts_n":
        verts = 1 - verts
        colors = (verts - verts.min())/(verts.max()- verts.min())
    
    assert colors.min() >= 0 and colors.max() <= 1
    return colors 

def get_tri_color_lights_for_view(cams, verbose=False, has_specular=False, point_lights=True):
    """
    Create RGB lights direction in the half dome
    The direction is given in the same coordinates as the pointcloud
    Args:
        cams
    Returns:
        Lights with three RGB light sources (B: right, G: left, R: bottom)
    """
    # from src.core.lighting import PointLights, DirectionalLights

    # init
    elev = math.pi / 180.0 * torch.tensor(((30, 30, 30),),device=cams.device)
    azim = math.pi / 180.0 * torch.tensor(((-60, 60, 180),),device=cams.device)
    x = torch.cos(elev) * torch.sin(azim)
    y = torch.sin(elev)
    z = torch.cos(elev) ** 2
    light_directions = torch.stack([-x, y, z], dim=-1)
    
    # rotate
    cam_pos = cams.get_camera_center()
    R = look_at_rotation(torch.zeros_like(cam_pos), at=F.normalize(torch.cross(cam_pos, torch.rand_like(cam_pos)), dim=-1), up=cam_pos)
    light_directions = Rotate(R=R.transpose(1,2), device=cams.device).transform_points(light_directions)
    
    # colors
    ambient_color = torch.FloatTensor((((0.5, 0.5, 0.5), ), ))
    diffuse_color = torch.FloatTensor((((0.0, 0.0, 0.8), (0.0, 0.8, 0.0), (0.8, 0.0, 0.0), ), ))
    specular_color = torch.FloatTensor(((0, 0, 0), (0, 0, 0), (0, 0, 0), ), )
    if has_specular:
        specular_color = 0.15 * diffuse_color
        diffuse_color *= 0.85

    assert ambient_color.min() >= 0 and ambient_color.max() <= 1
    assert specular_color.min() >= 0 and specular_color.max() <= 1
    assert diffuse_color.min() >= 0 and diffuse_color.max() <= 1


    if not point_lights:
        return DirectionalLights(ambient_color=ambient_color, diffuse_color=diffuse_color,  specular_color=specular_color, direction=light_directions)
    else:
        return PointLights(ambient_color=ambient_color[0], diffuse_color=diffuse_color[:,0,:]+diffuse_color[:,1,:], specular_color=specular_color[:1], location=light_directions[:,0,:]*5)


def get_light_for_view(cams, point_lights, has_specular):

    # init
    elev = torch.tensor(((math.pi / 180.0*40),), dtype=torch.float, device=cams.device)
    azim = torch.tensor(((math.pi / 180.0*180),), dtype=torch.float, device=cams.device)

    x = torch.cos(elev) * torch.sin(azim)
    y = torch.sin(elev)
    z = torch.cos(elev) ** 2
    light_directions = torch.stack([x, y, z], dim=-1)

    # tri-color lights
    ambient_color = torch.FloatTensor((((1,1,1),),))
    diffuse_color = torch.FloatTensor((((0.2, 0.2, 0.2),),))
    if has_specular:
        specular_color = 0.15 * diffuse_color
        diffuse_color *= 0.85
    else:
        specular_color = (((0, 0, 0),),)

    assert ambient_color.min() >= 0 and ambient_color.max() <= 1
    assert specular_color.min() >= 0 and specular_color.max() <= 1
    assert diffuse_color.min() >= 0 and diffuse_color.max() <= 1

    # transform from camera to world
    light_directions = cams.get_world_to_view_transform().inverse().transform_points(light_directions)
    if not point_lights:
        lights = DirectionalLights(ambient_color=ambient_color, diffuse_color=diffuse_color, specular_color=specular_color, direction=light_directions)
    else:
        lights = PointLights(ambient_color=ambient_color, diffuse_color=diffuse_color, specular_color=specular_color, location=light_directions*5)
    
    return lights