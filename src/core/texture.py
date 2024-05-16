"""
PointTexture class

Inputs should be fragments (including point location,
normals and other features)
Output is the color per point (doesn't have the blending step)

diffuse shader
specular shader
neural shader
"""
import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer.cameras import OrthographicCameras
from pytorch3d.renderer.lighting import PointLights, DirectionalLights

from .cloud import PointClouds3D
from .. import logger_py
from ..utils import gather_batch_to_packed


__all__ = ["SurfaceSplattingShader"]



def apply_lighting(points, normals, lights, cameras, specular=True, shininess=64):
    """
    Args:
        points: torch tensor of shape (P, 3).
        normals: torch tensor of shape (P, 3).
        lights: instance of the Lights class.
        cameras: instance of the Cameras class.
        shininess: scalar for the specular coefficient.
        specular: (bool) whether to add the specular effect
    Returns:
        ambient_color: same shape as materials.ambient_color
        diffuse_color: same shape as the input points
        specular_color: same shape as the input points
    """
    assert points.ndim == 2 and normals.ndim == 2
    assert points.shape[-1] == 3 and normals.shape[-1] == 3

    diffuse_color = lights.diffuse(normals=normals, points=points)  
    specular_color = torch.tensor([[0,0,0]]).to(points.device)
    if specular :
        specular_color = lights.specular(normals=normals,points=points, camera_position=cameras.get_camera_center(), shininess=shininess,)
    ambient_color = lights.ambient_color

    
    assert ambient_color.ndim== 2 and ambient_color.shape[1] > 1
    assert specular_color.ndim== 2 and diffuse_color.ndim== 2
    assert specular_color.shape[-1] == 3 and diffuse_color.shape[-1] == 3 and ambient_color.shape[-1] == 3

    return ambient_color, diffuse_color, specular_color


class SurfaceSplattingShader(nn.Module):
    def __init__(self, device="cpu", cameras=None, lights=None):
        super().__init__()
        self.lights = lights
        self.cameras = cameras

    def forward(self, pointclouds, verbose=False, shininess=64, **kwargs) -> PointClouds3D:
        """
        Args:
            pointclouds : src.core.cloud.Pointclouds3D
        Returns:
            pointclouds : src.core.cloud.Pointclouds3D with lighting set to RGB colors
        """
        cameras = kwargs.get("cameras", self.cameras).to(pointclouds.device)
        lights = kwargs.get("lights", self.lights)
        if lights is None :
            lights = self.lights
        else:
            self.lights = lights
            
        specular = kwargs.get("specular", False)
        assert pointclouds.isempty() == False

        if len(kwargs["cameras"]) != len(pointclouds) and len(pointclouds) == 1:
            pointclouds = pointclouds.extend(len(kwargs["cameras"]))
            
        vert_to_cloud_idx = pointclouds.packed_to_cloud_idx()
        if len(pointclouds) > 1:
            lights = lights.clone().gather_props(vert_to_cloud_idx)
            cameras = cameras.clone().gather_props(vert_to_cloud_idx)

        points = pointclouds.points_packed()
        normals = pointclouds.normals_packed()
        rgb = pointclouds.features_packed()[:, :3]
        assert normals is not None and rgb is not None

        # Calculate the illumination at each point
        ambient, diffuse, specular = apply_lighting(points, normals, lights, cameras, shininess=shininess, specular=specular )
        points_colors_shaded = (rgb * (ambient + diffuse) + specular)
        points_colors_shaded = points_colors_shaded.clamp(0,1)
        assert points_colors_shaded.min() >= 0 and points_colors_shaded.max() <= 1

        # Update point cloud
        pointclouds_colored = pointclouds.clone()
        pointclouds_colored.update_rgb_(points_colors_shaded)
        return pointclouds_colored