import torch
from pytorch3d.renderer.cameras import (FoVPerspectiveCameras)
from typing import Optional, Sequence, Tuple
import math
import torch.nn.functional as F
import pdb
from pytorch3d.renderer import convert_to_tensors_and_broadcast


class CameraSampler(object):
    """
    create camera transformations looking at the origin of the coordinate
    from varying distance

    Attributes:
        R, T: (num_cams_total, 3, 3) and (num_cams_total, 3)
        camera_type (Class): class to create a new camera
        camera_params (dict): camera parameters to call camera_type
            (besides R, T)
    """

    def __init__(self, num_cams_total, num_cams_batch,
                 distance_range=(5, 10), sort_distance=True,
                 return_cams=True, continuous_views=False,
                 camera_type=FoVPerspectiveCameras, camera_params=None):
        """
        Args:
            num_cams_total (int): the total number of cameras to sample
            num_cams_batch (int): the number of cameras per iteration
            distance_range (tensor or list): (num_cams_total, 2) or (1, 2) the range of camera distance for uniform sampling
            sort_distance: sort the created camera transformations by the distance in ascending order
            return_cams (bool): whether to return camera instances or just the R,T
            camera_type (class): camera type from pytorch3d.renderer.cameras
            camera_params (dict): camera parameters besides R, T
        """
        self.num_cams_batch = num_cams_batch
        self.num_cams_total = num_cams_total

        self.sort_distance = sort_distance
        self.camera_type = camera_type
        self.camera_params = {} if camera_params is None else camera_params

        # create camera locations
        distance_scale = distance_range[:, -1] - distance_range[:, 0]
        distances = torch.rand(num_cams_total) * distance_scale + distance_range[:, 0]
        if sort_distance:
            distances, _ = distances.sort(descending=True)

        # if continuous_views:
        # num_smooth = 8
        # elev = torch.linspace(180, 360, num_cams_total)
        # azim = torch.linspace(-180, 179, num_cams_total)
        
        # azim = torch.cat([azim, torch.linspace(1, 360, num_cams_total-num_smooth)])
        # elev = torch.cat([elev, torch.linspace(-120, 120, num_cams_total-num_smooth)])
        
        # azim = torch.cat([azim, torch.rand(num_cams_total-num_smooth) * 360 - 180])
        # elev = torch.cat([elev, torch.rand(num_cams_total-num_smooth) * 180 - 90])
        
        # else:
        azim = torch.rand(num_cams_total) * 360 - 180
        elev = torch.rand(num_cams_total) * 180 - 90

        # at = torch.rand((num_cams_total, 3)) * 0.1 - 0.05
        
        # if continuous_views:
        #     # make a trajectory of image views for validation set
        #     indexes = list(range(num_cams_total))
        #     indexes = indexes[num_cams_total%2::2] + indexes[::-2]
        #     elev = (elev.sort(descending=True)[0])[indexes]
        #     azim = (azim.sort(descending=True)[0])[indexes]
        at = torch.zeros((num_cams_total, 3))


        self.R, self.T = look_at_view_transform(distances, elev, azim, at=at, degrees=True)

        if not torch.is_tensor(self.R):
            self.R = torch.tensor(self.R, dtype=torch.float32)
            self.T = torch.tensor(self.T, dtype=torch.float32)

        self._idx = 0

    def __len__(self):
        return (self.R.shape[0] + self.num_cams_batch - 1) // self.num_cams_batch

    def __iter__(self):
        return self
    
    def __restart__(self):
        self._idx = 0

    def __next__(self):
        if self._idx >= len(self):
            raise StopIteration
        start_idx = self._idx * self.num_cams_batch
        end_idx = min(start_idx + self.num_cams_batch, self.R.shape[0])
        cameras = self.camera_type(R=self.R[start_idx:end_idx],T=self.T[start_idx:end_idx],**self.camera_params)
        self._idx += 1
        return cameras

def format_tensor(input, dtype=torch.float32, device: str = "cpu") -> torch.Tensor:
    """
    Helper function for converting a scalar value to a tensor.

    Args:
        input: Python scalar, Python list/tuple, torch scalar, 1D torch tensor
        dtype: data type for the input
        device: torch device on which the tensor should be placed.

    Returns:
        input_vec: torch tensor with optional added batch dimension.
    """
    if not torch.is_tensor(input):
        input = torch.tensor(input, dtype=dtype, device=device)
    if input.dim() == 0:
        input = input.view(1)
    if input.device != device:
        input = input.to(device=device)
    return input

def look_at_rotation(camera_position, at=((0, 0, 0),), up=((0, 0, 1),), device: str = "cpu") -> torch.Tensor:
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.

    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.

    Args:
        camera_position: position of the camera in world coordinates
        at: position of the object in world coordinates
        up: vector specifying the up direction in the world coordinate frame.

    The inputs camera_position, at and up can each be a
        - 3 element tuple/list
        - torch tensor of shape (1, 3)
        - torch tensor of shape (N, 3)

    The vectors are broadcast against each other so they all have shape (N, 3).

    Returns:
        R: (N, 3, 3) batched rotation matrices
    """
    # Format input and broadcast
    broadcasted_args = convert_to_tensors_and_broadcast( camera_position, at, up, device=device )
    camera_position, at, up = broadcasted_args
    for t, n in zip([camera_position, at, up], ["camera_position", "at", "up"]):
        if t.shape[-1] != 3:
            msg = "Expected arg %s to have shape (N, 3); got %r"
            raise ValueError(msg % (n, t.shape))
    z_axis = F.normalize(at - camera_position, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(dim=1, keepdim=True)
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    return R.transpose(1, 2)


def look_at_view_transform(
    dist=1.0,
    elev=0.0,
    azim=0.0,
    degrees: bool = True,
    eye: Optional[Sequence] = None,
    at=((0, 0, 0),),  # (1, 3)
    up=((0, 0, 1),),  # (1, 3)
    device="cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function returns a rotation and translation matrix
    to apply the 'Look At' transformation from world -> view coordinates [0].

    Args:
        dist: shapes ([1] or [N]) distance of the camera from the object
        
        elev: shapes ([1] or [N])  angle in degres or radians. This is the angle between the
            vector from the object to the camera, and the horizontal plane y = 0 (xz-plane).
        
        azim: shapes ([1] or [N])  angle in degrees or radians. The vector from the object to
            the camera is projected onto a horizontal plane y = 0.
            azim is the angle between the projected vector and a
            reference vector at (1, 0, 0) on the reference plane (the horizontal plane). 
            
        degrees: boolean flag to indicate if the elevation and azimuth
            angles are specified in degrees or radians.
        
        eye: the position of the camera(s) in world coordinates. If eye is not
            None, it will overide the camera position derived from dist, elev, azim.
        
        up: the direction of the x axis in the world coordinate system.
        
        at: the position of the object(s) in world coordinates.
        
        eye, up and at can be of shape (1, 3) or (N, 3).

    Returns:
        2-element tuple containing

        - **R**: the rotation to apply to the points to align with the camera.
        - **T**: the translation to apply to the points to align with the camera.

    References:
    [0] https://www.scratchapixel.com
    """

    if eye is not None:
        broadcasted_args = convert_to_tensors_and_broadcast(eye, at, up, device=device)
        eye, at, up = broadcasted_args
        C = eye
    else:
        broadcasted_args = convert_to_tensors_and_broadcast(dist, elev, azim, at, up, device=device)
        dist, elev, azim, at, up = broadcasted_args
        C = (camera_position_from_spherical_angles(dist, elev, azim, degrees=degrees, device=device)+ at)

    R = look_at_rotation(C, at, up, device=device)
    T = -torch.bmm(R.transpose(1, 2), C[:, :, None])[:, :, 0]
    return R, T

def camera_position_from_spherical_angles(
    distance, elevation, azimuth, degrees: bool = True, device: str = "cpu"
) -> torch.Tensor:
    """
    Calculate the location of the camera based on the distance away from
    the target point, the elevation and azimuth angles.

    Args:
        distance: distance of the camera from the object.
        elevation, azimuth: angles.
            The inputs distance, elevation and azimuth can be one of the following
                - Python scalar
                - Torch scalar
                - Torch tensor of shape (N) or (1)
        degrees: bool, whether the angles are specified in degrees or radians.
        device: str or torch.device, device for new tensors to be placed on.

    The vectors are broadcast against each other so they all have shape (N, 1).

    Returns:
        camera_position: (N, 3) xyz location of the camera.
    """
    broadcasted_args = convert_to_tensors_and_broadcast( distance, elevation, azimuth, device=device )
    dist, elev, azim = broadcasted_args
    if degrees:
        elev = math.pi / 180.0 * elev
        azim = math.pi / 180.0 * azim
    x = dist * torch.cos(elev) * torch.sin(azim)
    y = dist * torch.sin(elev)
    z = dist * torch.cos(elev) * torch.cos(azim)
    camera_position = torch.stack([x, y, z], dim=1)
    if camera_position.dim() == 0:
        camera_position = camera_position.view(1, -1)  # add batch dim.
    return camera_position.view(-1, 3)

