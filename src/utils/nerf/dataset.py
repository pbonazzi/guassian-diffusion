
import os
import pdb
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.nerf.load_blender import load_blender_data

def safe_path(path):
    if os.path.exists(path):
        return path
    else:
        os.mkdir(path)
        return path


# Ray helpers
def get_rays(H, W, K, c2w):
    device = c2w.device

    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device),
                          torch.linspace(0, H-1, H, device=device))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i, device=device)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


# Ray helpers
def get_uvs_from_ray(H, W, K, c2w,pts):
    RP = torch.bmm(c2w[:3,:3].T[None,:,:].repeat(pts.shape[0],1,1),pts[:,:,None])[:,:,0]
    t = torch.mm(c2w[:3,:3].T,-c2w[:3,-1][:,None])
    pts_local0 = torch.sum((pts-c2w[:3,-1])[..., None, :] * (c2w[:3,:3].T), -1)
    pts_local = pts_local0/(-pts_local0[...,-1][...,None]+1e-7)
    u = pts_local[...,0]*K[0][0]+K[0][2]
    v = -pts_local[...,1]*K[1][1]+K[1][2]
    uv = torch.stack((u,v),-1)
    return uv,pts_local0


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

def load_mem_data(mem):
    poses = mem.pose
    R, T = (poses[:, :3, :3]), poses[:, :3, -1]
    R, T = R, -(T[: ,None ,:] @ R)[: ,0]
    return mem.pts, mem.image, mem.K, R, T, poses, mem.mask
    
class MemDataset(object):
    def __init__(self,pose,image,mask,K):
        self.pose = pose
        self.mask = mask
        self.image = image
        self.K = K


class NerfDataset():
    def __init__(self, datadir, basedir):
        self.datadir = datadir
        self.logpath = safe_path(basedir)
        K = None

        half_res = True
        white_bkgd = False
        testskip = 0

        images, poses, render_poses, hwf, i_split = load_blender_data(self.datadir, half_res, testskip)
        # print('Loaded blender', images.shape, render_poses.shape, hwf, self.datadir)
        masks = images[..., -1:]
        near = 2.
        far = 6.
        if white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]


        self.i_split = i_split
        self.images = images
        self.masks = masks
        self.poses = poses
        self.render_poses = render_poses

        # Cast intrinsics to right types
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

        if K is None:
            self.K = np.array([
                [focal, 0, 0.5 * W],
                [0, focal, 0.5 * H],
                [0, 0, 1]
            ])
        else:
            self.K = K


        self.hwf = hwf
        self.near = near
        self.far = far

    def split(self, num):
        val = MemDataset(self.poses[num:],self.images[num:],self.masks[num:],self.K)
        train = MemDataset(self.poses[:num],self.images[:num],self.masks[:num],self.K)

        return train, val


