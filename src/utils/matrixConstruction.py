import torch
import math
import unittest

from .mathHelper import normalize


def rotationMatrixX(alpha):
    tc = torch.cos(alpha)
    ts = torch.sin(alpha)
    R = torch.eye(3, device=alpha.device)
    R[1, 1] = tc
    R[2, 2] = tc
    R[1, 2] = -ts
    R[2, 1] = ts
    return R


def rotationMatrixY(alpha):
    tc = torch.cos(alpha)
    ts = torch.sin(alpha)
    R = torch.eye(3, device=alpha.device)
    R[0, 0] = tc
    R[2, 2] = tc
    R[2, 0] = -ts
    R[0, 2] = ts
    return R


def rotationMatrixZ(alpha):
    tc = torch.cos(alpha)
    ts = torch.sin(alpha)
    R = torch.eye(3, device=alpha.device)
    R[0, 0] = tc
    R[1, 1] = tc
    R[0, 1] = -ts
    R[1, 0] = ts
    return R


def rotationMatrix(alpha, beta, gamma):
    return rotationMatrixX(alpha).mm(rotationMatrixY(beta).mm(rotationMatrixZ(gamma)))


def convertWorldToCameraTransform(Rw, C):
    """
    Takes a camera transformation in world space and returns the camera transformation in
    camera space to be used as extrinsic parameters
    Rw (B, 3, 3)
    C  (B, 3)
    """
    Rc = Rw.transpose(1, 2)
    t = -Rc.matmul(C.unsqueeze(-1)).squeeze(-1)
    return (Rc, t)


def batchAffineMatrix(R, t, scale=1.0, column_matrix=True):
    """
    affine transformation with uniform scaling->rotation->tranlation
    Args:
        R (..., 3, 3): rotation matrix
        t (..., 3): translation tensor
        scale (scaler or vector): scale vector
        column_matrix (bool): if True, [R | t] expect transformation R @ p,
            otherwise p @ R (in pytorch3d)

    Returns:
        (..., 4, 4) transformation matrix
    """
    assert R.shape[-2:] == (3, 3), "R must be of shape (..., 3, 3)"
    assert t.dim() == (R.dim() - 1), f"t must be of shape (..., 3) ({t.shape=}"
    out_shape = list(R.shape[:-2]) + [4, 4]
    T = R.new_zeros(out_shape)
    T[..., 3, 3] = 1.0
    T[..., :3, :3] = scale * R
    if column_matrix:
        T[..., :3, 3] = t
    else:
        T[..., 3, :3] = t
    return T


if __name__ == '__main__':
    unittest.main()
