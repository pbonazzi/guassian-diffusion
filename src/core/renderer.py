import torch, pdb
from pytorch3d.renderer import PointsRenderer
from pytorch3d.renderer.compositing import weighted_sum
from .. import logger_py

from typing import Tuple

import torch
import torch.nn as nn
from pytorch3d.structures import Meshes, Pointclouds

from pytorch3d.renderer.points.rasterize_points import rasterize_points
from src.models.sh import eval_sh

__all__ = ['SurfaceSplattingRenderer']


class SurfaceSplattingRenderer(PointsRenderer):

    def __init__(self, rasterizer, compositor):
        super().__init__(rasterizer, compositor)

    def forward(self, point_clouds, only_base=True, verbose=False, **kwargs) -> torch.Tensor:
        """
        point_clouds_filter: used to get activation mask and update visibility mask
        cutoff_threshold
        """

        assert not point_clouds.isempty()

        # rasterize
        fragments, point_clouds = self.rasterizer(point_clouds, **kwargs)

        # compute weight: scalar*exp(-0.5Q)
        weights = torch.exp(-0.5 * fragments.qvalue) * fragments.scaler
        weights = weights.permute(0, 3, 1, 2)
        weights = torch.clamp( weights, min=0, max=2000)

        # from fragments to rgba
        # pts_color = point_clouds.features_packed()[...,:4].permute(1, 0)
        pts_color = point_clouds.features_packed().permute(1, 0)
        pts_color = torch.clamp(pts_color, min=0, max=1)
        frgm = fragments.idx.long().permute(0, 3, 1, 2)

        if self.compositor is None:
            images = weighted_sum(frgm, weights, pts_color,  **kwargs)
        else:
            images = self.compositor(frgm, weights, pts_color, **kwargs)

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)
        mask = fragments.occupancy

        images = torch.cat([images, mask.unsqueeze(-1)], dim=-1)

        if verbose:
            return images, fragments

        return images
