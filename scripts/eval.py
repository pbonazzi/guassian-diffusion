import open3d as o3d
import os
import argparse
import torch
import pdb
import numpy as np
from pytorch3d.structures import Pointclouds
from pytorch3d.loss import chamfer_distance


def evaluate(save_images= False, save_point_cloud = False):

    trainer.model.eval()

    # points
    pred_pc = trainer.model.get_point_clouds(with_colors=False, with_normals=True,require_normals_grad=False)
    if save_point_cloud:
        ## 

    gt_pc = trainer.dataset.get_point_clouds()
    cd_p, cd_n = chamfer_distance(pred_pc, gt_pc, x_lengths=pred_pc.num_points_per_cloud(), y_lengths=gt_pc.num_points_per_cloud())

    # images
    n = len(val_loader)
    indexes = list(range(n))

    with torch.autograd.no_grad():
        eval_dict = {}
        for met in cfg["training"]["stats_list"]:
            eval_dict[met] = 0.0

        for i, ind in enumerate(indexes):

            batch , idx = val_loader.dataset[ind]
            data = trainer.process_data_dict(batch, val_cameras, epoch=epoch_it, visualize=True)              

            data["camera"].R = data["camera"].R.unsqueeze(0)
            data["camera"].T = data["camera"].T.unsqueeze(0)
            data["camera"]._N = 1

            eval_dict = trainer.eval_step(data=data, eval_dict=eval_dict, val_loader=val_loader, it=it, batch_it=i, cameras=data["camera"])

            if save_images:
                path = os.path.join(trainer.vis_dir, "val", str(epoch_it))
                os.makedirs(path, exist_ok=True)
                name = os.path.join(path, "%06d.png" %indexes[i])
                trainer.visualize(  data=data, cameras=data["camera"], step=epoch_it,  vis_type='image', save=epoch_save_img, log=epoch_log_img, name=name)

    
    if 'psnr' in list(eval_dict.keys()) and epoch_metrics:
        eval_dict['psnr']/=len(indexes)
        trainer.tb_logger.add_scalars('eval', eval_dict, global_step=it)

    if cfg["generic"]["ray_tuning"]:
        batch , idx = val_loader.dataset[0]
        data = trainer.process_data_dict(batch, val_cameras,epoch=epoch_it, visualize=True)              
        data["camera"].R = data["camera"].R.unsqueeze(0)
        data["camera"].T = data["camera"].T.unsqueeze(0)
        data["camera"]._N = 1

        # first_pc
        pc1 = trainer.model.get_point_clouds(with_colors=False, with_normals=True,require_normals_grad=False, cameras=data["camera"])
        cd_p, cd_n = chamfer_distance(pc1, gt_pc, x_lengths=pc1.num_points_per_cloud(), y_lengths=gt_pc.num_points_per_cloud())
        tune.report(loss=loss, chamfer_distance=cd_p)

    if epoch_exit:
        if cfg["training"]["debug_every"] > 0 and epoch_it > cfg["training"]["debug_every"]:
            for t in trainer._threads:
                t.join()
        exit(3)

    return
        


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser( description='Train implicit representations without 3D supervision.')
    parser.add_argument('--pc1', default=None, type=str, help="Path to data directory or path to mesh/pointcloud")
    parser.add_argument('--pc2',  default=None,type=str, help="Path to data directory or path to mesh/pointcloud")
    
    parser.add_argument('--img1', default=None, type=str,  help="Path to data directory or path to mesh/pointcloud")
    parser.add_argument('--img2', default=None, type=str, help="Path to data directory or path to mesh/pointcloud")

    args = parser.parse_args()

    main(args)