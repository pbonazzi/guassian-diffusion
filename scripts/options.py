import argparse
import os
import pdb
import logging
import warnings
import shutil
import uuid
import copy
import ray
import numpy
import torch
import math

# custom
from src.misc.imageFilters import Pix2PixInitialize, Img2ImgDiffusionInitialize
from src.training.clip_encoder import get_embed_fn

def get_configurations():

    # Arguments
    parser = argparse.ArgumentParser( description='Train implicit representations without 3D supervision.')


    parser.add_argument('--dataset_name',  type=str, default="any_obj", choices=["any_obj", "blender", "mesh", "noise", "images"], help="Path to data directory or path to mesh/pointcloud")
    
    parser.add_argument('--input_path',  type=str, required=True, help="Path to data directory or path to mesh/pointcloud")
    parser.add_argument('--input_type',  type=str, default="mesh", choices=["mesh", "point"], help="Path to data directory or path to mesh/pointcloud")
    parser.add_argument('--input_format',  type=str, default="ply", choices=["off","ply", "xyzn", "obj"], help="Path to data directory or path to mesh/pointcloud")
    parser.add_argument('--input_filter', type=str, default= 'none', choices=['StableDiffusion', 'L0Smooth', 'SuperPixel',  'Pix2Pix', 'none'])
    parser.add_argument('--input_xyz_rotation', type=list, default=numpy.array([math.pi/2,math.pi*1.5,math.pi/2]))
    parser.add_argument('--input_color',  type=str, default="black", choices=["grey","black","blue", "from_verts","from_verts_n"])

    parser.add_argument('--init_path',  type=str,help="Path to data directory or path to mesh/pointcloud")
    parser.add_argument('--init_xyz_rotation', type=list, default=numpy.array([0,0,0]))
    parser.add_argument('--init_type',  type=str, default="mesh", choices=["mesh", "point"], help="Path to data directory or path to mesh/pointcloud")
    parser.add_argument('--init_format',  type=str, default="ply", choices=["ply", "xyzn", "obj"], help="Path to data directory or path to mesh/pointcloud")
    parser.add_argument('--init_color',  type=str, default="grey", choices=["grey","black", "blue", "from_verts","from_verts_n"])

    parser.add_argument('--init_another_mesh', default=False, action='store_true')
    parser.add_argument('--init_from_noisy', default=False, action='store_true')
    parser.add_argument('--init_from_images', default=False, action='store_true')
    parser.add_argument('--init_icosphere', default=False, action='store_true')
    parser.add_argument('--init_cone', default=False, action='store_true')
    parser.add_argument('--init_cylinder', default=False, action='store_true')
    parser.add_argument('--ray_tuning', action='store_true')

    # image - filters
    parser.add_argument('--path_to_denoising_diffusion', type=str, default="/data/storage/bpietro/huggingface/diffusers/model-stable-diffusion-2-1-denoising-01")
    parser.add_argument('--point_renderer', action='store_true',  default=False)

    # data - camera
    parser.add_argument('--camera_min_dist',type=float, default=1.7)
    parser.add_argument('--camera_max_dist',type=float, default=2.2)
    parser.add_argument('--camera_znear',type=float, default=0.01)
    parser.add_argument('--camera_zfar',type=float, default=100.0)
    parser.add_argument('--camera_fov',type=float, default=60.0)

    # number of camera poses
    parser.add_argument('--train_images',type=int, default=128)
    parser.add_argument('--train_batch_size',type=int, default=8)

    parser.add_argument('--val_images',type=int, default=128)

    parser.add_argument('--image_log_every',type=int, default=2)
    parser.add_argument('--image_save_every',type=int, default=0)

    parser.add_argument('--save_views', action='store_true',  default=False)
    parser.add_argument('--save_pointcloud', action='store_true', default=False)

    # model - number of parameters
    parser.add_argument('--n_points_per_cloud',type=int, default=3000)
    parser.add_argument('--size_neigh_for_normals',type=int, default=5)

    # model - require grad
    parser.add_argument('--learn_points', action='store_true',  default=True)
    parser.add_argument('--lr_points', type=float,  default=3e-3)
    
    parser.add_argument('--learn_normals', action='store_true',  default=False)
    parser.add_argument('--lr_normals', type=float,  default=3e-4)

    parser.add_argument('--learn_colors', action='store_true',  default=False)
    parser.add_argument('--lr_colors', type=float,  default=3e-3)
    
    parser.add_argument('--learn_alphas', action='store_true',  default=False)
    parser.add_argument('--lr_alphas', type=float,  default=3e-3)
    
    parser.add_argument('--learn_sh', action='store_true',  default=False)
    parser.add_argument('--lr_sh', type=float,  default=3e-3)

    # model - room light
    parser.add_argument('--use_light', action='store_true', default=False)
    parser.add_argument('--light_source', type=str, default='camera_flash', choices=['camera_flash', 'static' ])
    parser.add_argument('--light_type', type=str, default='point', choices=['ambient', 'tricolor', 'point' , 'directional' ])

    # training - image loss
    parser.add_argument('--lambda_mse_rgb_loss', type=float, default=1.0)
    parser.add_argument('--lambda_mse_rgba_loss', type=float, default=0.0)
    parser.add_argument('--lambda_mse_a_loss', type=float, default=1.0)
    parser.add_argument('--lambda_smape_loss', type=float, default=0.0)
    
    # training - variance loss
    parser.add_argument('--lambda_variance_loss', type=float, default=0.0)

    # training - normal loss
    parser.add_argument('--lambda_normal_smoothness_loss', type=float, default=0.0)

    # training - point loss
    parser.add_argument('--lambda_proj_loss', type=float, default=0.02)
    parser.add_argument('--lambda_repel_loss', type=float, default=0.05)
    parser.add_argument('--repel_proj_scale', type=float, default=4.0)
    parser.add_argument('--repel_proj_knn_k', type=int, default=12)

    # training - semantic loss
    parser.add_argument('--lambda_sem_cons_loss', type=float, default=0.00)
    parser.add_argument('--sem_cons_every', type=int, default=2)
    parser.add_argument('--sem_cons_epoch_start', type=int, default=0)
    parser.add_argument('--sem_cons_n_of_close_cam', type=int, default=8)
    parser.add_argument('--sem_cons_n_of_add_images', type=int, default=256)
    parser.add_argument('--sem_cons_curriculum_learning',  action='store_true',  default=False)
    parser.add_argument('--sem_cons_clip_model_type',  type=str,  default="clip_vit", choices=["clip_vit", "clip_rn50"])

    # training - learning parameters
    parser.add_argument('--with_reduce_splat', action='store_true',  default=False)
    parser.add_argument('--with_restarts', action='store_true',  default=False)
    parser.add_argument('--remove_and_repeat', action='store_true',  default=False)
    
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay_steps', type=int, default=500)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--beta0', type=int, default=0.5)
    parser.add_argument('--beta1', type=int, default=0.9)
    parser.add_argument('--scheduler_milestones', type=list, default=[500,800])
    parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    parser.add_argument('--scheduler', type=str, default="ExponentialLR", choices=["MultiStepLR", "ReduceLROnPlateau", "ExponentialLR"])

    # raster settings
    parser.add_argument('--raster_points_per_pixel',type=int, default=10)
    parser.add_argument('--raster_binsize',type=int, default=23)
    parser.add_argument('--raster_backface_culling', action='store_true',  default=False)
    parser.add_argument('--frnn_radius',type=float)
    parser.add_argument('--raster_knn_points',type=int, default=7)
    parser.add_argument('--raster_radii_backward_scaler', type=float, default=10)
    parser.add_argument('--raster_image_size',type=int, default=512)

    parser.add_argument('--max_epochs', type=int, default=300, help='Checkpoint and exit after specified number of seconds with exit code 2.')

    # init
    parser.add_argument('--overwrite', action='store_true', default=False, help="overwrite")
    parser.add_argument("--output_dir", type=str, default="/data/storage/bpietro/thesis/DSS/output")
    parser.add_argument('--output_name',type=str, default=str(uuid.uuid4()), help="Name of the output run")
    parser.add_argument('--warnings', default=False, help="transform warnings to errors")

    args = parser.parse_args()


    if args.warnings: warnings.filterwarnings("error")

    out_dir = os.path.join(args.output_dir, args.output_name)
    if args.overwrite: shutil.rmtree(out_dir, ignore_errors=True)

    
    if os.path.exists(os.path.join(out_dir, "config.yaml")):
        return config.load_config(os.path.join(out_dir, "config.yaml"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # model type
    config = {
        "generic":{
            "point_renderer": args.point_renderer,
            "device": device,
            "output_dir" : out_dir,
            "ray_tuning" : args.ray_tuning, 
        },
        "model" : {
            # number of parameters
            "n_points_per_cloud": args.n_points_per_cloud,
            "size_neigh_for_normals": args.size_neigh_for_normals,

            # require grad
            "learn_points": args.learn_points,
            "learn_normals": args.learn_normals,
            "learn_colors": args.learn_colors,
            "learn_alphas": args.learn_alphas,
            "learn_sh": args.learn_sh,

            # room light
            "use_light": args.use_light,
            "light_source": args.light_source, 
            "light_type": args.light_type, 
        },

        "training": {

            # image loss
            "lambda_mse_rgb_loss": args.lambda_mse_rgb_loss,
            "lambda_mse_rgba_loss": args.lambda_mse_rgba_loss,
            "lambda_mse_a_loss": args.lambda_mse_a_loss,
            "lambda_smape_loss": args.lambda_smape_loss, 
            
            # normal loss
            "lambda_normal_smoothness_loss": args.lambda_normal_smoothness_loss,
            "normal_neighboor_size": 3,

            # point loss
            "lambda_proj_loss":args.lambda_proj_loss,
            "lambda_repel_loss": args.lambda_repel_loss,
            "repel_proj_scale": args.repel_proj_scale,
            "repel_proj_knn_k": args.repel_proj_knn_k,

            # variance loss
            "lambda_variance_loss": args.lambda_variance_loss,
            
            # semantic loss
            "lambda_sem_cons_loss": args.lambda_sem_cons_loss,
            "sem_cons_every": args.sem_cons_every,
            "sem_cons_epoch_start": args.sem_cons_epoch_start,
            "sem_cons_n_of_close_cam": args.sem_cons_n_of_close_cam,
            "sem_cons_n_of_add_images": args.sem_cons_n_of_add_images,
            "clip_model_type" : args.sem_cons_clip_model_type,
            "init_clip": get_embed_fn(model_type=args.sem_cons_clip_model_type,  device=device, num_layers=-1,  clip_cache_root=os.path.expanduser("~/.cache/clip")) if args.lambda_sem_cons_loss > 0 else None,
            "curriculum_learning": args.sem_cons_curriculum_learning,

            # learning parameters
            "lr": 0.01,
            "lr_points": args.lr_points, 
            "lr_normals": args.lr_normals, 
            "lr_alphas": args.lr_alphas, 
            "lr_colors": args.lr_colors, 
            "lr_sh": args.lr_sh,
            "lr_decay_steps": 500,
            "lr_decay_rate": 0.93, # 0.1
            "betas": [0.5, 0.9],

            "scheduler": {
                "scheduler_name": args.scheduler,
                "scheduler_milestones": [500, 800],
                "scheduler_gamma": 0.5,
                "scheduler_decay": 0.93,
                },
            
            # logs
            "last_epoch": 0,
            "ckp_backup_every": 0,
            "ckp_resume_from": "model.pt",
            "image_log_every": args.image_log_every,
            "image_save_every": args.image_save_every,
            "pointcloud_log_every": 0,
            "pointcloud_stats_every": 0,

            "stats_list": ["chamfer_point", "chamfer_normal", "psnr", "iou", "fid_score"],
            "stats_model_selection_metric": "chamfer_point",
            "stats_model_selection_mode": "minimize",

            "console_log_every": 1,  
            "debug_every": 0,          
            "output_dir": args.output_dir,
            "output_file":"train.log",
            "max_epochs": args.max_epochs,
            "patience": 30, 
            "remove_and_repeat": args.remove_and_repeat,
            "with_reduce_splat": args.with_reduce_splat,
            "with_restarts": args.with_restarts,
            
        },

        "data": {

            "dataset_name": args.dataset_name,

            "input_path": args.input_path,
            "input_xyz_rotation": args.input_xyz_rotation,
            "input_format":  args.input_format,
            "input_type": args.input_type,
            "input_filter": args.input_filter,
            "input_color": args.input_color,

            "init_path": args.init_path,
            "init_xyz_rotation": args.init_xyz_rotation,
            "init_type": args.init_type,
            "init_format": args.init_format,
            "init_color": args.init_color,
            
            "init_pix2pix": Pix2PixInitialize() if args.input_filter == "Pix2Pix" else None ,
            "init_diffusion": Img2ImgDiffusionInitialize(args.path_to_denoising_diffusion) if args.input_filter == "StableDiffusion" else None ,

            "init_icosphere":  args.init_icosphere,
            "init_cone": args.init_cone,
            "init_cylinder": args.init_cylinder,
            "init_from_images": args.init_from_images,
            "init_from_noisy": args.init_from_noisy,
            "init_another_mesh": args.init_another_mesh,

            "camera_type": "FoVPerspectiveCameras",  # 'FoVPerspectiveCameras' | 'FoVOrthographicCameras'
            "camera_min_dist": args.camera_min_dist,
            "camera_max_dist": args.camera_max_dist,
            "camera_znear": args.camera_znear,
            "camera_zfar": args.camera_zfar,
            "camera_fov": args.camera_fov,
            "camera_image_size": args.raster_image_size,
            "camera_aspect_ratio": 1.0,
            
            "train_images" : args.train_images,
            "train_batch_size": min(args.train_images, args.train_batch_size),
            "val_images" : args.val_images,
            "val_batch_size": 1,
            "save_views" : args.save_views,
            "save_pointcloud" : args.save_pointcloud,
            "sampler": True,

        },

        "dss": {
            "steps_dss_backward_radii": int(args.max_epochs * args.train_images / args.train_batch_size),
            "gamma_dss_backward_radii": 0.9 ,
            "limit_dss_backward_radii": 1,

            "raster_settings" : {
                "backface_culling": args.raster_backface_culling,
                "cutoff_threshold": 1.0,
                "depth_merging_threshold": 0.05,
                "Vrk_invariant": True,
                "Vrk_isotropic": False,
                "knn_points": args.raster_knn_points,
                "radii_backward_scaler": args.raster_radii_backward_scaler,
                "image_size":  args.raster_image_size,
                "points_per_pixel": args.raster_points_per_pixel,
                "bin_size": None, # cuda kernel upper limit , managed internally with heurestics
                "max_points_per_bin": None, # default = max(num_points/5, 10000)
                "clip_pts_grad": 0.05,
                "antialiasing_sigma" : 1.0,
                "frnn_radius": -1 , 
            },

        }

    }


    return config
