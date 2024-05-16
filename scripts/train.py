# dependencies
import argparse
import time
import os
import pdb
import logging
import copy
import open3d as o3d
import ray

from tqdm import tqdm
import torch
import torch.optim as optim
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds
import numpy as np
import cv2
import imageio.v2 as imageio
from scipy.spatial.distance import directed_hausdorff


# customized modules
from src.utils import tolerating_collate, slice_dict, slice_dict_data
from src.utils.sampler import WeightedSubsetRandomSampler
from src.utils.dataset import GenericDataset
from src.utils.nerf.load_blender import load_blender_data
from src.utils.nerf.load_mesh import load_image_data, load_mesh_data, load_noise_data
from src import logger_py, set_deterministic_
set_deterministic_()
from src.misc.checkpoints import CheckpointIO
from src.training.trainer import create_trainer
from utils.config import create_model, create_generator, create_scheduler, create_scaler, create_optimizer, save_config
from scripts.options import get_configurations
from pytorch3d.renderer import PointsRasterizer, AlphaCompositor, PointsRenderer, PointsRasterizationSettings

def main(args):
    if not args["generic"]["ray_tuning"]:
        return init(args)

    config = {
        "lr": ray.tune.loguniform(1e-6, 1e-1),
        "knn_points": ray.tune.choice([2, 4, 8]),
        "radii_backward_scaler": ray.tune.choice([2, 4, 8]),
        "args": args
    }
    scheduler = ray.tune.schedulers.ASHAScheduler(
        max_t=args.epoch_exit,
        grace_period=1,
        reduction_factor=2)

    result = ray.tune.run(
        ray.tune.with_parameters(init),
        resources_per_trial={"cpu": 2, "gpu": 1},
        config=config,
        metric="chamfer_distance",
        mode="min",
        num_samples=10,
        scheduler=scheduler
    )

    best_trial = result.get_best_trial("chamfer_distance", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation chamfer distance: {}".format(best_trial.last_result["chamfer_distance"]))


def early_stopping(train_loss, lowest_loss, trigger_times, patience, trainer):
        # Early Stoping
    if train_loss > lowest_loss:
        trigger_times += 1
        print('Trigger Times:', trigger_times)
    else:
        trigger_times = 0

    if trigger_times >= patience:
        print('Early stopping!\nStart to validate process.')
        return True, trigger_times
    for param_group in trainer.optimizer.param_groups :
        if param_group['lr'] < 6.9e-6:
            print('Early stopping for Learning Rate!\nStart to validate process.')
            return True, trigger_times
    return False, trigger_times

def log_chamfer(val_loader, trainer, epoch_it):
        # refinement policy
    #if cfg["data"]["init_from_noisy"]:
    val_cameras = val_loader.dataset.get_cameras()
    pointcloud = trainer.model.get_point_clouds(with_colors=True, with_lights=False, with_normals=True,require_normals_grad=False, cameras=val_cameras)
    pointcloud_tgt = val_loader.dataset.get_pointclouds(num_points=trainer.n_eval_points).to(device=pointcloud.device)
    cd_p, cd_n = chamfer_distance(pointcloud_tgt, pointcloud, x_lengths=pointcloud_tgt.num_points_per_cloud(), y_lengths=pointcloud.num_points_per_cloud(),)
    dist = directed_hausdorff(pointcloud.points_packed().detach().cpu(), pointcloud_tgt.points_packed().detach().cpu())[0]
        
    trainer.tb_logger.add_scalar(f"val/c_points", cd_p, epoch_it)
    trainer.tb_logger.add_scalar(f"val/c_normals", cd_n, epoch_it)
    trainer.tb_logger.add_scalar(f"val/dist_hausdorff", dist, epoch_it)
    # logger_py.info('%s = %.4g' %('chamfer_point', cd_p.item()))

    logger_py.info('%s = %.4g' %('hausdorff_distance', dist))
    logger_py.info('%s = %.4g' %('chamfer_point', cd_p.item()))

def init(cfg):
    
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ray Tuning
    if cfg["generic"]["ray_tuning"]:
         cfg["dss"]["knn_points"] = ray_config["knn_points"]
         cfg["dss"]["radii_backward_scaler"] = ray_config["radii_backward_scaler"]
         cfg["training"]["lr"] = ray_config["lr"]

    # Logging
    os.makedirs(cfg["generic"]["output_dir"], exist_ok=True)
    fileHandler = logging.FileHandler(os.path.join(cfg["generic"]["output_dir"], cfg["training"]["output_file"]))
    fileHandler.setLevel(logging.DEBUG)
    logger_py.addHandler(fileHandler)

    # Datasets
    train_dataset = GenericDataset(name='train', device=device, settings=cfg)
    val_dataset = GenericDataset(name='val', device=device, settings=cfg)

    if cfg["data"]["dataset_name"] == "blender":
        images, poses, _, intrinsic, _ = load_blender_data(cfg["data"]["input_path"], image_size=cfg["data"]["camera_image_size"])
        train_dataset.load_from_images(images, poses,  intrinsic, pts=None, locations=None, noisy_points=None)
        val_dataset.load_from_images(images, poses, intrinsic, pts=None, locations=None)

    elif cfg["data"]["dataset_name"] == "images":
        pts, noisy_points, images, poses, intrinsic, locations = load_image_data(cfg["data"]["input_path"], cfg["data"]["init_from_noisy"])
        train_dataset.load_from_images(images, poses, intrinsic, pts, locations, noisy_points)
        val_dataset.load_from_images(images, poses, intrinsic, pts, locations)


    elif cfg["data"]["dataset_name"] == "mesh":
        pts, images, poses, intrinsic, locations = load_mesh_data( path=cfg["data"]["input_path"],
                                                        znear=cfg["data"]["camera_min_dist"], 
                                                        zfar=cfg["data"]["camera_max_dist"],
                                                        input_type=cfg["data"]["input_type"], 
                                                        input_format=cfg["data"]["input_format"], 
                                                        num_of_points=cfg["model"]["n_points_per_cloud"], 
                                                        image_size=cfg["data"]["camera_image_size"], 
                                                        num_cams=400)
        train_dataset.load_from_images(images, poses, intrinsic, pts, locations)
        val_dataset.load_from_images(images, poses, intrinsic, pts, locations)

    elif cfg["data"]["dataset_name"] == "noise":
        pts, noisy_points, images, poses, intrinsic, locations = load_noise_data(znear=cfg["data"]["camera_min_dist"], 
                                                        zfar=cfg["data"]["camera_max_dist"], 
                                                        image_size=cfg["data"]["camera_image_size"], 
                                                        num_cams=15, 
                                                        path=cfg["data"]["input_path"],
                                                        num_of_points=cfg["model"]["n_points_per_cloud"], 
                                                        input_type=cfg["data"]["input_type"], 
                                                        input_format=cfg["data"]["input_format"], 
                                                        diffusion=cfg["data"]["init_diffusion"]
                                                        )
        train_dataset.load_from_images(images, poses, intrinsic, pts, locations, noisy_points)
        val_dataset.load_from_images(images, poses, intrinsic, pts, locations)


    data_dict = train_dataset.data_dict

    texture = None
    if cfg["data"]["dataset_name"] != "blender":
        texture = train_dataset.texture

    sem_cons_dict = train_dataset.sem_cons_dict
    images = train_dataset.rgb_images

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg["data"]["val_batch_size"], shuffle=False, collate_fn=tolerating_collate,) 

    # Model
    model = create_model(cfg, device=device, data_dict=data_dict, texture=texture, images=images)
    nparameters = sum(p.numel() for p in model.parameters())
    logger_py.info('Total number of parameters: %d' % nparameters)
    cfg["model"]['nparameters'] = nparameters


    optimizer = create_optimizer(cfg, model)
    scheduler = create_scheduler(cfg["training"]["scheduler"], optimizer)
    scaler = create_scaler()

    # Config
    epoch_it, it = 0, 0
    cfg["generic"]["start_time"]  = time.time()
    save_config(os.path.join(cfg["generic"]["output_dir"], 'config.yaml'), cfg)

    generator = create_generator(cfg, model, device=device)
    trainer = create_trainer(cfg=cfg, device=device,model=model, scaler=scaler, optimizer=optimizer, scheduler=scheduler, 
        sem_cons_dict=sem_cons_dict, generator=generator, val_loader=val_loader)


    # Training Loop
    torch.autograd.set_detect_anomaly(True) 
    trainer.tb_logger.add_scalar('train/points', trainer.model.points.shape[1] , 0)

    # Early stopping
    last_loss, lowest_loss = 0, float("inf")
    patience = cfg["training"]["patience"]
    trigger_times = 0

    # Core
    # trainer.sem_cons_epoch_start = float("inf")
    semantic_started, only_base  = False,  True
    torch.cuda.empty_cache()

    if cfg["generic"]["point_renderer"]:
        raster_settings = PointsRasterizationSettings( bin_size=23, image_size=cfg["data"]["camera_image_size"], radius=0.0015, points_per_pixel=15)
        rasterizer = PointsRasterizer(raster_settings=raster_settings)
        trainer.renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor() ) 

    epoch_it = -1
    try:
        while True:

            # denoising - ours
            if cfg["data"]["init_from_noisy"]:
                num_restarts, num_remove, num_reduce = 1, 2, 2
                while True:
                    epoch_it += 1
                    log_chamfer(val_loader, trainer, epoch_it)
                    trainer, train_loss = train(train_dataset, trainer, cfg, epoch_it, logger_py, only_base)
                    lowest_loss = min(lowest_loss, train_loss)
                    do_stop, trigger_times = early_stopping(train_loss, lowest_loss, trigger_times, patience, trainer)
                    if do_stop:
                        return evaluate(val_loader, trainer, cfg, epoch_it, logger_py, save_img=True, save_pc=True)
                    
                    # restarts
                    if cfg["training"]["with_restarts"] and epoch_it > 0 and epoch_it % 30 == 0 and num_restarts > 0:
                        logger_py.info("Restart : "+str(num_restarts))
                        lowest_loss, trigger_times = float("inf"), 0
                        trainer.reset_optimizer(cfg) 
                        num_restarts -= 1   


            # few-shot deformation -dss
            # if True:
            #     num_restarts, num_remove, num_reduce = 6, 2, 2
            #     while True:
            #         epoch_it += 1
            #         trainer, train_loss = train(train_dataset, trainer, cfg, epoch_it, logger_py, only_base)
            #         lowest_loss = min(lowest_loss, train_loss)
            #         do_stop, trigger_times = early_stopping(train_loss, lowest_loss, trigger_times, patience, trainer)
            #         log_chamfer(val_loader, trainer, epoch_it)
            #         if do_stop:
            #             return evaluate(val_loader, trainer, cfg, epoch_it, logger_py, save_img=True, save_pc=True)
            #         # restarts
            #         if cfg["training"]["with_restarts"] and epoch_it % 30 == 0 and num_restarts >= 0:
            #             logger_py.info("Restart : "+str(num_restarts))
            #             lowest_loss, trigger_times = float("inf"), 0
            #             trainer.reset_optimizer(cfg) 
            #             num_restarts -= 1   

            #         if cfg["training"]["remove_and_repeat"]and num_remove > 0 and num_restarts < 0 and epoch_it % 30 == 0:
            #             logger_py.info("Remove and Repeat : "+str(num_remove))
            #             lowest_loss, trigger_times = float("inf"), 0
            #             trainer.model.remove_and_repeat()
            #             trainer.tb_logger.add_scalar('train/points', trainer.model.points.shape[1] , epoch_it)
            #             trainer.reset_optimizer(cfg) 
            #             num_remove -= 1  

            # few-shot deformation - ours
            if cfg["data"]["init_icosphere"] and cfg["data"]["train_images"] < 12:
                num_restarts, num_remove, num_reduce = 25, 0, 3
                stop_alphas = False
                for n in range(cfg["training"]["max_epochs"]):
                    epoch_it += 1

                    # restarts
                    if cfg["training"]["with_restarts"] and epoch_it % 30 == 0 and num_restarts >= 0:
                        logger_py.info("Restart : "+str(num_restarts))
                        lowest_loss, trigger_times = float("inf"), 0
                        # cfg["training"]["lr_sh"] = 3e-3
                        # cfg["training"]["lr_points"] = 3e-3
                        # cfg["training"]["lr_alphas"] = 3e-3
                        trainer.reset_optimizer(cfg) 
                        num_restarts -= 1   


                    # if epoch_it == 100:
                    #     logger_py.info("RGB : "+str(num_restarts))
                    #     trainer.lambda_mse_rgb_loss = 1
                    #     lowest_loss, trigger_times = float("inf"), 0
                    #     trainer.reset_optimizer(cfg) 

                    if num_restarts < 5:
                        cfg["training"]["lr_points"] = 3e-3
                        cfg["training"]["lr_normals"] = 3e-3
                        cfg["training"]["lr_sh"] = 3e-3

                    # repeat points
                    if cfg["training"]["remove_and_repeat"]and epoch_it in [300, 400, 500]:
                        logger_py.info("Remove and Repeat : "+str(num_remove))
                        lowest_loss, trigger_times = float("inf"), 0
                        trainer.model.remove_and_repeat()
                        trainer.tb_logger.add_scalar('train/points', trainer.model.points.shape[1] , epoch_it)
                        trainer.reset_optimizer(cfg) 


                    if cfg["training"]["with_reduce_splat"] and epoch_it == 310 :
                        new_radi = trainer.model.renderer.rasterizer.raster_settings["radii_backward_scaler"]/2
                        trainer.model.renderer.rasterizer.raster_settings["radii_backward_scaler"] = int(new_radi)
                        only_base = True
                        trainer.model.learn_sh = False
                        trainer.model.use_light = True
                        trainer.model.renderer.compositor.background_color = [1] * 3
                        trainer.lambda_normal_smoothness_loss = 0.01

                        num_restarts = 4
                        trainer.tb_logger.add_scalar(f"radii",  int(new_radi), epoch_it)
                        trainer.lambda_normal_smoothness_loss = 0.01

                    # repeat points
                    if cfg["model"]["learn_sh"] and only_base and epoch_it == 200 and False:
                        logger_py.info("Learning Alphas Now")
                        only_base = False
                        lowest_loss, trigger_times = float("inf"), 0
                        trainer.tb_logger.add_scalar('train/points', trainer.model.points.shape[1] , epoch_it)
                        cfg["training"]["lr_sh"] = 22e-3
                        cfg["training"]["lr_points"] = 22e-3
                        #cfg["training"]["lr_alphas"] = 2e-3
                        
                        trainer.reset_optimizer(cfg)
                        #trainer.lambda_normal_smoothness_loss = 0.01

                    trainer, train_loss = train(train_dataset, trainer, cfg, epoch_it, logger_py, only_base)
                    lowest_loss = min(lowest_loss, train_loss)
                    do_stop, trigger_times = early_stopping(train_loss, lowest_loss, trigger_times, patience, trainer)
                    log_chamfer(val_loader, trainer, epoch_it)
                    if do_stop:
                        return evaluate(val_loader, trainer, cfg, epoch_it, logger_py, save_img=True, save_pc=True)
                return evaluate(val_loader, trainer, cfg, epoch_it, logger_py, save_img=True, save_pc=True)

            # cosine restarts
            # if cfg["training"]["with_restarts"] and epoch_it % 30 == 0 and num_refinement > 0:
            #     print('Restart.')
            #     lowest_loss, trigger_times = float("inf"), 0
            #     trainer.reset_optimizer(cfg) 
            #     num_refinement -= 1

            # # # start training alphas
            # if epoch_it == 100 and cfg["model"]["learn_sh"] and only_base:
            #     only_base = True
            #     lowest_loss, trigger_times = float("inf"), 0
            #     cfg["training"]["lr_sh"] = 22e-3
            #     trainer.reset_optimizer(cfg) 

            # if epoch_it == 60 :
            #     only_base = False
            #     lowest_loss, trigger_times = float("inf"), 0
            #     cfg["training"]["lr_sh"] = 22e-3
            #     trainer.reset_optimizer(cfg) 

            # if epoch_it > 50 :
            #     trainer.lambda_normal_smoothness_loss = 0.01

            # if epoch_it == 10 and cfg["training"]["with_reduce_splat"]: 
            #     raster_settings = PointsRasterizationSettings( bin_size=23, image_size=cfg["data"]["camera_image_size"], radius=0.0015, points_per_pixel=15)
            #     rasterizer = PointsRasterizer(raster_settings=raster_settings)
            #     trainer.renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor() )
            #     trainer.reset_optimizer(cfg) 

            # repeat points
            # if epoch_it > 299 and epoch_it % 50 == 0 and cfg["training"]["remove_and_repeat"] :
            #     lowest_loss, trigger_times = float("inf"), 0
            #     trainer.model.remove_and_repeat()
            #     trainer.tb_logger.add_scalar('train/points', trainer.model.points.shape[1] , epoch_it)
            #     trainer.reset_optimizer(cfg) 

            # if epoch_it == 300:
            #     lowest_loss, trigger_times = float("inf"), 0
            #     cfg["training"]["lr_points"] = 3e-3
            #     cfg["training"]["lr_normals"] = 3e-3
            #     cfg["training"]["lr_sh"] = 3e-3
            #     trainer.reset_optimizer(cfg) 


            if epoch_it == cfg["training"]["max_epochs"]:
                print('Early stopping, Reached Max Num of Epochs !\nStart to validate process.')
                return evaluate(val_loader, trainer, cfg, epoch_it, logger_py, save_img=True, save_pc=True)

            torch.cuda.empty_cache()
        
    except KeyboardInterrupt as e:
        print(e)
        print('KeyboardInterrupt ! \nStart to validate process.')
        torch.cuda.empty_cache()
        return evaluate(val_loader, trainer, cfg, epoch_it, logger_py, save_img=True, save_pc=True)



"""

eTRAIN

"""


def train(train_dataset, trainer, cfg, epoch_it, logger_py, only_base=True):
    t0b = time.time()
    
    # Solve
    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=cfg["data"]["train_batch_size"], 
                                                sampler=WeightedSubsetRandomSampler(list(range(len(train_dataset))), \
                                                    np.ones(len(train_dataset)).astype('float32')),  
                                                drop_last=True, collate_fn=tolerating_collate)
    train_cameras = train_dataset.get_cameras()  

    it = epoch_it * len(train_loader)
    image_logs = False
    
    # Logs images
    if cfg["training"]["image_log_every"] > 0 and (epoch_it % cfg["training"]["image_log_every"]) == 0:
        image_logs = True

    for i, (batch, idx) in enumerate(train_loader):
        it += 1

        # TRAIN
        loss = trainer.train_step(batch, cameras=train_cameras, image_logs=image_logs, it=it, idx=idx, epoch=epoch_it, only_base=only_base)

        # Debug
        if it > 0 and cfg["training"]["debug_every"] > 0 and (epoch_it % cfg["training"]["debug_every"]) == 0:
            logger_py.info('Visualizing gradients')
            trainer.debug(batch, cameras=train_cameras, it=it, epoch=epoch_it)

        # Backup
        if it > 0 and cfg["training"]["ckp_backup_every"] > 0 and (epoch_it % cfg["training"]["ckp_backup_every"]) == 0 and False:
            logger_py.info('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)

    # Scheduler Step
    metric = loss if isinstance(trainer.scheduler, optim.lr_scheduler.ReduceLROnPlateau) else None
    trainer.update_learning_rate(it, metric=metric)
    
    # Print
    if cfg["training"]["console_log_every"] > 0 and (epoch_it % cfg["training"]["console_log_every"]) == 0:
        logger_py.info('[Epoch %02d] it=%03d, loss=%.4f, time=%.4f' % (epoch_it, it, loss/len(train_loader), time.time() - t0b))
        t0b = time.time()


    return trainer, loss/len(train_loader)
    

"""

EVAL


"""

#mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))
mse2psnr = lambda x : 20*torch.log10(255 / torch.sqrt(x*255))

def evaluate(val_loader, trainer, cfg, epoch_it, logger_py, save_img=False, save_pc=False):

    logger_py.info('Evaluation ')
    n = len(val_loader)
    path = os.path.join(trainer.vis_dir, "val", str(epoch_it))
    os.makedirs(path, exist_ok=True)

    # evaluate 2D
    indexes = list(range(n))
    eval_dict = {"lpips_alex": 0, "lpips_vgg": 0, "l1": 0, "l2": 0, "psnr": 0}
    val_cameras = val_loader.dataset.get_cameras()

    with torch.autograd.no_grad():

        for i, (batch, idx) in tqdm(enumerate(val_loader)):

            data = trainer.process_data_dict(batch, val_cameras, epoch=epoch_it, visualize=True)              

            img_pred = trainer.generator.generate_images(data, locations=data["camera_loc"], view=data["view"], cameras=data["camera"], )[0]/255 # W, H, 3
            assert img_pred.min() >= 0 and img_pred.max() <= 1 
            
            if save_img:
                # set background as white for visualization
                #mask = data.get('mask_img')[0].cpu().data.numpy().reshape(img_pred.shape[0], img_pred.shape[1], 1)
                #saved = img_pred*mask+1-mask
                saved = img_pred
                imageio.imwrite(os.path.join(path, "%06d.png" %indexes[i]),  (saved[..., :3]*255).astype(np.uint8))

            img_gt = data.get('img').float()    # 1, 3, W, H
            
            assert img_gt.min() >= 0 and img_gt.max() <= 1 
            
            # normalize with [-1, 1]
            img_pred = torch.tensor(np.transpose(img_pred, [2,0,1])).float().unsqueeze(0).to(trainer.device)
            img_pred_norm = ((img_pred-img_pred.min())/(img_pred.max() - img_pred.min()))*2 - 1
            img_gt_norm = (((img_gt-img_gt.min())/(img_gt.max() - img_gt.min()))*2 - 1)
            eval_dict["lpips_alex"] += trainer.lpips_alex_loss(img_pred_norm, img_gt_norm).item()
            eval_dict["lpips_vgg"] += trainer.lpips_vgg_loss(img_pred_norm.float(), img_gt_norm.float()).item()

            eval_dict["l1"] += trainer.l1_loss(img_pred.float(), torch.tensor(img_gt).float()).item()
            mse = trainer.l2_loss(img_pred, img_gt)
            eval_dict["l2"] += mse.item()
            eval_dict["psnr"] += mse2psnr(mse).item()


    logger_py.info('Evaluation 2D')
    for k, v in eval_dict.items():
        trainer.tb_logger.add_scalar(f"val/{k}", v/n, epoch_it)
        logger_py.info('%s = %.4g' %(k, v/n))

    # evaluate 3D
    if cfg["data"]["dataset_name"] != "blender":
        pointcloud = trainer.model.get_point_clouds(with_colors=True, with_normals=True,require_normals_grad=False, cameras=val_cameras)
        pointcloud_tgt = val_loader.dataset.get_pointclouds(num_points=trainer.n_eval_points).to(device=pointcloud.device)
        cd_p, cd_n = chamfer_distance(pointcloud_tgt, pointcloud, x_lengths=pointcloud_tgt.num_points_per_cloud(), y_lengths=pointcloud.num_points_per_cloud(),)
        
        dist = directed_hausdorff(pointcloud.points_packed().detach().cpu(), pointcloud_tgt.points_packed().detach().cpu())[0]
        
        logger_py.info('Evaluation 3D')
        logger_py.info('%s = %.4g' %('hausdorff_distance', dist))
        logger_py.info('%s = %.4g' %('chamfer_point', cd_p.item()))
        logger_py.info('%s = %.4g' %('chamfer_normal', cd_n.item()))
        
        if save_pc:
            o3d_pc = trainer.generator.generate_pointclouds(data, cameras=val_cameras)[0]
            o3d_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius = 0.1, max_nn = 30))
            o3d_pc.orient_normals_consistent_tangent_plane(k=10)
            o3d.io.write_point_cloud(os.path.join(trainer.vis_dir, str(epoch_it)+"pc.ply"), o3d_pc)



if __name__ == "__main__":

    cfg = get_configurations()
    main(cfg)


