from collections import OrderedDict, defaultdict
import copy
import datetime
import imageio
from PIL import Image
import os
import pdb
import numpy as np
import time
import cv2
import trimesh
import open3d as o3d
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torchvision import transforms 
from scipy.stats import expon
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
from pytorch3d.loss import chamfer_distance


import lpips
from dependencies.CLIP import clip_utils

# custom scripts
from .. import set_debugging_mode_, get_debugging_tensor, logger_py
from src.utils import slice_dict, check_weights, slice_dict_data
from src.utils.mathHelper import decompose_to_R_and_t
from src.training.losses import ( IouLoss, ProjectionLoss, RepulsionLoss, SemanticConsistencyLoss, L2Loss, L1Loss, SmapeLoss, NormalLoss, NormalLengthLoss, grad_loss, vivi_loss)
from src.training.scheduler import TrainerScheduler
from src.misc import Thread
from src.misc.visualize import plot_2D_quiver, plot_3D_quiver
from src.training.clip_encoder import get_embed_fn, resize_img_for_clip
from src.models import PointModel
from src.core.camera import CameraSampler
from utils.common import get_light_for_view, normalize_unit_sphere, get_rays
from utils.config import create_scheduler, create_scaler, create_optimizer

def create_trainer(cfg, model, scaler, optimizer, scheduler, generator, val_loader, device, sem_cons_dict, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
        generator (Generator): generator instance to
            generate meshes for visualization
    '''
    out_dir = os.path.join(cfg['generic']['output_dir'])
    vis_dir = os.path.join(out_dir, 'vis')
    debug_dir = os.path.join(out_dir, 'debug')
    log_dir = os.path.join(out_dir, 'logs')
    val_dir = os.path.join(out_dir, 'val')

    trainer = Trainer(cfg=cfg, model=model, scaler=scaler, optimizer=optimizer, scheduler=scheduler, generator=generator, val_loader=val_loader, sem_cons_dict=sem_cons_dict,
        device=device, vis_dir=vis_dir, debug_dir=debug_dir, log_dir=log_dir, val_dir=val_dir)

    return trainer


class BaseTrainer(object):
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, *args, **kwargs):
        self.model = model
        self.optimizer = optimizer

    def forward(self, *args, mode="train", **kwargs):
        """One forward pass, returns all necessary outputs to getting the losses or evaluation """
        raise NotImplementedError

    def train_step(self, *args, **kwargs):
        ''' Performs a training step.'''
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        """ Performs a evaluation step """
        raise NotImplementedError

    def compute_loss(self, *args, **kwargs):
        """ Returns the training loss (a scalar)  """
        raise NotImplementedError

    def evaluate(self, val_dataloader, reduce=True, **kwargs):
        """Make models eval mode during test time"""
        eval_list = defaultdict(list)

        for data in tqdm(val_dataloader):
            eval_step_dict = self.eval_step(data, **kwargs)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: torch.stack(v) for k, v in eval_list.items()}
        if reduce:
            eval_dict = {k: torch.mean(v) for k, v in eval_dict.items()}
        return eval_dict

    def update_learning_rate(self):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def __init__(self, cfg, model, scaler, optimizer, scheduler, generator, val_loader, device='cpu',
                 sem_cons_dict=None, log_dir=None, vis_dir=None, debug_dir=None, val_dir=None, **kwargs):
        """Initialize the BaseModel class.
        Args:
            model (nn.Module)
            optimizer: optimizer
            scheduler: scheduler
            device: device
        """
        self.cfg = cfg
        self.device = device
        self.model = model
        self.val_loader = val_loader
        self.tb_logger = SummaryWriter(log_dir)

        # implicit function model
        self.vis_dir = vis_dir
        self.val_dir = val_dir

        # losses
        self.lambda_proj_loss = cfg["training"]["lambda_proj_loss"]
        self.projection_loss = ProjectionLoss(reduction='mean', filter_scale=cfg["training"]["repel_proj_scale"], knn_k=cfg["training"]["repel_proj_knn_k"])

        self.lambda_repel_loss = cfg["training"]["lambda_repel_loss"]
        self.repulsion_loss = RepulsionLoss(reduction='mean', filter_scale=cfg["training"]["repel_proj_scale"], knn_k=cfg["training"]["repel_proj_knn_k"])
        
        
        self.iou_loss = IouLoss(reduction='mean', channel_dim=None)
        self.l1_loss = L1Loss(reduction='mean')
        self.l2_loss = L2Loss(reduction='mean')
        self.normal_loss = NormalLoss(reduction = 'mean', neighborhood_size=cfg["training"]["normal_neighboor_size"])
        self.normal_lenght_loss = NormalLengthLoss(reduction = 'mean', neighborhood_size=cfg["training"]["normal_neighboor_size"])
        
        self.lambda_smape_loss = cfg["training"]["lambda_smape_loss"]
        self.smape_loss = SmapeLoss(reduction='mean')

        self.lambda_mse_rgb_loss = cfg["training"]["lambda_mse_rgb_loss"]
        self.lambda_mse_a_loss = cfg["training"]["lambda_mse_a_loss"]
        self.lambda_mse_rgba_loss = cfg["training"]["lambda_mse_rgba_loss"]

        self.lambda_smape_loss = cfg["training"]["lambda_smape_loss"]
        self.lambda_normal_smoothness_loss = cfg["training"]["lambda_normal_smoothness_loss"]
        self.lambda_variance_loss = cfg["training"]["lambda_variance_loss"]

        
        self.lpips_alex_loss = lpips.LPIPS(net='alex').to(self.device) # best forward scores
        self.lpips_vgg_loss = lpips.LPIPS(net='vgg').to(self.device) # closer to "traditional" perceptual loss, when used for optimization


        # evaluation 
        self.generator = generator
        self.n_eval_points = cfg["model"]["n_points_per_cloud"]

        #  tuple (score, mesh)
        init_dss_backward_radii = self.model.renderer.rasterizer.raster_settings["radii_backward_scaler"]
        self.training_scheduler = TrainerScheduler(init_dss_backward_radii=init_dss_backward_radii,
                                                steps_dss_backward_radii=self.cfg["dss"]["steps_dss_backward_radii"],
                                                limit_dss_backward_radii=self.cfg["dss"]["limit_dss_backward_radii"],
                                                steps_proj=self.cfg.get('steps_proj', -1),
                                                gamma_proj=self.cfg.get('gamma_proj', 5))

        # debug
        self.debug_dir = debug_dir
        self.hooks = []
        self._mesh_cache = None


        # training
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler

        
        # semantic consistency loss
        self.lambda_sem_cons_loss = cfg["training"]["lambda_sem_cons_loss"]
        if self.lambda_sem_cons_loss > 0:
            self.l2_loss_sem = L2Loss(reduction='mean')
            self.f_sem_cons_loss = SemanticConsistencyLoss(reduction='mean')

            # to adjust the frequency of semantic loss computation
            self.sem_cons_every = cfg["training"]["sem_cons_every"]
            self.sem_cons_epoch_start = cfg["training"]["sem_cons_epoch_start"]
            self.last_sem_loss = 0

            # for closest gt comparaison
            self.n_of_closest_gt_to_compare = cfg["training"]["sem_cons_n_of_close_cam"]

            # curriculum learning
            self.use_curriculum_learning = cfg["training"]["curriculum_learning"]
            self.minim_loss = float('inf')
            
            # clip encode generated images
            self.consistency_model_type = cfg["training"]["clip_model_type"]
            self.nn_clip_embed = cfg["training"]["init_clip"]
            
            # list of target clip embeddings and normalized clip images
            self.target_emb = sem_cons_dict["target_emb"]
            self.clip_images = sem_cons_dict["clip_images"]

            # KD tree of target camera locations
            self.tree = sem_cons_dict["tree"]

            # validation views
            self.new_cam_count = 0 if self.use_curriculum_learning else cfg["training"]["sem_cons_n_of_add_images"]
            if cfg["training"]["sem_cons_n_of_add_images"] > 0 :
                self.new_camera_mat, self.new_camera_loc, self.new_view_dir = [], [], []

                camera_sampler = CameraSampler( num_cams_total=cfg["training"]["sem_cons_n_of_add_images"], 
                                                num_cams_batch= 1,  
                                                distance_range=torch.tensor(((cfg["data"]["camera_min_dist"], cfg["data"]["camera_max_dist"]),)), 
                                                sort_distance=True, 
                                                camera_type=eval(cfg["data"]["camera_type"]), 
                                                camera_params={"znear": cfg["data"]["camera_znear"], "device": self.device})
                for cams in camera_sampler:
                    cams = cams.to(self.device)
                    world_mat = cams.get_world_to_view_transform().get_matrix().to(self.device)
                    camera_loc = cams.get_camera_center().cpu()
                    self.new_camera_mat.append(world_mat)
                    self.new_camera_loc.append(camera_loc)
                    cam_K = cams.get_projection_transform().get_matrix().to(self.device)
                    rays_o, rays_d = get_rays(self.cfg["dss"]["raster_settings"]["image_size"], self.cfg["dss"]["raster_settings"]["image_size"], cam_K[0, :3, :3], world_mat[0])
                    self.new_view_dir.append(torch.nn.functional.normalize(rays_d, dim=2))


    def evaluate_3d(self, val_loader, it, **kwargs):
        logger_py.info("[3D Evaluation]")
        t0 = time.time()
        os.makedirs(self.val_dir, exist_ok=True)

        # create mesh using generator
        pointcloud = self.model.get_point_clouds(with_colors=True, with_normals=True,require_normals_grad=False, **kwargs)
        pointcloud_tgt = val_loader.dataset.get_pointclouds(num_points=self.n_eval_points).to(device=pointcloud.device)
        cd_p, cd_n = chamfer_distance(pointcloud_tgt, pointcloud, x_lengths=pointcloud_tgt.num_points_per_cloud(), y_lengths=pointcloud.num_points_per_cloud(),)
        
        # save to "val" dict
        t1 = time.time()
        eval_dict = {'chamfer_point': cd_p.item(), 'chamfer_normal': cd_n.item()}
        logger_py.info('(%s): %.4g, (%s): %.4g, time=%.4f' %('chamfer_point', cd_p.item(), 'chamfer_normal', cd_n.item(), t1 - t0))
        if not pointcloud.isempty():
            self.tb_logger.add_mesh('eval',pointcloud.points_padded(), global_step=it)
        return eval_dict

    def reset_optimizer(self, cfg):
            self.optimizer = create_optimizer(cfg, self.model)
            self.scheduler = create_scheduler(cfg["training"]["scheduler"], self.optimizer)
            self.scaler = create_scaler()

    def eval_step(self, data, eval_dict, val_loader, it, batch_it, **kwargs):
        """
        evaluate with image mask iou or image rgb psnr
        """
        metrics_list = list(eval_dict.keys())

        with torch.autograd.no_grad():
            self.model.eval()

            if 'chamfer_point' in metrics_list and batch_it==0 :
                res_dict = self.evaluate_3d(val_loader, it, **kwargs)
                eval_dict = {**eval_dict, **res_dict}

            if ('psnr' in metrics_list or 'iou' in metrics_list) :

                img_mask = data['mask_img']
                img = data['img']

                # render image
                rgbas = self.generator.generate_images(data, locations=data['locations'], cameras=data['camera'])[0]/255
                rgba = torch.tensor(rgbas[None, ...], dtype=torch.float, device=img_mask.device).permute(0, 3, 1, 2)[0]

                # # compare iou
                # mask_gt = F.interpolate(img_mask.float(), img_size, mode='bilinear', align_corners=False).squeeze(1)
                # mask_pred = rgba[:, 3, :, :]
                # eval_dict['iou'] += self.iou_loss(mask_gt.float(), mask_pred.float(), reduction='mean')

                # compare psnr
                # rgb_gt = F.interpolate(img, img_size, mode='bilinear', align_corners=False)
                # img_pred = rgba[:3, :, :]
                # eval_dict['psnr'] += self.l2_loss( img, img_pred, channel_dim=1, reduction='mean', align_corners=False).detach().item()

                # # # compare FID score
                # pdb.set_trace()
                # images1 = preprocess_input(rgbas)
                # images2 = preprocess_input(img.reshape(rgbas.shape).detach().cpu().numpy())
                # fid = calculate_fid(self.model.inception, images1[None, ...], images2[None, ...])


        return eval_dict


    def train_step(self, data, cameras, image_logs, only_base, **kwargs):
        """
        Args:
            data (dict): contains img, img.mask and img.depth and camera_mat
            cameras (Cameras): Cameras object from pytorch3d
        Returns:
            loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        it = kwargs.get("it", None)
        epoch = kwargs.get("epoch", None)
        if hasattr(self, 'training_scheduler'):
            self.training_scheduler.step(self, it)

        # pre process
        data = self.process_data_dict(data, cameras, epoch=epoch)
        batch_size = data['img'].shape[0]
        assert data['img'].min() >= 0 and data['img'].max() <= 1 
        assert data['mask_img'].min() >= 0 and data['mask_img'].max() <= 1 
        n_of_GT, _, h, w = data['img'].shape

        # forward
        model_outputs = self.model(n_of_GT=n_of_GT, view=data['view'], mask_img=data['mask_img'], cameras=data['camera'], locations=data['camera_loc'], verbose=False, epoch=epoch, it=it, only_base=only_base)

        # loss
        loss = self.compute_loss(data['img'], data['mask_img'], model_outputs, data['camera_loc'], it=it, epoch=epoch)
        avg_loss = loss/batch_size
        
        # Backprop
        self.scaler.scale(avg_loss).backward()

        # curriculum learning for semantic consistency loss
        if self.lambda_sem_cons_loss > 0 and self.use_curriculum_learning and self.sem_cons_epoch_start < epoch:
            this_loss = avg_loss.detach().item()

            # add a new pose at each n epochs
            if epoch > 0 and epoch % self.sem_cons_every == 0 :
                self.new_cam_count += 1

        # Gradient clipping
        # self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.5, norm_type=2)

        # Logging
        total_norm = 0
        for p_name, p in self.model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                self.tb_logger.add_scalar(f"grad/norm_{p_name}", param_norm.item(), it)
                total_norm += param_norm.item() ** 2
            elif it/batch_size == epoch:
                print("Gradients for "+ p_name + " are None")
                continue

        total_norm = total_norm ** 0.5
        self.tb_logger.add_scalar(f"grad/norm_tot", total_norm, it)
        
        if image_logs:
            idx = kwargs.get("idx", None)
            mask_img_pred = model_outputs.get('mask_img_pred')
            img_pred = model_outputs.get('img_pred')
            harmonics = model_outputs.get('harmonics')

            # RGB
            log_rgb_pred = (255*img_pred.detach().cpu().numpy()).astype("uint8")
            log_rgb_gt = (255*data['img'].permute(0, 3, 2, 1).permute(0, 2, 1, 3).detach().cpu().numpy()).astype("uint8")

            # mask
            log_mask_pred = (255*mask_img_pred.repeat(1,1,1,3).detach().cpu().numpy()).astype("uint8")
            log_mask_gt = (255*data['mask_img'].permute(0, 3, 2, 1).permute(0, 2, 1, 3).repeat(1,1,1,3).detach().cpu().numpy()).astype("uint8")

            # merged RGB
            #log_pred = (255*rgba2rgb(torch.cat([img_pred, mask_img_pred], dim=3), self.device).detach().cpu().numpy()).astype("uint8")
            #log_gt =  (255*rgba2rgb(torch.cat([data['img'], data['mask_img']], dim=1).permute(0, 3, 2, 1).permute(0, 2, 1, 3), self.device).detach().cpu().numpy()).astype("uint8")

            for i, id in enumerate(idx) :
                self.tb_logger.add_image("rgb/"+str(id.item()), cv2.hconcat([log_rgb_pred[i], log_rgb_gt[i]]), global_step=epoch, dataformats='HWC')
                os.makedirs(os.path.join(self.vis_dir, str(id.item())), exist_ok=True)
                imageio.imwrite(os.path.join(self.vis_dir, str(id.item()), "%06d.png"%epoch), log_rgb_pred[i])
                self.tb_logger.add_image("mask/"+str(id.item()), cv2.hconcat([log_mask_pred[i], log_mask_gt[i]]), global_step=epoch, dataformats='HWC')
                os.makedirs(os.path.join(self.vis_dir, str(id.item())+"_mask"), exist_ok=True)
                imageio.imwrite(os.path.join(self.vis_dir, str(id.item())+"_mask", "%06d.png"%epoch), log_mask_pred[i])
                if harmonics is not None:
                    self.tb_logger.add_image("harmonics/"+str(id.item()), harmonics[i], global_step=epoch, dataformats='HWC')
                #self.tb_logger.add_image("train/"+str(id.item())+"/merged", cv2.hconcat([log_pred[i], log_gt[i]]), global_step=epoch, dataformats='HWC')

        # Optimizer/scheduler steps
        self.scaler.step(self.optimizer)
        scale = self.scaler.get_scale()
        self.scaler.update()
        self.skip_lr_sched = (scale > self.scaler.get_scale())

        # Additional check for inf/nan 
        check_weights(self.model.state_dict())

        return avg_loss

    def process_data_dict(self, data, cameras, epoch=None, visualize=False, verbose=False):
        ''' Processes the data dictionary and returns respective tensors
        Args:
            data (dictionary): data dictionary
        '''
        device = self.device
        self.epoch = epoch

        # Get target data
        img = data.get('img.rgb').to(device)
        mask_img = data.get('img.mask').to(device)
        view = data.get('img.view').to(device)

        assert(img.min() >= 0 and img.max() <=1), "Image must be a floating number between 0 and 1."
        assert(mask_img.min() >= 0 and mask_img.max() <=1), "Image must be a floating number between 0 and 1."

        camera_mat = data.get('camera_mat', None)
        camera_loc = data.get('camera_loc', None)
        
        # Increase prediction count for semantic consistency loss
        if not visualize and self.lambda_sem_cons_loss > 0 and (epoch % self.sem_cons_every) == 0 and self.sem_cons_epoch_start < epoch :

            # slicing of validation poses by idx
            if self.cfg["training"]["sem_cons_n_of_add_images"] > 0:
                self.new_cam_count = min(len(self.new_camera_mat), self.new_cam_count)
                # indexes = list(torch.randint(0, self.new_cam_count, (1, img.shape[0])).squeeze(0))
                indexes = list(torch.randint(0, self.new_cam_count, (1, 2)).squeeze(0))
                # indexes = list(np.arange(0, self.new_cam_count))
                if self.new_cam_count > 0:
                    new_camera_mat,  new_camera_loc , new_view_dir = [], [], []
                    for i in indexes : 
                        new_view_dir.append(self.new_view_dir[i])
                        new_camera_mat.append(self.new_camera_mat[i])
                        new_camera_loc.append(self.new_camera_loc[i].to(self.device))
                    
                    camera_mat = torch.cat([camera_mat,  *new_camera_mat], dim=0)
                    camera_loc = torch.cat([camera_loc,  *new_camera_loc], dim=0)
                    view = torch.cat([view, torch.stack(new_view_dir)], dim=0)
    
        # set camera matrix to cameras
        if camera_mat is None:
            logger_py.warning("Camera matrix is not provided! Using the default matrix")
        else:
            cameras.R, cameras.T = decompose_to_R_and_t(camera_mat)
            cameras._N = cameras.R.shape[0]
            cameras.to(device)

        return {'img': img, 'mask_img': mask_img, 'view': view, 'camera': cameras, 'camera_loc': camera_loc}


    def compute_loss(self, img, mask_img, model_outputs, camera_loc, eval_mode=False, it=None, epoch=None):
        ''' Compute the loss.
        Args:
            data (dict): data dictionary
            eval_mode (bool): whether to use eval mode
            it (int): training iteration
        '''

        assert img.min() >= 0 and img.max() <= 1 
        assert mask_img.min() >= 0 and mask_img.max() <= 1 

        loss = {}
        n_of_GT, _, h, w = img.shape
        loss['loss'] = 0

        point_clouds = model_outputs.get('iso_pcl')
        mask_img_pred = model_outputs.get('mask_img_pred')
        img_pred = model_outputs.get('img_pred')
        
        assert img_pred.min() >= 0 and img_pred.max() <= 1 
        assert mask_img_pred.min() >= 0 and mask_img_pred.max() <= 1 

        img = img.permute(0, 2, 3, 1)
        mask_img=mask_img.reshape(-1, h, w)
        mask_img_pred = mask_img_pred.reshape(-1, h, w)

        assert img.shape == img_pred[:n_of_GT].shape
        assert mask_img.shape == mask_img_pred[:n_of_GT].shape

        # pixel loss
        if (self.lambda_mse_a_loss + self.lambda_mse_rgb_loss + self.lambda_smape_loss + self.lambda_mse_rgba_loss) > 0:
            self.calc_dr_loss(  img=img, img_pred=img_pred[:n_of_GT, ...], 
                                mask_img=mask_img, 
                                mask_img_pred=mask_img_pred[:n_of_GT, ...], 
                                reduction_method='mean', loss=loss , it=it)

        # point cloud loss
        if (self.lambda_proj_loss + self.lambda_repel_loss) > 0:
            self.calc_pcl_reg_loss( point_clouds, reduction_method='mean', loss=loss, it=it)

        # normal dloss
        if self.lambda_normal_smoothness_loss > 0:
            self.calc_normal_smoothness_loss(point_clouds, reduction_method='mean', loss=loss, it=it)

        # needs to be at the end for correct computation of curriculum learning
        if self.lambda_sem_cons_loss > 0 and self.sem_cons_epoch_start < epoch:
            self.calc_sem_cons_loss(img=img, 
                                    img_pred=img_pred,  
                                    mask=mask_img.unsqueeze(3), 
                                    mask_pred=mask_img_pred.unsqueeze(3),
                                    camera_loc=camera_loc, loss=loss, it=it, epoch=epoch)
            
            self.tb_logger.add_scalar("val/add_views", self.new_cam_count, it)


        if self.lambda_variance_loss > 0 and img.shape[0] > 3 and mask_img.shape[0] > 3:
            self.calc_variance_loss(img=img, img_pred=img_pred[:n_of_GT, ...], 
                                mask=mask_img, 
                                mask_pred=mask_img_pred[:n_of_GT, ...], loss=loss , it=it)

        # logging
        for k, v in loss.items():
            mode = 'val' if eval_mode else 'train'
            if isinstance(v, torch.Tensor):
                self.tb_logger.add_scalar('%s/%s' % (mode, k), v.item(), it)
            else:
                self.tb_logger.add_scalar('%s/%s' % (mode, k), v, it)

        return loss if eval_mode else loss['loss']

    def calc_normal_smoothness_loss(self, point_clouds, reduction_method='mean', loss={}, **kwargs):
        """
        Args:
            point_clouds (PointClouds3D): point clouds in source space (object coordinate)
        """
        loss_normals, loss_Lnormals = 0, 0

        if self.lambda_normal_smoothness_loss > 0:
            loss_normals  = self.normal_loss(point_clouds)
            loss_Lnormals = self.normal_lenght_loss(point_clouds.normals_packed())

        loss['loss'] = loss_normals * self.lambda_normal_smoothness_loss + loss_Lnormals * self.lambda_normal_smoothness_loss + loss['loss']
        loss['loss_normals'] = loss_normals
        loss['loss_Lnormals'] = loss_Lnormals

    def calc_variance_loss(self, img, img_pred, mask, mask_pred, loss={}, **kwargs):
        """
        Args:
            point_clouds (PointClouds3D): point clouds in source space (object coordinate)
        """
        loss_variance = 0, 0

        if self.lambda_variance_loss > 0:
            loss_variance = grad_loss(torch.cat([img, mask.unsqueeze(3)], dim=3), torch.cat([img_pred, mask_pred.unsqueeze(3)],  dim=3))
        
        loss['loss_variance'] = loss_variance
        loss['loss'] = loss_variance * self.lambda_variance_loss +  loss['loss']


    def calc_pcl_reg_loss(self, point_clouds, reduction_method='mean', loss={}, **kwargs):
        """
        Args:
            point_clouds (PointClouds3D): point clouds in source space (object coordinate)
        """
        loss_dr_repel = 0
        loss_dr_proj = 0
        if self.lambda_proj_loss > 0:
            loss_dr_proj  = self.projection_loss(point_clouds, rebuild_knn=True, points_filter=self.model.points_filter)
        if self.lambda_repel_loss > 0:
            loss_dr_repel = self.repulsion_loss(point_clouds, rebuild_knn=True, points_filter=self.model.points_filter)

        loss['loss'] = loss_dr_proj * self.lambda_proj_loss + loss_dr_repel * self.lambda_repel_loss + loss['loss']
        loss['loss_proj'] = loss_dr_proj
        loss['loss_repel'] = loss_dr_repel


    def calc_sem_cons_loss(self, img, img_pred, mask, mask_pred, camera_loc, it, loss={}, epoch=0, **kwargs):
        """
        Calculates  semantic consistency loss

        Args:
            img (tensor): (N,H,W,C) range [0, 1]
            img_pred (tensor): (N,H,W,C) range [0, 1]

            mask (tensor): (N,H,W,C) range [0, 1]
            mask_pred (tensor): (N,H,W,C) range [0, 1]

        """
        
        n_of_losses, loss_sem_cons, loss_pred_variance = 0, self.last_sem_loss, 0

        if epoch % self.sem_cons_every == 0 :

            loss_sem_cons = 0
            
            assert img.min() >= 0 and img.max() <= 1 
            assert mask.min() >= 0 and mask.max() <= 1
            
            assert img_pred.min() >= 0 and img_pred.max() <= 1  
            assert mask_pred.min() >= 0 and mask_pred.max() <= 1 
            assert img_pred.shape[0] == mask_pred.shape[0] and camera_loc.shape[0]

            n_of_GT = img.shape[0]
            n_of_PRED = img_pred.shape[0]

            for i in range(n_of_GT, n_of_PRED):

                # resize and embed prediction
                pred_image_i_rgb = rgba2rgb(torch.cat([img_pred[i], mask_pred[i]], dim=2), device=self.device)  # [512, 512, 3]
                pred_image_i_norm = resize_img_for_clip(pred_image_i_rgb) # [1, 3, 224, 224]
                log_image = (pred_image_i_norm*255).detach().cpu().numpy()[0].astype("uint8") # for logging
                #pe = torch.tensor(pred_image_i_norm*255, dtype=torch.uint8).cpu()
                rendered_emb_i = self.nn_clip_embed(pred_image_i_norm)[0]

                # find prediction closest GT image from the tree
                array_of_closest_gt_distances, array_of_closest_gt_indices = self.tree.query(camera_loc[i:i+1].detach().cpu(), k=min(self.tree.data.shape[0],self.n_of_closest_gt_to_compare))
                #indexes = list(torch.randint(0, self.n_of_closest_gt_to_compare, (1, 1)).squeeze(0))
                #array_of_closest_gt_indices = array_of_closest_gt_indices[:, indexes]
                
                # if self.n_of_closest_gt_to_compare == 1 and i < n_of_GT:
                #     assert array_of_closest_gt_distances.max() == 0

                # accumulate the semantic losses
                for closest_indices in array_of_closest_gt_indices:
                    for index in closest_indices:
                        if self.consistency_model_type =="clip_rn50":
                            assert self.target_emb[index][0, :].shape == rendered_emb_i.shape
                            loss_sem_cons += -torch.nn.CosineSimilarity(dim=1, eps=1e-08)(self.target_emb[index][0,:].float(), rendered_emb_i.float()).mean()

                        elif self.consistency_model_type =="clip_vit":
                            assert self.target_emb[index][0, 0, :].shape == rendered_emb_i[0].shape
                            loss_sem_cons += self.f_sem_cons_loss(self.target_emb[index][0, 0, :].float(), rendered_emb_i[0].float())
                        else:
                            raise ValueError

                        # log the pair comparaison of normalized images before the embedding
                        # loss_pred_variance += vivi_loss(pe.permute(0,2,3,1).float(), torch.tensor(self.clip_images[index]).permute(0,2,3,1).float(), verbose=True)
                        self.tb_logger.add_image("CLIP/#GT"+str(index), cv2.hconcat([log_image, self.clip_images[index][0]]), global_step=it, dataformats='CHW')
                        
                        # increase counter
                        n_of_losses += 1


            # average the semantic loss
            loss_sem_cons = loss_sem_cons/n_of_losses
            loss_pred_variance = loss_pred_variance/n_of_losses
            self.last_sem_loss = loss_sem_cons.detach()

        loss['loss_pred_variance'] = loss_pred_variance
        loss['loss_sem_cons'] = loss_sem_cons
        loss['loss'] = loss['loss'] + loss_sem_cons * self.lambda_sem_cons_loss + loss_pred_variance*0.001



    def calc_dr_loss(self, img, img_pred, mask_img, mask_img_pred, it, reduction_method='mean', loss={}, **kwargs):
        """
        Calculates image loss
        Args:
            img (tensor): (N,H,W,C) range [0, 1]
            img_pred (tensor): (N,H,W,C) range [0, 1]
            mask_img (tensor): (N,H,W) range [0, 1]
            mask_img_pred (tensor): (N,H,W) range [0, 1]
        """

        assert(img.shape == img_pred.shape), "Ground truth mage shape and predicted image shape is unequal"


        if self.lambda_smape_loss > 0:
            loss_smape = self.smape_loss(x=torch.cat([img, mask_img.unsqueeze(3)], dim=3), 
                                        y=torch.cat([img_pred, mask_img_pred.unsqueeze(3)],  dim=3),  
                                        reduction=reduction_method)
            loss['loss_smape'] = loss_smape
            loss['loss'] = loss_smape * self.lambda_smape_loss +  loss['loss']

        if self.lambda_mse_rgba_loss > 0:

            loss_rgba = self.l2_loss(   x=torch.cat([img_pred, mask_img_pred.unsqueeze(3)], dim=3), 
                                        y=torch.cat([img, mask_img.unsqueeze(3)], dim=3), 
                                        reduction=reduction_method)
            
            loss['loss_rgba'] = loss_rgba
            loss['loss'] = loss_rgba * self.lambda_mse_rgba_loss + loss['loss']


        if self.lambda_mse_rgb_loss > 0:
            loss_rgb = self.l1_loss(x=img, y=img_pred, reduction=reduction_method)

            loss['loss_rgb'] = loss_rgb
            loss['loss'] = loss_rgb * self.lambda_mse_rgb_loss +  loss['loss']


        if self.lambda_mse_a_loss > 0:
            loss_mask = (mask_img.float() - mask_img_pred).abs()
            loss_mask = loss_mask.mean()

            loss_iou = self.iou_loss(mask_img.float(), mask_img_pred)
            loss_dr_silhouette = (0.01 * loss_iou + loss_mask)

            loss['loss_dr_silhouette'] = loss_dr_silhouette
            loss['loss'] =loss_dr_silhouette * self.lambda_mse_a_loss  +  loss['loss']

    def visualize(self, data, cameras, locations, step=0, vis_type='mesh', name="train/img", save=False, log=False, **kwargs):
        ''' Visualized the data.

        Args:
            data (dict): data dictionary
            step (int): training iteration
            vis_type (string): visualization type
        '''
        
        # visualize the rendered image and pointcloud
        if vis_type == 'image':
            img_list = self.generator.generate_images(data, locations=locations,  cameras=cameras, **kwargs)
            img_gt = (np.transpose(data.get('img')[:3, ...].cpu().numpy(), [2,1,0])*255).astype("uint8")

            for i, img in enumerate(img_list):
                if save:
                    imageio.imwrite(name,  img[..., :3])
                if log:
                    self.tb_logger.add_image(name+str(i), cv2.hconcat([img[..., :3], np.transpose(img_gt, [1,0,2])]), global_step=step, dataformats='HWC')

        elif vis_type == 'pointcloud':
            pcl_list = self.generator.generate_pointclouds(data, cameras=cameras, **kwargs)
            camera_threejs = {}
            if isinstance(cameras, FoVPerspectiveCameras):
                camera_threejs = {  'cls': 'PerspectiveCamera', 
                                    'fov': cameras.fov.item(),
                                    'far': cameras.zfar.item(), 
                                    'near': cameras.znear.item(),
                                    'aspect': cameras.aspect_ratio.item()}

            for i, pcl in enumerate(pcl_list):
                if save:
                    o3d.io.write_point_cloud(os.path.join(self.vis_dir, name), pcl)

                elif isinstance(pcl, trimesh.Trimesh):
                    self.tb_logger.add_mesh(name,  np.array(pcl.vertices)[None, ...],config_dict=camera_threejs,global_step=step)

        elif vis_type == 'mesh':
            mesh = self.generator.generate_mesh(data, with_colors=False, with_normals=False)
            camera_threejs = {}
            if isinstance(cameras, FoVPerspectiveCameras):
                camera_threejs = {  'cls': 'PerspectiveCamera', 
                                    'fov': cameras.fov.item(),
                                    'far': cameras.far.item(), 
                                    'near': cameras.near.item(),
                                    'aspect': cameras.aspect_ratio.item()}
            if isinstance(mesh, trimesh.Trimesh):
                self.tb_logger.add_mesh(name, np.array(mesh.vertices)[None, ...],
                                        faces=np.array(mesh.faces)[None, ...],
                                        config_dict=camera_threejs, global_step=step)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def update_learning_rate(self, it, metric=None):
        """Update learning rates for all modifiers from DietNerf
        """

        if not self.skip_lr_sched:
            self.scheduler.step(metric)
            
        for param_group in self.optimizer.param_groups:
            self.tb_logger.add_scalar('train/lr_' +param_group["name"], param_group["lr"], it)

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ExponentialLR):
            return

        decay_steps = self.cfg["training"]["lr_decay_steps"] * 1000
        new_lrate = self.cfg["training"]["lr"] * (self.cfg["training"]["lr_decay_rate"] ** (it / decay_steps))

        for param_group in self.optimizer.param_groups:
            if param_group['lr'] > new_lrate : 
                param_group['lr'] = new_lrate
            else: 
                self.cfg["training"]["lr"] = param_group['lr']



def rgba2rgb( rgba, device, background=(1,1,1), normalized=True):
    if len(rgba.shape) == 4:
        n_im, row, col, ch = rgba.shape
    elif  len(rgba.shape) == 3:
        row, col, ch = rgba.shape
    else:
        pdb.set_trace()
    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = torch.zeros_like(rgba[...,:3], dtype=torch.float32).to(device)
    r, g, b, a = rgba[...,0]*255, rgba[...,1]*255, rgba[...,2]*255, rgba[...,3]

    R, G, B = background

    rgb[...,0] = r * a + (1.0 - a) * R
    rgb[...,1] = g * a + (1.0 - a) * G
    rgb[...,2] = b * a + (1.0 - a) * B

    if normalized:
        return rgb/255
    else:
        return rgb
