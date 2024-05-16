import argparse
import hashlib
import itertools
import math
import lpips
import json
import datetime
import random
import os, pdb
from pathlib import Path
from typing import Optional
import torchvision.transforms.functional as fn
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import cv2
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DModel, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from dependencies.SD.pipe_denoising import StableDiffusionImg2ImgDenoisingPipeline
from dependencies.CLIP import clip_utils
from src.training.clip_encoder import get_embed_fn, resize_img_for_clip


auth_token = "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"
model_id_or_path = "/data/storage/bpietro/huggingface/diffusers/models--runwayml--stable-diffusion-v1-5/snapshots/3beed0bcb34a3d281ce27bd8a6a1efbb68eada38/"
logger = get_logger(__name__)
torch.autograd.set_detect_anomaly(True) 

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default=model_id_or_path,help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--revision", type=str, default=None, help="Revision of pretrained model identifier from huggingface.co/models.",)
    
    parser.add_argument("--tokenizer_name", type=str, default=os.path.join(model_id_or_path, "tokenizer"), help="Pretrained tokenizer name or path if not the same as model_name",)
    parser.add_argument("--instance_data_dir",type=str, default="/data/storage/bpietro/thesis/DSS/output/gt/uniform/vis/val_gt/", required=True, help="A folder containing the training data of instance images.",)
    parser.add_argument("--class_data_dir",type=str, default=None, required=False,help="A folder containing the training data of class images.",)
    
    parser.add_argument("--instance_prompt", type=str, default=None, required=True, help="Either a path to a JSON or a prompt with identifier specifying the instance",    )
    parser.add_argument("--multi_instance_prompt_file", type=str, default=None, help="A path to a JSON where the properties identify the s specifying the instance",    )
    parser.add_argument("--class_prompt",type=str, default=None, help="The prompt to specify images in the same class as provided instance images.",)
    
    parser.add_argument("--with_prior_preservation", default=False, action="store_true", help="Flag to add prior preservation loss.",)
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument("--num_class_images", type=int, default=100, help=( "Minimal class images for prior preservation loss. If not have enough images, additional images will be sampled with class_prompt."),)
    
    parser.add_argument( "--output_dir", type=str, default="text-inversion-model", help="The output directory where the model predictions and checkpoints will be written.",)
    
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument( "--resolution",  type=int, default=512, help=("The resolution for input images, all the images in the train/validation dataset will be resized to this resolution"),)
    parser.add_argument("--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution")
    
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument("--train_vae", action="store_true", help="Whether to train the text encoder")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images.")
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--max_train_steps",  type=int, default=None, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",)
    
    parser.add_argument("--gradient_accumulation_steps",  type=int,  default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",)
    
    parser.add_argument("--learning_rate",type=float,default=5e-6,help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--scale_lr",action="store_true",default=False,help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",)
    parser.add_argument("--lr_scheduler",type=str,default="constant", help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'' "constant", "constant_with_warmup"]'),)
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--hub_model_id", type=str,default=None,help="The name of the repository to keep in sync with the local `output_dir`.",)
    
    parser.add_argument("--logging_dir", type=str,default="logs",help=("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),)
    
    parser.add_argument("--mixed_precision",type=str,default="no",choices=["no", "fp16", "bf16"], help=("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU."),)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")


    # additional 

    parser.add_argument("--with_clip_loss", default=False, action="store_true", help="Flag to add prior preservation loss.",)
    parser.add_argument("--with_pose_preservation", default=False, action="store_true", help="Flag to add prior preservation loss.",)
    parser.add_argument("--pose_loss_weight", type=float, default=10.0, help="The weight of prior preservation loss.")
    parser.add_argument("--outlier_noise", default=False, action="store_true", help="The weight of prior preservation loss.")
    
    parser.add_argument("--num_denoising_steps",type=int, default=1, help="")
    parser.add_argument("--vary_noise_std", default=False, action="store_true", help="")


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.instance_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        tokenizer,
        instance_data_root,
        instance_prompt,
        multi_instance_prompt_file,
        class_prompt=None,
        size=256,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        # chair_roots = ["complex/chair36.txt/Clean", "simple/chair23.txt/Clean", "ordinary/Chair2.txt/Clean"]
        chair_roots=[]
        for i in range(0, 90):
            chair_roots.append("train_"+str(i)+"/Clean")
        self.instance_images_path = []

        for i in range(len(chair_roots)):
            self.instance_data_root = Path(os.path.join(instance_data_root, chair_roots[i]))
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")
            inst_list = list(Path(self.instance_data_root).iterdir())[:-2]
            self.instance_images_path = [*self.instance_images_path, *inst_list]

        random.shuffle(self.instance_images_path)


        # self.instance_images_path = self.instance_images_path[:1]
        # print(self.instance_images_path)

        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        self.class_prompt = "A mesh"
        self.mid_prompt = "@clean the mesh"
        
        self.instance_1 = "A photo of a @clean mesh"
        self.instance_2 = "A photo of a @noisy mesh"

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        filename = self.instance_images_path[index % self.num_instance_images]
        
        if "train_6/" in str(filename) or "train_10/" in str(filename) or "train_23/" in str(filename) or "train_69/" in str(filename) :
            example["test"] = [0]
        else:
            example["test"] = [1]

        # load images
        instance_image = Image.open(filename)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images_clean"] = self.image_transforms(instance_image)
        
        instance_image = Image.open(str(filename).replace("CLN", "NOS").replace("Clean", "Noise"))
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images_noise"] = self.image_transforms(instance_image)

        # load prompts
        example["instance_prompt_clean"] = self.tokenizer(
        self.instance_1 ,padding="do_not_pad",
        truncation=True,max_length=self.tokenizer.model_max_length,).input_ids

        example["instance_prompt_noise"] = self.tokenizer(
            self.instance_2 ,padding="do_not_pad",
            truncation=True,max_length=self.tokenizer.model_max_length,).input_ids

        example["class_images"] = example["instance_images_clean"]

        example["class_prompt"] = self.tokenizer(
        self.class_prompt, padding="do_not_pad",
        truncation=True,max_length=self.tokenizer.model_max_length,).input_ids

        example["mid_prompt"] = self.tokenizer(
        self.mid_prompt, padding="do_not_pad",
        truncation=True,max_length=self.tokenizer.model_max_length,).input_ids


        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"



def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    from torch.utils.tensorboard import SummaryWriter
    tb_logs = SummaryWriter(Path(args.output_dir, args.logging_dir))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
        kwargs_handlers=[InitProcessGroupKwargs(timeout = datetime.timedelta(seconds=600000))],
    )


    CLIP_embed = get_embed_fn(model_type="clip_vit",  device=accelerator.device, num_layers=-1,  clip_cache_root=os.path.expanduser("~/.cache/clip")) 

    device = accelerator.device

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
        )
        
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, "tokenizer"),
            revision=args.revision,
        )

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, "text_encoder"),
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, "vae"),
        revision=args.revision,
    ).float().to(device)

    unet = UNet2DConditionModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, "unet"),
        revision=args.revision,
    ).float().to(device)

    unet_uncond = UNet2DModel(
        os.path.join(args.pretrained_model_name_or_path, "unet"),
        in_channels=4,
        out_channels=4,
    ).float().to(device)


    # def init_all(model):
    #     for module in model.modules() :
    #         if not isinstance(module, torch.nn.Sequential):
    #             pdb.set_trace()
    #             init_all(module)

    
    #torch.nn.init.xavier_uniform(module) 
    #init_all(text_encoder) 
    # init_all(vae) 
    # init_all(unet) 
    # init_all(unet_uncond) 

    if not args.train_vae:
        vae.requires_grad_(False)

    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        unet_uncond.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
        if args.train_vae:
            train_vae.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = ( args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = [unet.parameters(), unet_uncond.parameters()]
    if args.train_text_encoder :
        params_to_optimize = itertools.chain(unet.parameters(), unet_uncond.parameters(), text_encoder.parameters()) 
    
    if args.train_text_encoder :
        params_to_optimize = itertools.chain(unet.parameters(), unet_uncond.parameters(), text_encoder.parameters()) 
    
    else :
        params_to_optimize = itertools.chain(unet.parameters(), unet_uncond.parameters())


    optimizer = optimizer_class(params_to_optimize,lr=args.learning_rate,betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,eps=args.adam_epsilon,)

    noise_scheduler = DDPMScheduler.from_config(os.path.join(args.pretrained_model_name_or_path, "scheduler"),)

    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        multi_instance_prompt_file=args.multi_instance_prompt_file,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    def collate_fn(examples):
        
        # load images and prompt for index and clean equivalent
        instance_values = [example["test"] for example in examples]
        instance_values += [example["mid_prompt"] for example in examples]

        pixel_values = [example["instance_images_noise"] for example in examples]
        pixel_values += [example["instance_images_clean"] for example in examples]


        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        instance_values = tokenizer.pad({"input_ids": instance_values}, padding="max_length",max_length=tokenizer.model_max_length, return_tensors="pt",).input_ids

        batch = { "instance_values": instance_values, "pixel_values": pixel_values,}
        
        return batch

    train_dataloader = torch.utils.data.DataLoader( train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=1)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(args.lr_scheduler,optimizer=optimizer,num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps, num_training_steps=args.max_train_steps * args.gradient_accumulation_steps, )

    if args.train_text_encoder:
        unet, unet_uncond, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, unet_uncond, text_encoder, optimizer, train_dataloader, lr_scheduler)
    else:
        unet,unet_uncond, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, unet_uncond, optimizer, train_dataloader, lr_scheduler)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    # vae.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    # random affine transformations
    rnd_transform = transforms.RandomAffine( degrees=(-180, 180), translate=((0.1, 0.1)), fill=1)
    lpips_fn = lpips.LPIPS(net='alex').to(accelerator.device)

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):

            # Unpack Inputs (+ affine transformation)
            image_noise, img_gt = torch.chunk(rnd_transform(batch["pixel_values"]), 2, dim=0)
            #image_noise, img_gt = torch.chunk((batch["pixel_values"]), 2, dim=0)
            test_or_train, prompt_mid = torch.chunk(batch["instance_values"], 2, dim=0)

            if test_or_train[0,0].item() ==  0:
                mode = "eval"
                unet.eval()
                unet_uncond.eval()                                        
                if args.train_text_encoder:
                    text_encoder.eval()
                if args.train_vae:
                    vae.eval()

                image_noise, img_gt = image_noise.detach(), img_gt.detach()
                test_or_train, prompt_mid = test_or_train.detach(), prompt_mid.detach()
     
            elif test_or_train[0,0].item() ==  1:
                mode = "train"
                unet.train()
                unet_uncond.train()
                if args.train_text_encoder:
                    text_encoder.train()
                if args.train_vae:
                    vae.train()

            with accelerator.accumulate(unet) and accelerator.accumulate(unet_uncond):

                # Embeddings
                latents_gt = vae.encode(img_gt).latent_dist.sample() * 0.18215
                latents_noise = vae.encode(image_noise).latent_dist.sample() * 0.18215
                # latents_noise += torch.randn_like(latents_noise)
                encoder_hidden_states = text_encoder(prompt_mid)[0]

                # Timesteps
                bsz = latents_gt.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents_gt.device)
                timesteps = timesteps.long()

                # Reverse Diffusion                
                noise_pred = unet(latents_noise, timesteps, encoder_hidden_states).sample
                latents_pred = latents_noise - noise_pred 

                # Decode Image
                img_pred = vae.decode(1 / 0.18215 *latents_pred[:1]).sample

                # 1. lpips loss
                #loss_lpips = 0
                loss_lpips = lpips_fn(img_pred, img_gt)
                tb_logs.add_scalar(mode+"/loss_lpips", loss_lpips, global_step=step+len(train_dataloader)*epoch)
                
                # 2. image loss
                img_pred = (img_pred / 2 + 0.5).clamp(0,1)
                img_gt = (img_gt / 2 + 0.5).clamp(0,1)
                loss_img = 0
                if args.with_pose_preservation :
                    loss_img = F.mse_loss(img_pred, img_gt, reduction="mean")
                    tb_logs.add_scalar(mode+"/loss_img", loss_img, global_step=step+len(train_dataloader)*epoch)
                    
                # 3. clip loss
                loss_clip = 0
                if args.with_clip_loss :
                    loss_clip = -F.cosine_similarity(CLIP_embed(resize_img_for_clip(img_pred.permute(3,2,1,0).squeeze(3)))[0,0], CLIP_embed(resize_img_for_clip(img_gt.permute(3,2,1,0).squeeze(3)))[0,0], dim=0)
                    tb_logs.add_scalar(mode+"/loss_clip", loss_clip, global_step=step+len(train_dataloader)*epoch)

                # 4. custom noise loss
                noise = latents_noise - latents_gt
                loss_noise = F.mse_loss(noise, noise_pred, reduction="none").mean([1, 2, 3]).mean()
                tb_logs.add_scalar(mode+"/loss_noise", loss_noise, global_step=step+len(train_dataloader)*epoch)

                # 6. aggregate loss
                loss = args.pose_loss_weight*loss_img + loss_noise + loss_lpips + loss_clip
                tb_logs.add_scalar(mode+"/loss", loss, global_step=step+len(train_dataloader)*epoch)

                # Logging
                if step % 100 == 0 or mode == "eval":
                    orig_image = torch.hstack([img_gt.detach().cpu()[0], (image_noise/ 2 + 0.5).detach().cpu()[0]])
                    tb_logs.add_image(os.path.join(mode, str(step)+"orig"), orig_image.numpy(), global_step=epoch, dataformats='CHW')
                    tb_logs.add_image(os.path.join(mode, str(step)+"pred"), img_pred.detach().cpu()[0].numpy(), global_step=epoch, dataformats='CHW')

                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = ( itertools.chain(unet.parameters(), unet_uncond.parameters(), text_encoder.parameters()) if args.train_text_encoder else itertools.chain(unet.parameters(), unet_uncond.parameters()))
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
            # torch.cuda.empty_cache()
            
        accelerator.wait_for_everyone()
    

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionImg2ImgDenoisingPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=accelerator.unwrap_model(vae),
            unet=accelerator.unwrap_model(unet),
            unet_uncond= accelerator.unwrap_model(unet_uncond),
            num_denoising_steps = args.num_denoising_steps,
            text_encoder=accelerator.unwrap_model(text_encoder),
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    torch.autograd.set_detect_anomaly(True)
    main(args)