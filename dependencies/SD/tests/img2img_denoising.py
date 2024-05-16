"""
Pipeline to denoise point clouds from their renderings
Training Shapes : https://polybox.ethz.ch/index.php/s/sCC6pYAngLa9d8Y
Test Shapes : https://polybox.ethz.ch/index.php/s/wPxFlfxKbgVkfNb

Noise from a range between +- 1% of the bounding box diagonal of the shapes is added to training pointcloud.
PointClouds are then rendered using DSS rasterizer : "https://github.com/yifita/DSS"
"""

import os
from tqdm import tqdm
import numpy as np
import imageio.v2 as imageio
import torch
import pdb

# custom 
from dependencies.SD.pipe_denoising import StableDiffusionImg2ImgDenoisingPipeline
from src.misc.imageFilters import Pix2PixDenoising, Pix2PixInitialize


def normalize_image(images):
    img_norm = (((images-images.min())/(images.max() - images.min()))*2 - 1)
    return img_norm
    
# handle i/o
model_path = "/data/storage/bpietro/huggingface/diffusers/model-stable-diffusion-2-1-denoising-ours"

# initialize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = StableDiffusionImg2ImgDenoisingPipeline.from_pretrained(model_path, revision=None, torch_dtype=torch.float32,).to(device)
generator = torch.Generator(device=device)
generator.manual_seed(2147483647)
#pix2pix = Pix2PixInitialize()
for i in [0, 3, 10]:
    for j in range(0, 30):
        # Load
        init_image = imageio.imread(os.path.join("/data/storage/bpietro/datasets/sd_data/test_"+str(i), "Noise","NOS_%06d.png" % j))
        init_image_tens= torch.tensor(np.transpose(init_image, [2,0,1])/255).unsqueeze(0)

        # Denoise
        img_denoised = pipe(init_image=init_image_tens, generator=generator)    

        # Pix2Pix
        # image_pix2pix = Pix2PixDenoising(torch.tensor(init_image/255).float().unsqueeze(0).to(device), pix2pix)[0]

        # Save
        out_dir = os.path.join("/data/storage/bpietro/datasets/sd_data/test_"+str(i), "Ours")
        os.makedirs(out_dir, exist_ok=True)
        imageio.imwrite(os.path.join(out_dir, "DEN_%06d.png" % j), (img_denoised.cpu().permute(0, 2, 3, 1).float().numpy()[0]*255).astype("uint8"))
        # out_dir = os.path.join("/data/storage/bpietro/datasets/sketchfab_images/train_"+str(i), "Pix2PixTRI")
        # os.makedirs(out_dir, exist_ok=True)
        # imageio.imwrite(os.path.join(out_dir, "P2P_%06d.png" % j), (image_pix2pix.cpu().numpy()*255).astype("uint8"))
