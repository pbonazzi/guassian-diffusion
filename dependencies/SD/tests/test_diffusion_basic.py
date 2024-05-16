import requests
import torch
from PIL import Image
from io import BytesIO
import numpy as np
import imageio.v2 as imageio
import os
from diffusers import StableDiffusionImg2ImgPipeline

# load the pipeline
device = "cuda"
#model_id ="/data/storage/bpietro/huggingface/diffusers/models--stabilityai--stable-diffusion-2-1/snapshots/f7f33030acc57428be85fbec092c37a78231d75a"
model_path = "/data/storage/bpietro/huggingface/diffusers/model-stable-diffusion-2-1-denoising-ours"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16, feature_extractor=None, safety_checker=None)

pipe = pipe.to(device)
for i in [0, 3, 10]:
    for j in range(0, 30):
        # Load
        init_image = imageio.imread(os.path.join("/data/storage/bpietro/datasets/sd_data/test_"+str(i), "Noise","NOS_%06d.png" % j))
        init_image_tens= torch.tensor(np.transpose(init_image, [2,0,1])/255).unsqueeze(0)

        prompt = "@clean the mesh"

        for s in [0.3, 0.5, 0.7]:
            images = pipe(prompt=prompt, image=init_image_tens, strength=s, guidance_scale=7.5).images
            out_dir = os.path.join("/data/storage/bpietro/datasets/sd_data/test_"+str(i), "Dreamboot", str(s))
            os.makedirs(out_dir, exist_ok=True)
            images[0].save(os.path.join(out_dir, "TEST_%06d.png" % j))

