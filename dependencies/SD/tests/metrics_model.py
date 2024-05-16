import lpips 
import pdb
import imageio.v2 as imageio
from tqdm import tqdm
import torch
import os
from src.misc.imageFilters import Pix2PixDenoising, Pix2PixInitialize


def normalize_image(images):
    img_norm = (((images-images.min())/(images.max() - images.min()))*2 - 1)
    return img_norm

models = ["Pix2Pix", "Denoised", "WImgDenoised", "WClipDenoised", "OnlyTokenDenoised"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_loss = lpips.LPIPS(net='alex').to(device) # best forward scores
mse2psnr = lambda x : -10. * torch.log(x) \
            / torch.log(torch.tensor([10.], device=x.device))

pix2pix = Pix2PixInitialize()

for model in models :

    num_imgs = 0
    lpips, mse, psnr = 0,0,0

    for i in tqdm(range(8)):

        for j in range(10):
            gt_image = imageio.imread(os.path.join("/data/storage/bpietro/datasets/sketchfab_images/test_"+str(i), "Clean", "CLN_%06d.png" % j))
            
            if model == "Pix2Pix":
                noise_image = imageio.imread(os.path.join("/data/storage/bpietro/datasets/sketchfab_images/test_"+str(i), "Noise", "NOS_%06d.png" % j))
                image = Pix2PixDenoising(torch.tensor(noise_image/255).float().unsqueeze(0).to(device), pix2pix)[0]
                out_dir = os.path.join("/data/storage/bpietro/datasets/sketchfab_images/test_"+str(i), "Pix2Pix")
                os.makedirs(out_dir, exist_ok=True)
                imageio.imwrite(os.path.join(out_dir, "DEN_%06d.png" % j), (image.cpu().numpy()*255).astype("uint8"))
                image = image*255

            else:
                # Load
                image = imageio.imread(os.path.join("/data/storage/bpietro/datasets/sketchfab_images/test_"+str(i), model, "DEN_%06d.png" % j))
            
            # Metrics
            lpips += lpips_loss(normalize_image(torch.tensor(image).permute(2,1,0).to(device)), normalize_image(torch.tensor(gt_image).permute(2,1,0).to(device))).item()  
            mse_ = torch.nn.functional.mse_loss(torch.tensor(image).float().to(device)/255, torch.tensor(gt_image).float().to(device)/255)
            mse += mse_.item()    
            psnr += mse2psnr(mse_).item()     

            num_imgs+=1   


    print("---- {} ----- mse : {} , psnr : {} , lpips : {} ".format(str(model), mse/num_imgs, psnr/num_imgs, lpips/num_imgs))