from dependencies.CLIP import clip_utils
import timm
import torch
import pdb
import torch.nn as nn
import torchvision
from torch.utils.checkpoint import checkpoint as run_checkpoint

from torchvision import transforms 

def get_embed_fn(model_type, device, num_layers=-1, spatial=False, checkpoint=False, clip_cache_root=None):

    if model_type.startswith('clip_vit'):
        assert clip_cache_root
        if model_type == 'clip_vit':
            clip_utils.load_vit(root=clip_cache_root,device=device)

        elif model_type == 'clip_vit_b16':
            clip_utils.load_vit('ViT-B/16', root=clip_cache_root, device=device)
        embed = lambda ims: clip_utils.clip_model_vit(images_or_text=ims.float(), num_layers=num_layers)  # [N,L=50,D]
        assert not clip_utils.clip_model_vit.training
    
    elif model_type == 'clip_rn50':
        assert clip_cache_root
        clip_utils.load_rn(jit=False, root=clip_cache_root)
    
        embed = lambda ims: clip_utils.clip_model_rn(images_or_text=ims.float(), num_layers=num_layers)
        assert not clip_utils.clip_model_rn.training
        
    else:    
        raise NotImplementedError

    if checkpoint:
        return lambda x: run_checkpoint(embed, x)

    return embed


def resize_img_for_clip(image):
    image = torch.clamp(image, min=0, max=1)
    image = image.permute(2, 0, 1) # C, H, W
    transform = transforms.Resize((224, 224))
    out = transform(image).unsqueeze(0)
    out_img = clip_utils.CLIP_NORMALIZE(out)
    return out_img
