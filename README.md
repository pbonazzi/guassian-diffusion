#  Few-shot point cloud reconstruction and denoising via learned Guassian splats renderings and fine-tuned diffusion features  👀

### [💻 Blog](https://pietrobonazzi.com/projects/splatting) |[📜 Paper](https://arxiv.org/pdf/2404.01112v3) 


## Table of Content 

- [Setup](#setup)
- [Demos](#demos)
  - [Train](#train)
  - [Evaluate](#evaluate)
- [Citation and Acknowledgement](#citation-and-acknowledgement)


## Setup

1. On clusters

If running on a computer-node you can skip this, if you are on a server login (i.e. slurm) , follow these step to initialize conda and cuda . 
```bash
module load generic
module load anaconda3
conda init
module load vesta
module load cuda/10.2

# in your .bashrc
export PATH="/sapps/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/cuda-10.2.89-ooyc5cf36frbhli3zkdup27oh3d5xdej/bin:$PATH"
export LD_LIBRARY_PATH="/sapps/opt/spack/linux-ubuntu16.04-x86_64/gcc-5.4.0/cuda-10.2.89-ooyc5cf36frbhli3zkdup27oh3d5xdej/lib64:$LD_LIBRARY_PATH"

```

2. Dependencies

```bash
# clone repository
git clone --recursive https://github.com/uzh-rpg/master_thesis_pietro_bonazzi.git
cd master_thesis_pietro_bonazzi

# create a conda virtual environment
conda create --yes -n master_thesis_pietro_bonazzi python=3.8
conda activate ~/data/.conda/envs/master_thesis_pietro_bonazzi/

## install dependencies
pip3 install torch==1.10.0+cu102 torchvision==0.11.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
pip3 install pytorch3d
pip3 install -r requirements.txt
pip3 install "git+https://github.com/mmolero/pypoisson.git"

# compile libraries
cd dependencies/prefix_sum
pip install .
cd ../FRNN
pip install .
cd ../torch-batch-svd
pip install .
cd ../../CLIP
pip install -e .
cd ../../
pip install -e .

```


## Downloads

## Data

The only required flag for the Differentiable Rendererer (DR) is the `--input` flag , which takes the path to a geometry (format `obj`, `ply`, `txt`) or an image folder . 
You can download example data meshes here :
```
https://github.com/yifita/DSS/tree/master/example_data
```

The SCUT dataset here :
```
https://github.com/Gorilla-Lab-SCUT/SCUTSurface-code/
```

## Model

DR do not require pre-training, but the we open sourced a checkpoint of the Stable Diffusion Denoising Model, which can be downloaded from here :
```
https://huggingface.co/pebonazzi/stable-diffusion-2-1-denoising
```



## Demos

Flags let us control the training parameters , see the full list at the end of the script `scripts/options.py` to run experiment with different parameters. 
The only required flag is the `--input` flag , which takes the path to a geometry (format `obj`, `ply`, `txt`) or an image folder . 


### 3D Reconstruction
An example of 3D Reconstruction :
```
 python3 scripts/train.py --output_name reconstruction/  --overwrite \
--save_views --save_pointcloud --train_images 8 --train_batch_size 8 \
--init_icosphere --init_color grey  --dataset_name mesh --input_path /data/storage/bpietro/datasets/example_data/pointclouds/Kangaroo_V10k.ply --input_format ply \
--learn_points --lr_points 1e-2 --learn_normals --lr_normals 1e-3 --use_light --remove_and_repeat --with_restarts \
--max_epochs 600 --learn_sh --lr_sh 1e-2 --lambda_sem_cons_loss 0.01 --sem_cons_epoch_start 200 
```

### 3D Denoising

The denoising example first requires the creation of a denoised folder of images.
You can create the folder using the `point_cloud_denoising.py` script.

Next you can run the training script with the dataset_name `images` and the path to the denoised images, for examples:
```
python3 scripts/train.py --output_name denosing/ --max_epoch 128 --overwrite \
--train_images 8 --train_batch_size 8 \
--save_views --save_pointcloud --init_from_noisy \
--dataset_name images --input_path path/to/denoising_folder/ \
--learn_points --lr_points 8e-4 --with_restarts --use_light \
--learn_normals --lr_normals 8e-5  --lambda_normal_smoothness_loss 0.01 --lambda_sem_cons_loss 0.01
```


### Evaluate 

To results are automatically logged in the `train.log` file in the output folder. 
To save the results use the flag `--save_images` and `--save_pointcloud` during training.


## Citation and Acknowledgement

If our work is relevant to your project, thank you for citing our associated paper :

```
@article{thesis_pietro_23,
author = {Bonazzi, Pietro and Rakatosaona, Marie-Julie and Cannici, Marco},
title = {Point Cloud Reconstruction and Denoising via Learned-based Rendering Features},
year = {2023},
}
```

