export MODEL_NAME="/data/storage/bpietro/huggingface/diffusers/models--stabilityai--stable-diffusion-2-1/snapshots/f7f33030acc57428be85fbec092c37a78231d75a"
#export MODEL_NAME="/data/storage/bpietro/huggingface/diffusers/model-stable-diffusion-2-1-denoising-01"
export OUTPUT_DIR="/data/storage/bpietro/huggingface/diffusers/model-stable-diffusion-2-1-denoising-dreamboot"

export CLASS_DIR="/data/storage/bpietro/datasets/sd_data/"
export INSTANCE_DIR="/data/storage/bpietro/datasets/sd_data/"

accelerate launch train_dreamboot.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="A photo of a @clean mesh" \
  --class_prompt="A photo of a @noisy mesh" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800