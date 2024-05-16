export MODEL_NAME="/data/storage/bpietro/huggingface/diffusers/models--stabilityai--stable-diffusion-2-1/snapshots/f7f33030acc57428be85fbec092c37a78231d75a"
#export MODEL_NAME="/data/storage/bpietro/huggingface/diffusers/model-stable-diffusion-2-1-denoising-01"
export OUTPUT_DIR="/data/storage/bpietro/huggingface/diffusers/model-stable-diffusion-2-1-denoising-ours"

export CLASS_DIR="/data/storage/bpietro/datasets/sd_data/"
export INSTANCE_DIR="/data/storage/bpietro/datasets/sd_data/"


accelerate launch --multi_gpu --num_cpu_threads_per_process 48 train_our_model.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_DIR \
  --instance_prompt=" A <noisy> mesh." \
  --class_prompt="A <clean> mesh" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --multi_instance_prompt_file="./multi_prompts.json" \
  --num_denoising_steps=1 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-5 \
  --train_text_encoder \
  --train_vae \
  --lr_scheduler="linear" \
  --lr_warmup_steps=0 \
  --resolution=512 \
  --max_train_steps 3000 \
  --with_pose_preservation --pose_loss_weight=10.0 \