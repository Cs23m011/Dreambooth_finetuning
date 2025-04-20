#!/bin/bash

# Set environment variables with default fallbacks
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME="${HF_HOME:-./cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="${TORCH_HOME:-./cache/torch}"
export WANDB_DIR="${WANDB_DIR:-./cache/wandb}"
export WANDB_CACHE_DIR="$WANDB_DIR"

# Define model and dataset paths
export MODEL_NAME="${MODEL_NAME:-sd-legacy/stable-diffusion-v1-5}"
export INSTANCE_DIR="${INSTANCE_DIR:-./data/instance_images}"
export OUTPUT_DIR="${OUTPUT_DIR:-./outputs/dreambooth}"
export CLASS_DIR="${CLASS_DIR:-./data/class_images_person}"

# Run DreamBooth training
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="A photo of sks man" \
  --class_prompt="A photo of a man" \
  --num_class_images=200 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --snr_gamma=5.0 \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --train_text_encoder \
  --max_train_steps=700
