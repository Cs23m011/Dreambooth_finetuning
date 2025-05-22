#!/usr/bin/env bash
#set -e

# ─── GPU & CACHES ─────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="1"
export HF_HOME="/projects/data/mtechteam/amar/data/huggingface"
export HF_DATASETS_CACHE="/projects/data/mtechteam/amar/data/huggingface"
export WANDB_DIR="/projects/data/mtechteam/amar/data/wandb"
export WANDB_CACHE_DIR="/projects/data/mtechteam/amar/data/wandb"
export TORCH_HOME="/projects/data/mtechteam/amar/data/torch"
export TRANSFORMERS_CACHE="/projects/data/mtechteam/amar/data/transformers"

# ─── PATHS & TOKENS ──────────────────────────────────────────────────────────
SCRIPT="train_dreambooth_sd3.py"
BASE_MODEL="stabilityai/stable-diffusion-3.5-medium"
INSTANCE_DIR="/projects/data/mtechteam/amar/data/datasets/rajni_image"
INSTANCE_PROMPT="sks rajinikanth"
OUTPUT_DIR="sd3-dreambooth-output3"
CLASS_DIR='/projects/data/mtechteam/amar/data/datasets/class_images'  
CLASS_TOKEN="a photo of person"

mkdir -p "$OUTPUT_DIR"

# ─── LAUNCH (NO per-image captions! Only instance_prompt is used) ──────────────
accelerate launch "$SCRIPT" \
  --pretrained_model_name_or_path "$BASE_MODEL" \
  --instance_data_dir "$INSTANCE_DIR" \
  --instance_prompt "$INSTANCE_PROMPT" \
  --gradient_checkpointing \
  --train_text_encoder \
  --resolution 1024 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-6 \
  --text_encoder_lr 1e-7 \
  --max_train_steps 1000 \
  --use_8bit_adam \
  --checkpointing_steps 900 \
  --mixed_precision bf16 \
  --output_dir "$OUTPUT_DIR"
