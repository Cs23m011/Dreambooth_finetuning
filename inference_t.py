#!/usr/bin/env python
# simple_inference.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/projects/data/mtechteam/amar/data/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/projects/data/mtechteam/amar/data/huggingface"
os.environ["WANDB_DIR"] = "/projects/data/mtechteam/amar/data/wandb"
os.environ["WANDB_CACHE_DIR"] = "/projects/data/mtechteam/amar/data/wandb"
os.environ["TORCH_HOME"] = "/projects/data/mtechteam/amar/data/torch"
os.environ["TRANSFORMERS_CACHE"] = "/projects/data/mtechteam/amar/data/transformers"
import torch
from diffusers import StableDiffusion3Pipeline

# Path to your fine-tuned model
MODEL_PATH = "/projects/data/mtechteam/amar/diffusers/examples/dreambooth/sd3-dreambooth-output3"

# Prompt for inference (must include your instance token)
PROMPT = "sks rajinikanth wearing a white dress and laying down in a beach on a wooden chair which is made of wood and visible in image."

# Inference settings
NUM_INFERENCE_STEPS = 60
GUIDANCE_SCALE = 8.5
OUTPUT_PATH = "/projects/data/mtechteam/amar/data/dog-bucket1.png"

def main():
    # Load the fine-tuned pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16
    ).to("cuda")

    # Enable memory-efficient attention if available
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass

    # Generate image
    image = pipe(
        PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE
    ).images[0]

    # Save output
    image.save(OUTPUT_PATH)
    print(f"Saved image to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
