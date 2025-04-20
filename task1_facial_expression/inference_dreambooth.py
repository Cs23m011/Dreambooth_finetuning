import os
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch

# Set environment variables (with defaults for portability)
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")
os.environ["HF_HOME"] = os.getenv("HF_HOME", "./cache/huggingface")
os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_CACHE", os.environ["HF_HOME"])
os.environ["WANDB_DIR"] = os.getenv("WANDB_DIR", "./cache/wandb")
os.environ["WANDB_CACHE_DIR"] = os.getenv("WANDB_CACHE_DIR", os.environ["WANDB_DIR"])
os.environ["TORCH_HOME"] = os.getenv("TORCH_HOME", "./cache/torch")
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "./cache/transformers")

# Load fine-tuned components
UNET_PATH = os.getenv("UNET_PATH", "./models/dreambooth_output/unet")
TEXT_ENCODER_PATH = os.getenv("TEXT_ENCODER_PATH", "./models/dreambooth_output/text_encoder")
BASE_MODEL = os.getenv("BASE_MODEL", "sd-legacy/stable-diffusion-v1-5")

# Load components
unet = UNet2DConditionModel.from_pretrained(UNET_PATH)
text_encoder = CLIPTextModel.from_pretrained(TEXT_ENCODER_PATH)

# Load pipeline with fine-tuned unet and text encoder
pipeline = DiffusionPipeline.from_pretrained(
    BASE_MODEL,
    unet=unet,
    text_encoder=text_encoder,
    torch_dtype=torch.float16,
).to("cuda")

# Prompt and generation
prompt = os.getenv("PROMPT", "A realistic photo of sks person in stadium")
output_path = os.getenv("OUTPUT_IMAGE", "output.png")

image = pipeline(prompt, num_inference_steps=50, guidance_scale=7).images[0]
image.save(output_path)

print(f"Saved generated image to {output_path}")
