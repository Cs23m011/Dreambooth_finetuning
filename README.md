# Dreambooth_finetuning
first create a conda env with python 3.10 .
```
conda create -n env python=3.10 -y

conda activate env
```
now install the requirement.txt
```
pip install requirement.txt
```
after this try the below two steps.
```
pip install git+https://github.com/huggingface/diffusers
pip install -U -r diffusers/examples/dreambooth/requirements.txt
```
now after this step you will get diffuser folder in your work space.
go to the dreambooth folder inside diffuser.
```
cd path/diffusers/examples/dreambooth
```
replace the  file named 'train_dreambooth_sd3.py' with the file above in the github repo. 

now copy the two file name below in the dreambooth folder which you will find in this repo.
```
train.sh 
inference.py
```
```
train.sh to finetune the diffusion model
inference.py to create face consistent images giving your prompt.
```
Below are the arguments you can use to finetune the dreambooth framework.
Below are the key training arguments used when fine-tuning a model using DreamBooth:

| Argument | Description |
|----------|-------------|
| `--pretrained_model_name_or_path` | Path to the base Stable Diffusion model (e.g., `CompVis/stable-diffusion-v1-4`). |
| `--instance_data_dir` | Directory containing instance images (your subject). |
| `--caption_file` | Path to a JSON or TXT file that maps image file names to custom prompts. This overrides `--instance_prompt` per image. |
| `--instance_prompt` | General prompt for the instance (used only if `caption_file` is not provided). |
| `--train_text_encoder` | Enables training of the text encoder for better prompt understanding. |
| `--gradient_checkpointing` | Reduces memory usage by checkpointing activations during backpropagation. |
| `--resolution` | Resolution to which all training images will be resized (e.g., `1024`). |
| `--train_batch_size` | Batch size for training (commonly `1` for DreamBooth). |
| `--gradient_accumulation_steps` | Accumulates gradients over multiple steps to simulate a larger batch size. |
| `--learning_rate` | Learning rate for the UNet. |
| `--text_encoder_lr` | Learning rate specifically for the text encoder. |
| `--max_train_steps` | Total number of training steps (e.g., `1000`). |
| `--use_8bit_adam` | Enables memory-efficient 8-bit Adam optimizer (requires `bitsandbytes`). |
| `--checkpointing_steps` | Frequency (in steps) at which checkpoints are saved. |
| `--mixed_precision` | Use `bf16` or `fp16` for reduced memory usage and faster training. |
| `--output_dir` | Directory to save the output model and checkpoints. |

here caption_file create a json file which contain image name as key and caption as values. please see the example of caption.json in this repo.
instance image I used the rajnikant image provived by you also the crop images of face only.
