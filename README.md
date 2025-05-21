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
