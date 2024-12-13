
import torch
from huggingface_hub import snapshot_download

model_name = "TechxGenus/starcoder2-7b-instruct" # or "codellama/CodeLlama-7b-Instruct-hf" "codellama/CodeLlama-13b-Instruct-hf" "deepseek-ai/deepseek-coder-7b-instruct-v1.5" "meta-llama/Meta-Llama-3.1-8B-Instruct"
snapshot_download(repo_id=model_name,local_dir = "starcoder2-7b")

