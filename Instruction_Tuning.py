# load model from https://huggingface.co/jmcinern/qwen3-8b-base-cpt
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback, TrainerControl
import torch
model_name = "jmcinern/qwen3-8b-base-cpt"
cache_path = "./cache"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_path,
    trust_remote_code=True, 
    torch_dtype=torch.float16,
    subfolder="checkpoint-33000"
)