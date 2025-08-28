import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---- Reload model + LoRA ----
base_model_id = "Qwen/Qwen3-0.6B-base"
lora_path = "qwen3-8b-ga-en-lora"

tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dtype = (torch.bfloat16 if torch.cuda.is_available()
         and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=dtype,
    trust_remote_code=True
)

# attach LoRA weights
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

# ---- Inference example ----
prompt = "Translate this to Irish: 'Hello, how are you?'"

inputs = tokenizer.apply_chat_template(
    [{"role":"user","content":prompt}],
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    output = model.generate(
        inputs,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
