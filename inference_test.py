# inference_test.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-0.6B"          # match the base you trained with
ADAPTER_DIR = "qwen3-8b-ga-en-lora"     # your saved adapters folder

dtype = (torch.bfloat16 if torch.cuda.is_available()
         and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenizer (unchanged)
tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

# Base model + LoRA adapters
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=dtype, trust_remote_code=True
).to(device)
model = PeftModel.from_pretrained(base, ADAPTER_DIR).to(device)
model.eval()

def chat(prompt, max_new_tokens=128, do_sample=False):
    # Neutral system; rely on the Qwen chat template
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    completion = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n>>> {prompt}\n{completion.strip()}\n")

if __name__ == "__main__":
    chat("Cad e priomhchathair na hEirinn")
    chat("what is the capital city of Ireland")
    chat("scriobh gearrceal dom", max_new_tokens=200, do_sample=True)  # creative: sampling on
