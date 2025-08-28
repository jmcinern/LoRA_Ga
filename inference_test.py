# inference_test.py
import torch, json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

CKPT = "qwen3-8b-lora-bilingual/checkpoint-5"  # your folder with adapter_config.json etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = (torch.bfloat16 if torch.cuda.is_available()
         and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16)

# 1) Which base did the adapter expect?
peft_cfg = PeftConfig.from_pretrained(CKPT)
BASE = peft_cfg.base_model_name_or_path  # e.g., "Qwen/Qwen3-0.6B"

# 2) Load tokenizer FROM CHECKPOINT (exact vocab + template)
tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "right"

# 3) Load base
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=DTYPE, trust_remote_code=True).to(DEVICE)
model.config.use_cache = False
model.config.pad_token_id = tok.eos_token_id

# 4) Resize base embeddings to tokenizer size (fast init)
model.resize_token_embeddings(len(tok), mean_resizing=False)

# 5) Load adapters
model = PeftModel.from_pretrained(model, CKPT).to(DEVICE).eval()

# Sanity: confirm sizes now match
print("tok size:", len(tok), "| emb size:", model.get_input_embeddings().num_embeddings)

def chat(prompt, max_new_tokens=128, do_sample=False):
    msgs = [{"role":"system","content":""},{"role":"user","content":prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
            eos_token_id=model.config.eos_token_id, pad_token_id=tok.pad_token_id,
        )
    print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())

if __name__ == "__main__":
    chat("Cad e priomhchathair na hEirinn")
    chat("what is the capital city of Ireland")
    chat("scriobh gearrceal dom", max_new_tokens=200, do_sample=True)
