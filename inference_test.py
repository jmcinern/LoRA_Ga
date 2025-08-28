# Inference: Base vs Base+LoRA (and merged)
# pip install -U "transformers>=4.53" "trl>=0.9.7" peft accelerate

import os, glob, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_id = "Qwen/Qwen3-0.6B-base"
output_dir = "qwen3-8b-lora-bilingual"   # same as your SFTConfig.output_dir
# If you saved checkpoints during training, pick the latest; else it uses output_dir
ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")),
               key=lambda p: int(p.split("-")[-1]))
lora_dir = ckpts[-1] if ckpts else output_dir

# --- Tokenizer ---
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tok.eos_token = "<|im_end|>"  # chat template for LoRA training marks turn taking with <|im_end|> # default is "<|endoftext|>"
tok.pad_token = tok.eos_token
tok.padding_side = "right"

# --- Helpers ---
def encode_chat(prompt: str):
    messages = [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    )

GEN_KW = dict(max_new_tokens=1000, do_sample=False, temperature=0.6)

def generate(model, prompt: str):
    model.eval()
    model.config.use_cache = True
    ids = encode_chat(prompt).to(model.device)
    with torch.no_grad():
        out = model.generate(ids, **GEN_KW, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=False).strip()

prompts = [
    "Cad í príomhchathair na hÉireann?",
    "Explain what a neural network is.",
]

# Pick dtype/device
dtype = (torch.bfloat16 if torch.cuda.is_available()
         and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16)
device_map = "auto" if torch.cuda.is_available() else None
torch.manual_seed(0)

# --- 1) BASE ---
base = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=dtype, device_map=device_map
)
print("\n=== BASE MODEL ===")
for p in prompts:
    print(f"\n[Prompt] {p}\n[Base]   {generate(base, p)}")

# --- 2) BASE + LoRA (PEFT adapter) ---
base_lora = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=dtype, device_map=device_map
)
base_lora = PeftModel.from_pretrained(base_lora, lora_dir)
print("\n=== BASE + LoRA (adapter) ===")
for p in prompts:
    print(f"\n[Prompt] {p}\n[LoRA]   {generate(base_lora, p)}")

