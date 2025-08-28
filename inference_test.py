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
tok.pad_token = tok.eos_token or "<|endoftext|>"
tok.padding_side = "right"
IM_END_ID = tok.convert_tokens_to_ids("<|im_end|>")  # EOS we want

def encode_chat(prompt: str):
    messages = [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(messages, add_generation_prompt=True,
                                   tokenize=True, return_tensors="pt")

# Sampling setup (prevents loops) + stop at <|im_end|>
GEN_KW = dict(
    max_new_tokens=512,
    do_sample=True,            # enables temperature/top_p
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.15,
    no_repeat_ngram_size=4,
    eos_token_id=IM_END_ID,    # ① stop on <|im_end|>
    pad_token_id=tok.eos_token_id,
)

def generate(model, prompt: str):
    model.eval()
    model.config.use_cache = True
    # also set on config to be safe
    model.generation_config.eos_token_id = IM_END_ID
    ids = encode_chat(prompt).to(model.device)
    with torch.no_grad():
        out = model.generate(ids, **GEN_KW)
    text = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=False)
    return text  # hide the marker


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

