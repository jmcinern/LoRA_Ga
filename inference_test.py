# pip install -U "transformers>=4.53" "trl>=0.9.7" peft accelerate
import os, json, glob, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- Choose a BASE to test ---
# 1) Known instruct checkpoint (no LoRA) to validate your script:
#    e.g. replace with the exact instruct variant you have locally.
model_id = "Qwen/Qwen3-1.7B"   # <- use a real instruct ckpt you have
use_lora = True

# 2) Or load your LoRA adapter (must match its base):
output_dir = "qwen3-8b-lora-bilingual"
ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")),
               key=lambda p: int(p.split("-")[-1]))
lora_dir = ckpts[-1] if ckpts else output_dir
if use_lora:
    with open(os.path.join(lora_dir, "adapter_config.json")) as f:
        base_in_adapter = json.load(f).get("base_model_name_or_path")
    assert base_in_adapter, "adapter_config.json missing base_model_name_or_path"
    model_id = base_in_adapter

# --- Tokenizer (load from BASE, not adapter) ---
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token or "<|endoftext|>"

IM_END_ID = tok.convert_tokens_to_ids("<|im_end|>")
EOT_ID    = tok.convert_tokens_to_ids("<|endoftext|>")
print("specials:", tok.all_special_tokens)
print("IM_END_ID:", IM_END_ID, "EOT_ID:", EOT_ID)

def encode_chat(prompt: str):
    msgs = [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                   tokenize=True, return_tensors="pt")

GEN_KW = dict(
    max_new_tokens=256,
    do_sample=True,                 # start GREEDY to test stopping
    eos_token_id=[IM_END_ID, EOT_ID],
    pad_token_id=tok.pad_token_id,
    return_dict_in_generate=True,
    temperature=0.6,
    top_p=0.95,
)

def generate(model, prompt: str):
    model.eval()
    ids = encode_chat(prompt).to(model.device)
    with torch.no_grad():
        out = model.generate(ids, **GEN_KW)
    seq = out.sequences[0]
    gen = seq[ids.shape[-1]:]
    tail = tok.convert_ids_to_tokens(gen[-32:].tolist())
    print("stopped_early:", getattr(out, "stopped_early", None))
    print("finish_reason:", getattr(out, "sequences_scores", None))  # HF lacks finish_reason; inspect tokens instead
    print("tail tokens:", tail)
    print("last_id==IM_END:", gen.numel()>0 and gen[-1].item()==IM_END_ID)
    text = tok.decode(gen, skip_special_tokens=False)
    return text

# --- Load model(s) ---
dtype = (torch.bfloat16 if torch.cuda.is_available()
         and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16)
device_map = "auto" if torch.cuda.is_available() else None
base = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True,
                                            torch_dtype=dtype, device_map=device_map)

if use_lora:
    base = PeftModel.from_pretrained(base, lora_dir)

# --- Test prompts ---
prompts = ["Answer in one word. What is the capital of Ireland?"]

for i in range(10):
    for p in prompts:
        print("\n[Prompt]", p)
        out = generate(base, p)
        print("[Output]\n", out)

