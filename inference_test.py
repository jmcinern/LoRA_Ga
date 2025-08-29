# inference.py
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---- Config (edit if needed) ----
BASE_ID = "Qwen/Qwen3-1.7B-Base"
OUTPUT_DIR = "qwen3-1p7b-lora-ga"   # must match your training output_dir
TEST_PROMPT = "What is the capital city of America?"
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.8
TopK = 20
Min_P = 0.01
# ---------------------------------

def _latest_adapter_dir(root: str) -> str:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Output dir not found: {root}")
    cps = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p) and name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[-1])
                cps.append((step, p))
            except ValueError:
                pass
    if cps:
        cps.sort(key=lambda x: x[0])
        latest = cps[-1][1]
        print(f"[INFO] Using latest checkpoint: {latest}")
        return latest
    print(f"[INFO] No checkpoints found; using root adapters: {root}")
    return root

def main():
    adapters_path = _latest_adapter_dir(OUTPUT_DIR)

    print("[INFO] Loading tokenizer and base model…")
    tok = AutoTokenizer.from_pretrained(adapters_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )


    print("[INFO] Attaching LoRA adapters…")
    model = PeftModel.from_pretrained(base, adapters_path)
    model.eval()

    print(f"[INFO] Special tokens -> EOS: {tok.eos_token!r}  PAD: {tok.pad_token!r}")
    messages = [{"role": "user", "content": TEST_PROMPT}]

    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    
    prompt = re.sub(r"<think>.*?</think>\s*", "", prompt, flags=re.DOTALL)
    print("[DEBUG] Prompt preview:", prompt[:300].replace("\n", "\\n"))

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TopK,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )

    gen_ids = outputs[0][inputs.input_ids.shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=False)


    print("\n=== Model reply ===")
    print(text.strip())

if __name__ == "__main__":
    main()
