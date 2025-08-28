import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

CKPT = "qwen3-8b-lora-bilingual/checkpoint-5"
peft_cfg = PeftConfig.from_pretrained(CKPT)
BASE = peft_cfg.base_model_name_or_path

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype, trust_remote_code=True).to(device)
model.resize_token_embeddings(len(tok), mean_resizing=False)
model = PeftModel.from_pretrained(model, CKPT).to(device).eval()

# block chain-of-thought / tool tokens
ban_tokens = ["<think>","</think>","<tool_call>","</tool_call>","<tool_response>","</tool_response>"]
bad_ids = [[i] for i in tok.convert_tokens_to_ids(ban_tokens) if i is not None and i != tok.unk_token_id]

# also stop on end-of-turn
im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
eos_ids = [t for t in [tok.eos_token_id, im_end_id] if t is not None]

def chat(prompt, sample=False, max_new=128):
    msgs = [{"role":"system","content":""},{"role":"user","content":prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=sample,
        eos_token_id=eos_ids,
        pad_token_id=tok.pad_token_id,
        bad_words_ids=bad_ids,
    )
    print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())

chat("Cad e priomhchathair na hEirinn")          # deterministic
chat("what is the capital city of Ireland")      # deterministic
chat("scriobh gearrceal dom", sample=True, max_new=200)  # sampled
