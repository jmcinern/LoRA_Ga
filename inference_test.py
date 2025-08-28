import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---- Paths ----
base_model_id = "Qwen/Qwen3-0.6B-base"
lora_path = "qwen3-8b-ga-en-lora"

# ---- Tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dtype = (torch.bfloat16 if torch.cuda.is_available()
         and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16)

# ---- Base model ----
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=dtype,
    trust_remote_code=True
)
base_model.resize_token_embeddings(len(tokenizer))
base_model.eval()

# ---- LoRA model ----
lora_model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        trust_remote_code=True
    ),
    lora_path
)
lora_model.resize_token_embeddings(len(tokenizer))
lora_model.eval()

# ---- Helper: run inference ----
def chat(model, prompt, max_new_tokens=128):
    inputs = tokenizer.apply_chat_template(
        [{"role":"user","content":prompt}],
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ---- Compare outputs ----
prompt = "What is the capital of Ireland?"

print("\n--- Base model ---")
print(chat(base_model, prompt))

print("\n--- LoRA-tuned model ---")
print(chat(lora_model, prompt))
