# pip install -U "transformers>=4.53" "trl>=0.21.0" peft accelerate datasets

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ---------------------------
# Load dataset & filter
# ---------------------------
ds = load_dataset("jmcinern/Instruction_Ga_En_for_LoRA")
print("[INFO] Original splits:", list(ds.keys()))


# print ds first sample
for split in ds:
    print(f"[INFO] {split} size before filter:", len(ds[split]))

ds = ds.filter(lambda x: x.get("lang") == "ga")
for split in ds:
    print(f"[INFO] {split} size after lang=='ga' filter:", len(ds[split]))

# Peek at a raw sample
print("[DEBUG] Raw sample (pre-format):", ds["train"][0])

# ---------------------------
# Model / tokenizer
# ---------------------------
model_id = "Qwen/Qwen3-1.7B-Base"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print(f"[INFO] Special tokens -> EOS: {tokenizer.eos_token!r}  PAD: {tokenizer.pad_token!r}")
print(f"[INFO] Chat template present: {bool(tokenizer.chat_template)}")

# ---------------------------
# Map to messages -> text (non-thinking)
# ---------------------------
def to_messages(ex):
    user = ex["instruction"] + (("\n\n" + ex["context"]) if ex.get("context") else "")
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": ex["response"]},
    ]}

cols = ds["train"].column_names
ds = ds.map(to_messages, remove_columns=[c for c in cols if c != "messages"])
print("[DEBUG] Sample messages:", ds["train"][0]["messages"])

# Pre-render to plain text with thinking disabled
def to_text(ex):
    txt = tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, enable_thinking=False  # training: no generation prompt
    )
    return {"text": txt}

ds = ds.map(to_text)
print("[DEBUG] Templated text (first 500 chars):", ds["train"][0]["text"][:500].replace("\n", "\\n"))
print("[DEBUG] Contains '<think>'? ->", "<think>" in ds["train"][0]["text"])

# Sanity: token lengths
enc = tokenizer(ds["train"][0]["text"], return_tensors="pt", add_special_tokens=False)
print("[DEBUG] First example token length:", enc.input_ids.shape[1])

# Check average length (cheap sample)
sample_n = min(64, len(ds["train"]))
avg_len = sum(len(tokenizer(s["text"], add_special_tokens=False).input_ids) for s in ds["train"].select(range(sample_n))) / sample_n
print(f"[DEBUG] Avg token length over {sample_n} train samples:", int(avg_len))

# ---------------------------
# LoRA (adapters only)
# ---------------------------
peft_cfg = LoraConfig(
    task_type="CAUSAL_LM",
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()
model.config.use_cache = False  # good with grad checkpointing

# ---------------------------
# Trainer config
# ---------------------------
MAX_LEN = 4096
eval_split = "test" if "test" in ds else ("validation" if "validation" in ds else None)
eval_strategy = "steps" if eval_split else "no"
print("[INFO] Using eval split:", eval_split)
print("[INFO] Evaluation strategy:", eval_strategy)

sft_cfg = SFTConfig(
    output_dir="qwen3-1p7b-lora-ga",
    max_length=MAX_LEN,
    packing=True,                      # pack multiple samples up to max_length
    dataset_text_field="text",         # we pre-rendered text; do NOT pass messages here
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    max_steps=100,
    logging_steps=20,
    save_strategy="steps",
    save_steps=20,
    bf16=True,
    eval_strategy=eval_strategy,
    eval_steps=100,
    optim="adamw_torch",
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_cfg,
    train_dataset=ds["train"],
    eval_dataset=ds[eval_split] if eval_split else None,
)

print("[INFO] Trainer args summary:", trainer.args)
print("[INFO] Samples seen in train:", len(ds["train"]))

# ---------------------------
# Train / Save
# ---------------------------
trainer.train()

# Final save (PEFT adapters only)
trainer.save_model()                 # saves to args.output_dir
tokenizer.save_pretrained(sft_cfg.output_dir)

print("[INFO] Saved artifacts in:", sft_cfg.output_dir)
print("[INFO] Files include adapter_model.safetensors, adapter_config.json, and tokenizer files.")
