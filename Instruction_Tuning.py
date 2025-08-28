# pip install -U "transformers>=4.53" "trl>=0.9.7" peft accelerate datasets

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ---------------- Data: load & convert to chat messages ----------------
ds = load_dataset("jmcinern/Instruction_Ga_En_for_LoRA")  # expects train/test

def to_messages(ex):
    user = ex["instruction"] + (("\n\n" + ex["context"]) if ex.get("context") else "")
    return {"messages": [
        {"role":"system","content":""},
        {"role":"user","content": user},
        {"role":"assistant","content": ex["response"]},
    ]}

cols = ds["train"].column_names
ds = ds.map(to_messages, remove_columns=[c for c in cols if c != "messages"])

# ---------------- Model & tokenizer (unchanged tokenizer) ---------------
model_id = "jmcinern/qwen3-8b-base-cpt"
subfolder = "checkpoint-33000"

tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=subfolder, trust_remote_code=True)
# runtime-only pad setting (no vocab change, not saved)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dtype = (torch.bfloat16 if torch.cuda.is_available()
         and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    subfolder=subfolder,
    torch_dtype=dtype,
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.eos_token_id

# ---------------- LoRA (no quantization) --------------------------------
lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.1, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
)
model = get_peft_model(model, lora)

# ---------------- Training (new API: assistant_only_loss) ----------------
cfg = SFTConfig(
    output_dir="qwen3-8b-lora-bilingual",
    max_seq_length=4096,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    logging_steps=20,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=1000,
    bf16=(dtype == torch.bfloat16),
    fp16=(dtype == torch.float16),
    packing=True,                  # works with assistant_only_loss
    optim="adamw_torch_fused",
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
    assistant_only_loss=True,      # <-- replaces old collator path
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=cfg,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    # No formatting_func needed: TRL detects `messages` and uses tokenizer.chat_template
)

trainer.train()
model.save_pretrained("qwen3-8b-ga-en-lora")   # save adapters only
