# pip install -U transformers>=4.53 datasets trl peft accelerate

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from trl.trainer.utils import DataCollatorForCompletionOnlyLM

OMP_NUM_THREADS = 1  
# ---------- Data ----------
ds = load_dataset("jmcinern/Instruction_Ga_En_for_LoRA")  # expects train/test

# ---------- Model & tokenizer (no tokenizer changes saved) ----------
model_id = "jmcinern/qwen3-8b-base-cpt"
subfolder = "checkpoint-33000"

tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=subfolder, trust_remote_code=True)
# In-memory padding only (no vocab change, no save)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    subfolder=subfolder,
    torch_dtype=(torch.bfloat16 if torch.cuda.is_available()
                 and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16),
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.eos_token_id  # runtime-only convenience

# ---------- LoRA ----------
lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.1, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
)
model = get_peft_model(model, lora)

# ---------- Qwen3 chat formatting + response-only loss ----------
def _user_text(inst, ctx):
    return f"{inst}\n\n{ctx}" if ctx and ctx.strip() else inst

def formatting_func(ex):
    messages = [
        {"role":"system","content":""},  # neutral
        {"role":"user","content": _user_text(ex["instruction"], ex.get("context",""))},
        {"role":"assistant","content": ex["response"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

assistant_prefix = tokenizer.apply_chat_template(
    [{"role":"assistant","content":""}], tokenize=False, add_generation_prompt=False
).lstrip()

collator = DataCollatorForCompletionOnlyLM(
    response_template=assistant_prefix,
    tokenizer=tokenizer,
)

# ---------- Training ----------
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
    bf16=(model.dtype == torch.bfloat16),
    fp16=(model.dtype == torch.float16),
    packing=True,
    optim="adamw_torch_fused",
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,   # good default for DDP
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=cfg,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    formatting_func=formatting_func,
    data_collator=collator,
)

trainer.train()
# Save only adapters (tokenizer unchanged on disk)
model.save_pretrained("qwen3-8b-ga-en-lora")

