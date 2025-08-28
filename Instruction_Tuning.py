# pip install -U "transformers>=4.53" "trl>=0.9.7" peft accelerate datasets

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# Load the EN-GA prompt-response dataset from HF
ds = load_dataset("jmcinern/Instruction_Ga_En_for_LoRA")
#print(ds["train"][:5])

# pre-trained Qwen-3 model on Irish text
model_id = "Qwen/Qwen3-0.6B-base" #"jmcinern/qwen3-8b-base-cpt"
subfolder = ""#"checkpoint-33000"

# format the dataset
def to_messages(ex):
    user = ex["instruction"] + (("\n\n" + ex["context"]) if ex.get("context") else "")
    return {"messages": [
        {"role":"system","content":""},
        {"role":"user","content": user},
        {"role":"assistant","content": ex["response"]},
    ]}

cols = ds["train"].column_names
ds = ds.map(to_messages, remove_columns=[c for c in cols if c != "messages"])

# use .apply_chat_template() to format messages for the model, conversation special tokens
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, subfolder=subfolder)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

ds = ds.map(lambda ex: tokenizer.apply_chat_template(ex["messages"], tokenize=False), batched=True)

print(ds["train"][:5])



dtype = (torch.bfloat16 if torch.cuda.is_available()
         and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16)

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=dtype, trust_remote_code=True, subfolder=subfolder
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.eos_token_id

# ----- LoRA -----
peft_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
)

model = get_peft_model(model, peft_cfg) # model weights frozen while training.

# ----- TRL (new API) -----
cfg = SFTConfig( 
    output_dir="qwen3-8b-lora-bilingual",
    max_length=2048,                 # <— use max_length, shorter for test
    packing=True,                    # uses max_length for block size 
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    logging_steps=20,
    save_steps=1000,
    eval_strategy="steps",
    eval_steps=1000,
    bf16=(dtype == torch.bfloat16),
    fp16=(dtype == torch.float16),
    optim="adamw_torch_fused",
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=cfg,
    train_dataset=ds["train"],  
    eval_dataset=ds["test"],
)

trainer.train()