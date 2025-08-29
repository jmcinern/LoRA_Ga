# train.py

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer, DPOConfig

# -------------------------
# 1. Load and preprocess dataset
# -------------------------
print("Loading dataset...")
raw_dataset = load_dataset("jmcinern/LIMA_ga",
                  data_files={"translated_IRT_ga.jsonl"}
)
                  

def to_dpo_format(example):
    return {
        "prompt": example["instruction"],
        "chosen": example["response1"],
        "rejected": example["response2"]
    }

dataset = raw_dataset["train"].map(to_dpo_format, remove_columns=raw_dataset["train"].column_names)

# -------------------------
# 2. Load base model + tokenizer
# -------------------------
print("Loading model and tokenizer...")
model_id = "jmcinern/qwen3-8b-base-cpt"
subfolder = "checkpoint-33000"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    subfolder=subfolder
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    trust_remote_code=True,
    subfolder=subfolder,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id


# -------------------------
# 3. Training setup
# -------------------------
print("Setting up DPO training...")
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=3,
    logging_steps=10,
    output_dir="./qwen3-dpo-finetuned",
    save_strategy="epoch"
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # TRL clones base model internally
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
)

# -------------------------
# 4. Train
# -------------------------
print("Starting training...")
trainer.train()
print("Training finished. Model saved at ./qwen3-dpo-finetuned")
