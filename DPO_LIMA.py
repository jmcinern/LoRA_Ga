# train_dpo.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

MODEL_ID = "jmcinern/qwen3-8b-base-cpt"
SUBFOLDER = "checkpoint-33000"
DATASET = "jmcinern/LIMA_ga"
DATA_FILE = "translated_IRT_ga.jsonl"   # adjust if needed

print("Loading dataset...")
raw = load_dataset(DATASET, data_files={"train": DATA_FILE})  # explicit split

print("Loading tokenizer/model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, subfolder=SUBFOLDER)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, subfolder=SUBFOLDER, torch_dtype="auto", device_map="auto"
)

# If you want the model prompted in chat style, template ONLY the prompt (not completions)
def to_dpo(example):
    prompt_text = example["instruction"]
    msgs = [{"role": "user", "content": prompt_text}]
    # Disable thinking in Qwen3 template to keep prefix consistent
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return {
        "prompt": prompt,
        "chosen": example["response1"],
        "rejected": example["response2"],
    }

dataset = raw["train"].map(to_dpo, remove_columns=raw["train"].column_names)

print("Setting up DPO training...")
args = DPOConfig(
    output_dir="./qwen3-dpo-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    # DPO-specific knobs (optional, shown for clarity)
    beta=0.1,                    # default
    max_prompt_length=512,       # adjust as needed
    max_completion_length=512,   # adjust as needed
    bf16=True,                   # if your GPUs support bfloat16
)

trainer = DPOTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    processing_class=tokenizer,  # <- correct per docs
    # ref_model=None  # omit; trainer will clone a reference automatically
)

print("Starting training...")
trainer.train()
trainer.save_model("./qwen3-dpo-finetuned")
tokenizer.save_pretrained("./qwen3-dpo-finetuned")
print("Done.")
