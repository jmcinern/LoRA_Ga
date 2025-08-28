from transformers import AutoTokenizer
import glob
import os

model_id = "Qwen/Qwen3-1.7B-base"
output_dir = "qwen3-8b-lora-bilingual"   # same as your SFTConfig.output_dir
# If you saved checkpoints during training, pick the latest; else it uses output_dir
ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")),
               key=lambda p: int(p.split("-")[-1]))
lora_dir = ckpts[-1] if ckpts else output_dir
tok = AutoTokenizer.from_pretrained(lora_dir or model_id, trust_remote_code=True)

to_id = tok.convert_tokens_to_ids
im_end = to_id("<|im_end|>")
eot    = to_id("<|endoftext|>")
unk    = tok.unk_token_id

print("IM_END_ID:", im_end)
print("ENDTEXT_ID:", eot)
print("UNK_ID:", unk)
print("Distinct & valid:",
      im_end not in (None, unk) and eot not in (None, unk) and im_end != eot)
print(tok.special_tokens_map)
