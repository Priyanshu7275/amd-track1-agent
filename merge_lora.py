"""
Merge the trained LoRA adapter back into the base model weights.

The result is a standalone model with no PEFT dependency, ready for
GGUF conversion. Takes ~30 seconds on the MI300X.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "/workspace/amd-track1-finetune/base_model"
ADAPTER = "/workspace/amd-track1-finetune/lora_adapter"
MERGED = "/workspace/amd-track1-finetune/merged_model"

model = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.float16, device_map="cuda"
)
model = PeftModel.from_pretrained(model, ADAPTER)

# Fold the low-rank updates into the base weights, then drop the adapter
model = model.merge_and_unload()

model.save_pretrained(MERGED, safe_serialization=True)   # ~3.1 GB fp16

tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.save_pretrained(MERGED)

print(f"Merged model saved to {MERGED}")
