"""
LoRA fine-tuning of Qwen2.5-1.5B-Instruct on AMD Instinct MI300X.

Environment: AMD Developer Cloud, ROCm 7.2 + PyTorch 2.9
Result: loss 1.77 -> 0.28 over 3 epochs (13.5 min, 5657 examples)

Note: bitsandbytes 8-bit quantisation is avoided — it has known issues
on ROCm. fp16 training is used instead, which the MI300X handles easily
given its 51 GB of VRAM.
"""
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

MODEL_PATH = "/workspace/amd-track1-finetune/base_model"
OUTPUT_DIR = "/workspace/amd-track1-finetune/lora_adapter"
DATA_PATH = "/workspace/amd-track1-finetune/train_data.jsonl"

# ---- load the training data -------------------------------------------------
rows = []
with open(DATA_PATH) as f:
    for line in f:
        rows.append(json.loads(line))
print(f"Loaded {len(rows)} training examples")

# ---- tokenizer --------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# TRL expects a flat "text" column, so we apply the chat template up front
def format_row(row):
    return {
        "text": tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
    }

dataset = Dataset.from_list(rows).map(format_row, remove_columns=["messages"])

# ---- base model -------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="cuda",          # ROCm aliases the CUDA API
)

# ---- LoRA configuration -----------------------------------------------------
# Targeting all attention and MLP projections gives the adapter enough
# capacity to reshape output format without touching the base knowledge.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # 18.4M trainable / 1.56B total (1.18%)

# ---- training ---------------------------------------------------------------
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=5,
    save_strategy="epoch",
    fp16=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")
