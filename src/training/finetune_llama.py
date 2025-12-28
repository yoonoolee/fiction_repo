"""
Fine-tune Llama-3.1-8B on Reddit TIFU + AO3 Stories

Optimized for A100 GPU with ~12k training examples.
Estimated training time: ~15-30 minutes for 5 epochs.

Requirements:
- Lightning AI Studio with A100 GPU
- HuggingFace token for Llama access
- Upload tifu_ao3.csv to root directory
"""

import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = "tifu_ao3.csv"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
OUTPUT_DIR = "./llama-3.1-reddit-ao3-lora"

# Training hyperparameters (optimized for A100 + 12k rows)
NUM_EPOCHS = 5
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 32
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 80

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ============================================================================
# HUGGINGFACE LOGIN
# ============================================================================

print("Logging in to Hugging Face...")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("Warning: HF_TOKEN not found. Run: huggingface-cli login")

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"\nLoading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total examples: {len(df):,}")

# Train/val/test splits (80/10/10)
train_size = int(0.8 * len(df))
val_size = int(0.1 * len(df))

train_df = df[:train_size]
val_df = df[train_size:train_size + val_size]
test_df = df[train_size + val_size:]

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df[['text']])
val_dataset = Dataset.from_pandas(val_df[['text']])
test_dataset = Dataset.from_pandas(test_df[['text']])

# ============================================================================
# LOAD MODEL
# ============================================================================

print(f"\nLoading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.config.use_cache = False

print("✓ Model loaded")

# ============================================================================
# CONFIGURE LORA
# ============================================================================

print("\nConfiguring LoRA...")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("✓ LoRA configured")

# ============================================================================
# TOKENIZE DATA
# ============================================================================

print("\nTokenizing data...")

def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print("✓ Data tokenized")

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

print("\nConfiguring training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    max_grad_norm=0.3,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=5,  # Keep all epoch checkpoints to preserve best model
    load_best_model_at_end=True,  # Load best model when training ends
    metric_for_best_model="eval_loss",  # Use validation loss to pick best
    greater_is_better=False,  # Lower eval_loss is better
    bf16=True,
    dataloader_num_workers=4,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ============================================================================
# TRAIN MODEL
# ============================================================================

print("\nInitializing trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80 + "\n")

trainer.train()

print("\n✓ Training complete!")

# ============================================================================
# SAVE MODEL
# ============================================================================

print(f"\nSaving model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✓ Model saved")

# ============================================================================
# EVALUATE ON TEST SET
# ============================================================================

print("\nEvaluating on test set...")
test_results = trainer.evaluate(tokenized_test)

print("\nTest Results:")
print("="*80)
for key, value in test_results.items():
    print(f"  {key}: {value:.4f}")
print("="*80)

# ============================================================================
# TEST GENERATION
# ============================================================================

print("\nTesting generation...")
print("="*80)

model.eval()

prompts = [
    "Sarah and Emma",
    "The moment when",
    "In the darkness,",
    "Alex couldn't believe",
    "They finally"
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Output: {generated_text}")
    print("-" * 80)

print("\n✓ DONE! Model saved to:", OUTPUT_DIR)
