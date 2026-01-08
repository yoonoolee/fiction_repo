"""
Fine-tune 3 models with Llama-3.1-8B-Instruct

Trains:
1. Combined model (balanced 10k one-liners + 10k short stories)
2. One-liner model (full ~2k dataset)
3. Short-story model (full ~50k dataset)

Prerequisites: Run notebooks/prepare_training_data.ipynb first to generate JSONL files.

Usage:
    # Local
    python src/training/finetune_all_models.py

    # Lightning.ai (auto-detects paths)
    python finetune_all_models.py

    # Custom paths
    python finetune_all_models.py --train_dir /path/to/data/train --output_dir /path/to/models
"""

import sys
import os
import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model


def get_base_dir():
    """Auto-detect base directory (works for local and lightning.ai)."""
    cwd = Path.cwd()

    # If we're in a git repo (has .git folder), use current directory
    if (cwd / ".git").exists():
        return cwd

    # Check parent directories for .git
    for parent in cwd.parents:
        if (parent / ".git").exists():
            return parent

    # Fall back to current directory
    return cwd


# Configuration
BASE_DIR = get_base_dir()
TRAIN_DIR = BASE_DIR / "data" / "train"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_BASE_DIR = BASE_DIR / "models"

# Training configurations for each model
TRAINING_CONFIGS = [
    {
        "dataset_name": "combined",
        "output_name": "llama-8b-combined",
        "num_epochs": 3,
        "description": "Balanced dataset (10k one-liners + 10k short stories)"
    },
    {
        "dataset_name": "one_liner",
        "output_name": "llama-8b-one-liner",
        "num_epochs": 3,
        "description": "One-liner only (~2k examples)"
    },
    {
        "dataset_name": "short_story",
        "output_name": "llama-8b-short-story",
        "num_epochs": 3,
        "description": "Short-story only (~50k examples)"
    }
]


def finetune_model(dataset_name, output_name, num_epochs=3):
    """
    Fine-tune Llama-3.1-8B-Instruct with LoRA.

    Args:
        dataset_name: Name of dataset (e.g., 'combined', 'one_liner', 'short_story')
        output_name: Name for output directory (e.g., 'llama-8b-combined')
        num_epochs: Number of training epochs (default: 3)

    Returns:
        output_dir: Path to saved model
    """
    print(f"\n{'='*80}")
    print(f"Starting training: {output_name}")
    print(f"{'='*80}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    print(f"Loading {dataset_name} dataset...")
    train_dataset = load_dataset('json', data_files=str(TRAIN_DIR / f"{dataset_name}_train.jsonl"), split='train')
    val_dataset = load_dataset('json', data_files=str(TRAIN_DIR / f"{dataset_name}_val.jsonl"), split='train')

    print(f"Train examples: {len(train_dataset):,}")
    print(f"Val examples: {len(val_dataset):,}")

    # Format and tokenize
    print("Formatting and tokenizing...")
    def format_chat_template(example):
        messages = example['messages']
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    def tokenize_function(examples):
        outputs = tokenizer(examples["text"], truncation=True, max_length=1024, padding="max_length")
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    train_dataset = train_dataset.map(format_chat_template, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(format_chat_template, remove_columns=val_dataset.column_names)

    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Training arguments
    output_dir = OUTPUT_BASE_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        bf16=True,
        report_to="none",
        gradient_checkpointing=True,
    )

    # Create trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting training...\n")
    trainer.train()

    # Save
    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"\n✓ Completed: {output_name}\n")

    # Clean up VRAM
    del model
    del trainer
    torch.cuda.empty_cache()

    return str(output_dir)


def main():
    """Train all 3 models sequentially."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fine-tune 3 Llama models")
    parser.add_argument("--train_dir", type=str, default=None, help="Path to training data directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to output models directory")
    args = parser.parse_args()

    # Use custom paths if provided
    global TRAIN_DIR, OUTPUT_BASE_DIR
    if args.train_dir:
        TRAIN_DIR = Path(args.train_dir)
    if args.output_dir:
        OUTPUT_BASE_DIR = Path(args.output_dir)

    print(f"\n{'='*80}")
    print("FINE-TUNING 3 MODELS WITH LLAMA-3.1-8B-INSTRUCT")
    print(f"{'='*80}\n")
    print(f"Base directory: {BASE_DIR}")
    print(f"Model: {MODEL_NAME}")
    print(f"Training data: {TRAIN_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}\n")

    # Check if training data exists
    if not TRAIN_DIR.exists():
        print(f"ERROR: Training data directory not found: {TRAIN_DIR}")
        print("Please run notebooks/prepare_training_data.ipynb first.")
        print(f"\nCurrent working directory: {Path.cwd()}")
        print(f"Files in current directory: {list(Path.cwd().iterdir())[:10]}")
        sys.exit(1)

    # Train all models
    trained_models = []
    for i, config in enumerate(TRAINING_CONFIGS, 1):
        print(f"\n{'='*80}")
        print(f"MODEL {i}/3: {config['output_name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*80}\n")

        try:
            model_path = finetune_model(
                dataset_name=config['dataset_name'],
                output_name=config['output_name'],
                num_epochs=config['num_epochs']
            )
            trained_models.append(model_path)
        except Exception as e:
            print(f"\n❌ ERROR training {config['output_name']}: {e}")
            import traceback
            traceback.print_exc()
            print("\nContinuing to next model...\n")
            continue

    # Summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}\n")

    if trained_models:
        print(f"Successfully trained {len(trained_models)}/{len(TRAINING_CONFIGS)} models:\n")
        for i, model_path in enumerate(trained_models, 1):
            print(f"{i}. {model_path}")
    else:
        print("❌ No models were successfully trained.")
        sys.exit(1)

    print("\nNext steps: Test the models with inference code.")


if __name__ == "__main__":
    main()
