import torch
import os
import gc
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login, HfApi

# 1. SECURE CREDENTIAL LOADING
if os.path.exists("/kaggle"):
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
    HF_USERNAME = user_secrets.get_secret("HF_USERNAME")
else:
    from dotenv import load_dotenv
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_USERNAME = os.getenv("HF_USERNAME")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found! Check your .env file or Kaggle Secrets.")

login(token=HF_TOKEN)

# Callback to upload checkpoints to HuggingFace for persistence across Kaggle sessions
class CheckpointUploadCallback(TrainerCallback):
    def __init__(self, repo_id, token):
        self.repo_id = repo_id
        self.token = token
        self.api = HfApi()
        self.repo_created = False

    def on_save(self, args, state, control, **kwargs):
        """Upload checkpoint to HuggingFace after each save"""
        # Create repo on first save if it doesn't exist
        if not self.repo_created:
            try:
                self.api.create_repo(
                    repo_id=self.repo_id,
                    token=self.token,
                    repo_type="model",
                    exist_ok=True,  # Don't error if repo already exists
                )
                print(f">>> ✓ Checkpoint repo ready: {self.repo_id}")
                self.repo_created = True
            except Exception as e:
                print(f">>> ✗ Failed to create checkpoint repo: {e}")
                return control

        checkpoint_folder = f"checkpoint-{state.global_step}"
        checkpoint_path = os.path.join(args.output_dir, checkpoint_folder)

        if os.path.exists(checkpoint_path):
            print(f">>> Uploading {checkpoint_folder} to HuggingFace: {self.repo_id}")
            try:
                self.api.upload_folder(
                    folder_path=checkpoint_path,
                    repo_id=self.repo_id,
                    path_in_repo=checkpoint_folder,
                    token=self.token,
                    repo_type="model",
                )
                print(f">>> ✓ Checkpoint uploaded successfully")
            except Exception as e:
                print(f">>> ✗ Failed to upload checkpoint: {e}")
        return control

# 2. PATH CONFIGURATION
if os.path.exists("/kaggle"):
    BASE_PATH = Path("/kaggle/working/fiction_repo/data/train")
    print(f">>> Kaggle Mode: Using data from repo: {BASE_PATH}")
else:
    BASE_PATH = Path("./data/train")
    print(f">>> Local Mode: Using data from: {BASE_PATH.absolute()}")

def train_and_upload(dataset_name, output_name, num_epochs=1):
    print(f"\n>>> Starting Training: {output_name}")

    # Aggressive memory cleanup before starting
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f">>> GPU Memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")

    # Verify file existence
    train_file = BASE_PATH / f"{dataset_name}_train.jsonl"
    val_file = BASE_PATH / f"{dataset_name}_val.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"Could not find dataset file: {train_file}")

    # 4-bit quantization config (fits on T4 GPU)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config (same parameters as Unsloth version)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    # Load dataset
    dataset = load_dataset('json', data_files={
        'train': str(train_file),
        'test': str(val_file)
    })

    def format_prompts(examples):
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
                 for convo in examples["messages"]]
        return {"text": texts}

    dataset = dataset.map(format_prompts, batched=True)

    # SFT Training configuration (2026 API)
    training_args = SFTConfig(
        output_dir=f"outputs/{output_name}",  # Unique dir per model
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=num_epochs,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        save_strategy="steps",
        save_steps=200,  # Save checkpoint every 200 steps (~1.3 hours)
        save_total_limit=2,  # Keep only the 2 most recent checkpoints to save space
        report_to="none",
        gradient_checkpointing=True,
        # SFT-specific parameters
        max_length=2048,  # Max sequence length for tokenization
        dataset_text_field="text",  # Column name containing training text
    )

    # Trainer with checkpoint upload callback
    checkpoint_callback = CheckpointUploadCallback(
        repo_id=f"{HF_USERNAME}/{output_name}-checkpoints",
        token=HF_TOKEN
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        processing_class=tokenizer,
        callbacks=[checkpoint_callback],
    )

    print(f">>> Training {output_name}...")

    # Check for existing checkpoint to resume from
    checkpoint_dir = f"outputs/{output_name}"
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")] if os.path.exists(checkpoint_dir) else []
    resume_from_checkpoint = None

    # First, try to find local checkpoints
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        resume_from_checkpoint = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f">>> Found local checkpoint: {resume_from_checkpoint}")
    else:
        # If no local checkpoints, try downloading from HuggingFace
        print(f">>> No local checkpoints found. Checking HuggingFace for existing checkpoints...")
        try:
            from huggingface_hub import snapshot_download
            checkpoint_repo = f"{HF_USERNAME}/{output_name}-checkpoints"

            # Try to download the checkpoint repo
            downloaded_path = snapshot_download(
                repo_id=checkpoint_repo,
                token=HF_TOKEN,
                local_dir=checkpoint_dir,
                repo_type="model",
            )

            # Check for checkpoints in downloaded path
            checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                resume_from_checkpoint = os.path.join(checkpoint_dir, latest_checkpoint)
                print(f">>> Downloaded checkpoint from HuggingFace: {resume_from_checkpoint}")
        except Exception as e:
            print(f">>> No existing checkpoints on HuggingFace: {e}")
            print(f">>> Starting training from scratch")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save LoRA adapters
    print(f">>> Saving LoRA adapters to outputs/{output_name}")
    model.save_pretrained(f"outputs/{output_name}")
    tokenizer.save_pretrained(f"outputs/{output_name}")

    # Upload LoRA adapters to HuggingFace
    print(f">>> Uploading LoRA adapters to: {HF_USERNAME}/{output_name}")
    model.push_to_hub(f"{HF_USERNAME}/{output_name}", token=HF_TOKEN)
    tokenizer.push_to_hub(f"{HF_USERNAME}/{output_name}", token=HF_TOKEN)

    # GGUF Export for LM Studio
    print(f">>> Merging LoRA adapters and exporting to GGUF...")

    # Free up memory before merging
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Load base model in 16-bit for merging (needs more memory temporarily)
    print(">>> Loading base model for merging...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Load and merge LoRA adapters
    print(">>> Merging LoRA adapters into base model...")
    merged_model = PeftModel.from_pretrained(base_model, f"outputs/{output_name}")
    merged_model = merged_model.merge_and_unload()

    # Save merged model
    merged_dir = f"outputs/{output_name}_merged"
    print(f">>> Saving merged model to {merged_dir}")
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    # Upload merged model to HuggingFace
    print(f">>> Uploading merged model to: {HF_USERNAME}/{output_name}-merged")
    merged_model.push_to_hub(f"{HF_USERNAME}/{output_name}-merged", token=HF_TOKEN)
    tokenizer.push_to_hub(f"{HF_USERNAME}/{output_name}-merged", token=HF_TOKEN)

    # Convert to GGUF using llama.cpp
    print(f">>> Converting to GGUF format...")
    try:
        # Install llama-cpp-python if not available
        import subprocess
        subprocess.run(["pip", "install", "llama-cpp-python", "--quiet"], check=True)

        # Convert to GGUF (q4_k_m quantization for LM Studio)
        from llama_cpp import Llama

        gguf_path = f"outputs/{output_name}.gguf"
        print(f">>> Creating GGUF file: {gguf_path}")

        # Use llama.cpp convert script
        subprocess.run([
            "python", "-m", "llama_cpp.convert",
            merged_dir,
            "--outfile", gguf_path,
            "--outtype", "q4_k_m"
        ], check=True)

        print(f">>> GGUF export complete: {gguf_path}")
        print(f">>> Note: Upload {gguf_path} to HuggingFace manually or use LM Studio locally")

    except Exception as e:
        print(f">>> GGUF conversion failed: {e}")
        print(f">>> You can convert manually using llama.cpp later")
        print(f">>> The merged model is available at: {HF_USERNAME}/{output_name}-merged")

    # Aggressive cleanup after completion
    del merged_model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f">>> GPU Memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")

    print(f">>> ✓ Completed: {output_name}")

# List of models to run
configs = [
    # ("one_liner", "llama-3.1-8b-one-liner", 3),  # Already completed
    ("combined", "llama-3.1-8b-combined", 1),
    ("short_story", "llama-3.1-8b-short-story", 1)
]

if __name__ == "__main__":
    for d_name, o_name, epochs in configs:
        train_and_upload(d_name, o_name, epochs)

    print("\n" + "="*50)
    print("🎉 ALL MODELS COMPLETED!")
    print("="*50)
    print("\nYou now have:")
    print("1. LoRA adapters on HuggingFace (for fine-tuning)")
    print("2. Merged models on HuggingFace (for inference)")
    print("3. GGUF files locally (for LM Studio)")
    print("\nTo use in LM Studio:")
    print("- Download the .gguf files from outputs/")
    print("- Or pull the merged models from HuggingFace")
