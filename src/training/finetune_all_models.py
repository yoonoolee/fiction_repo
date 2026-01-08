from unsloth import FastLanguageModel
import torch
import os
from pathlib import Path
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from huggingface_hub import login

# 1. SECURE CREDENTIAL LOADING
# This part works both locally (.env) and on Kaggle (Secrets)
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

# 2. PATH CONFIGURATION
# Detect if we are on Kaggle or local
BASE_PATH = Path("/kaggle/input/your-dataset-name") if os.path.exists("/kaggle") else Path("./data/train")

def train_and_upload(dataset_name, output_name, num_epochs=1):
    print(f"\n>>> Starting Training: {output_name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-8B-Instruct-bnb-4bit",
        max_seq_length = 2048,
        load_in_4bit = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )

    # Use the secure path we defined
    dataset = load_dataset('json', data_files={
        'train': str(BASE_PATH / f"{dataset_name}_train.jsonl"),
        'test': str(BASE_PATH / f"{dataset_name}_val.jsonl")
    })

    def format_prompts(examples):
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in examples["messages"]]
        return { "text" : texts }

    dataset = dataset.map(format_prompts, batched=True)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        dataset_text_field = "text",
        max_seq_length = 2048,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            num_train_epochs = num_epochs,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            output_dir = "outputs",
            report_to = "none",
        ),
    )
    trainer.train()

    # GGUF Export for LM Studio
    print(f">>> Exporting GGUF to Hugging Face...")
    model.push_to_hub_gguf(
        repo_id = f"{HF_USERNAME}/{output_name}-GGUF",
        tokenizer = tokenizer,
        quantization_method = "q4_k_m",
        token = HF_TOKEN,
    )

    del model, tokenizer, trainer
    torch.cuda.empty_cache()

# List of models to run
configs = [
    ("one_liner", "llama-3.2-8b-one-liner", 3),
    ("combined", "llama-3.2-8b-combined", 1),
    ("short_story", "llama-3.2-8b-short-story", 1)
]

if __name__ == "__main__":
    for d_name, o_name, epochs in configs:
        train_and_upload(d_name, o_name, epochs)