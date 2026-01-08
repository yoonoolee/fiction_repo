from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template 
import torch
import os
from pathlib import Path
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from huggingface_hub import login

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

# 2. PATH CONFIGURATION (UPDATED FOR GITHUB DATA)
# On Kaggle, the repo is cloned into /kaggle/working/fiction_repo
if os.path.exists("/kaggle"):
    BASE_PATH = Path("/kaggle/working/fiction_repo/data/train")
    print(f">>> Kaggle Mode: Using data from repo: {BASE_PATH}")
else:
    # Local mode: relative to where you run the script
    BASE_PATH = Path("./data/train")
    print(f">>> Local Mode: Using data from: {BASE_PATH.absolute()}")

def train_and_upload(dataset_name, output_name, num_epochs=1):
    print(f"\n>>> Starting Training: {output_name}")
    
    # Verify file existence before starting heavy model load
    train_file = BASE_PATH / f"{dataset_name}_train.jsonl"
    val_file = BASE_PATH / f"{dataset_name}_val.jsonl"
    
    if not train_file.exists():
        raise FileNotFoundError(f"Could not find dataset file: {train_file}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/meta-llama-3.1-8b-bnb-4bit",
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

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # Loading the dataset from the GitHub repo path
    dataset = load_dataset('json', data_files={
        'train': str(train_file),
        'test': str(val_file)
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
    ("one_liner", "llama-3.1-8b-one-liner", 3),
    ("combined", "llama-3.1-8b-combined", 1),
    ("short_story", "llama-3.1-8b-short-story", 1)
]

if __name__ == "__main__":
    for d_name, o_name, epochs in configs:
        train_and_upload(d_name, o_name, epochs)