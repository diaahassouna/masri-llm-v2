import os
import torch
from huggingface_hub import login
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 0. Authenticate with Hugging Face
HF_TOKEN = "PUT_YOUR_HUGGING_FACE_ACCESS_TOKEN_HERE"  # Replace with your newly generated token
os.environ["HF_TOKEN"] = HF_TOKEN
login(token=HF_TOKEN)

# 1. Config & Base Model Selection (512 max length for faster training)
max_seq_length = 512
model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    token = HF_TOKEN,
)

# 2. Attach Fast LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)

# 3. Load Local Dataset
dataset = load_dataset("json", data_files="masri-llm-v2/data/train.jsonl", split="train")

# 4. Format Messages to ChatML
def format_prompts(examples):
    texts = []
    for msg_list in examples["messages"]:
        formatted_chat = ""
        for msg in msg_list:
            role = msg["role"]
            content = msg["content"]
            formatted_chat += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        texts.append(formatted_chat.strip())
    return { "text" : texts }

dataset = dataset.map(format_prompts, batched = True)

# 5. Fast Training Loop Configuration
trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,                    # Disabled heavy packing for fast speed
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,    # Reduced accumulation overhead
        warmup_steps = 10,
        num_train_epochs = 1,               # 1 pass is ideal for 5k dialect pairs
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        output_dir = "masri_lora_model",
    ),
)

trainer_stats = trainer.train()

# 6. Save Fine-Tuned LoRA Weights Locally
model.save_pretrained("masri_lora_model")
tokenizer.save_pretrained("masri_lora_model")
print("✅ Local saving complete!")

# 7. Push LoRA Model to Hugging Face Hub
HF_USERNAME = "sightlake"
HUB_MODEL_ID = f"{HF_USERNAME}/masri-qwen2.5-3b-lora"

model.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN)
tokenizer.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN)
print(f"🚀 Model pushed successfully: https://huggingface.co/{HUB_MODEL_ID}")
