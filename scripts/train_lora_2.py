import os
import gc
import torch

# 1. System environment adjustments
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from huggingface_hub import login
from unsloth import FastLanguageModel, add_new_tokens
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 0. Authenticate with Hugging Face
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXX"  # Replace with your HF token
os.environ["HF_TOKEN"] = HF_TOKEN
login(token=HF_TOKEN)

# 1. Load Qwen3-4B-Base in 4-bit QLoRA
max_seq_length = 512
model_name = "unsloth/Qwen3-4B-bnb-4bit"  # 4-bit quantized base model

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    token = HF_TOKEN,
)

# Apply standard ChatML template for Qwen Base models
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
)

# 2. Load Local Dataset
dataset = load_dataset("json", data_files="masri-llm-v2/data/train.jsonl", split="train")

# 3. Detect and add missing characters
def find_uncovered_chars(dataset, tokenizer):
    chars = set()
    for msg_list in dataset["messages"]:
        for msg in msg_list:
            chars.update(msg["content"])
    uncovered = []
    for ch in sorted(chars):
        if ch.isascii():
            continue
        n_tokens = len(tokenizer.encode(ch, add_special_tokens=False))
        if n_tokens != 1:
            uncovered.append(ch)
    return uncovered

new_chars = find_uncovered_chars(dataset, tokenizer)
print(f"Adding {len(new_chars)} new tokens: {new_chars}")

if new_chars:
    add_new_tokens(model, tokenizer, new_tokens=new_chars)

# 4. Attach QLoRA Adapters targeting Qwen linear layers
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    modules_to_save = ["embed_tokens", "lm_head"] if new_chars else None,
)

# 5. Format Prompts via ChatML template
def format_prompts(examples):
    texts = []
    for msg_list in examples["messages"]:
        formatted = tokenizer.apply_chat_template(
            msg_list, 
            tokenize=False, 
            add_generation_prompt=False
        )
        texts.append(formatted)
    return { "text" : texts }

dataset = dataset.map(format_prompts, batched = True)

gc.collect()
torch.cuda.empty_cache()

# 6. Trainer Setup
trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 10,
        num_train_epochs = 3,
        learning_rate = 2e-4,                   # Qwen models respond well to slightly higher LR than Gemma
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        output_dir = "masri_qwen3_lora",
    ),
)
trainer_stats = trainer.train()

# 7. Save & Push LoRA Weights
model.save_pretrained("masri_qwen3_lora")
tokenizer.save_pretrained("masri_qwen3_lora")

HF_USERNAME = "sightlake"
HUB_MODEL_ID = f"{HF_USERNAME}/masri-qwen3-4b-lora"
model.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN)
tokenizer.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN)
print(f"🚀 Model pushed successfully: https://huggingface.co/{HUB_MODEL_ID}")
