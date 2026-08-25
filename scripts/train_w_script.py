%%writefile train_ddp.py
import os
import gc
import torch

#It comes with the script "command_script.py"

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from huggingface_hub import login
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXX"  # Replace with your HF token
os.environ["HF_TOKEN"] = HF_TOKEN

max_seq_length = 2048
model_name = "unsloth/Qwen3-4B-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    token = HF_TOKEN,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
)

dataset = load_dataset("json", data_files="masri-llm-v2/data/train.jsonl", split="train")

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

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

dataset = dataset.map(format_prompts, batched = True, num_proc = 4)

local_rank = int(os.getenv("LOCAL_RANK", "0"))
if local_rank == 0:
    print(f"✅ Active Dataset Size: {len(dataset)} examples")

gc.collect()
torch.cuda.empty_cache()

trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 4,
        packing = True,
        per_device_train_batch_size = 1,       # Fits completely in VRAM (No gradient offloading)
        gradient_accumulation_steps = 4,       # Effective batch size = 8
        max_steps = 150,                       # Exactly 1 full epoch of packed sequence data
        ddp_find_unused_parameters = False,
        learning_rate = 2e-4,
        weight_decay = 0.01,
        warmup_steps = 15,
        lr_scheduler_type = "cosine",
        fp16 = True,
        bf16 = False,
        logging_steps = 5,
        optim = "adamw_8bit",
        output_dir = "masri_qwen3_lora",
        report_to = "none",
    ),
)

trainer.train()

if local_rank == 0:
    model.save_pretrained("masri_qwen3_lora")
    tokenizer.save_pretrained("masri_qwen3_lora")
    
    HF_USERNAME = "sightlake"
    HUB_MODEL_ID = f"{HF_USERNAME}/masri-qwen3-4b-lora"
    model.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN)
    tokenizer.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN)
    print(f"🚀 Model pushed successfully: https://huggingface.co/{HUB_MODEL_ID}")
