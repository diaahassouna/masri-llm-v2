%%writefile train_ddp.py
import os
import gc
import torch

#COMMAND PROMPT
#!torchrun --nproc_per_node=2 train_ddp.py
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from huggingface_hub import login
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXX"  # Replace with your HF token
os.environ["HF_TOKEN"] = HF_TOKEN

# 1. Keep 4096 seq length so long examples are retained
max_seq_length = 4096
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

# 2. Load Local Dataset
dataset = load_dataset("json", data_files="masri-llm-v2/data/train.jsonl", split="train")

# 3. Standard QLoRA Setup
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

# 4. Format Prompts
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

dataset = dataset.map(format_prompts, batched = True, num_proc = 2)

# 5. Token Length Filter
def filter_by_token_length(batch):
    tokenized_inputs = tokenizer(batch["text"], truncation=False)["input_ids"]
    assistant_marker = "<|im_start|>assistant"
    
    keep_mask = []
    for text, ids in zip(batch["text"], tokenized_inputs):
        is_valid = (assistant_marker in text) and (len(ids) <= max_seq_length)
        keep_mask.append(is_valid)
    return keep_mask

dataset = dataset.filter(filter_by_token_length, batched = True, batch_size = 1000)

# Verify dataset is non-empty on Rank 0
local_rank = int(os.getenv("LOCAL_RANK", "0"))
if local_rank == 0:
    print(f"✅ Filtered Dataset Size: {len(dataset)} valid examples (<= {max_seq_length} tokens)")
    assert len(dataset) > 0, "Dataset is empty after filtering! Increase max_seq_length."

gc.collect()
torch.cuda.empty_cache()

# 6. DDP Trainer Setup
trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        per_device_train_batch_size = 1,      # 1 per GPU (2 total across GPUs)
        gradient_accumulation_steps = 4,      # 1 sample * 2 GPUs * 4 accum = 8 effective batch size
        ddp_find_unused_parameters = False,
        num_train_epochs = 1,
        learning_rate = 1e-4,
        weight_decay = 0.01,
        warmup_steps = 20,
        lr_scheduler_type = "cosine",
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        output_dir = "masri_qwen3_lora",
    ),
)

# 7. Mask User Prompts
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

trainer_stats = trainer.train()

# 8. Save Weights (Rank 0)
if local_rank == 0:
    model.save_pretrained("masri_qwen3_lora")
    tokenizer.save_pretrained("masri_qwen3_lora")
    
    HF_USERNAME = "sightlake"
    HUB_MODEL_ID = f"{HF_USERNAME}/masri-qwen3-4b-lora"
    model.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN)
    tokenizer.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN)
    print(f"🚀 Model pushed successfully: https://huggingface.co/{HUB_MODEL_ID}")
