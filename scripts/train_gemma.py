import os
import torch
from huggingface_hub import login
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 0. Authenticate with Hugging Face
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"  # Replace with your active HF token
os.environ["HF_TOKEN"] = HF_TOKEN
login(token=HF_TOKEN)

# 1. Config & Base Model Selection
max_seq_length = 512
model_name = "unsloth/gemma-2-2b-it-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    token = HF_TOKEN,
)

# 2. Load Local Dataset
dataset = load_dataset("json", data_files="masri-llm-v2/data/train.jsonl", split="train")

# 3. Detect and add missing characters to vocabulary
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
    torch.cuda.empty_cache()
    tokenizer.add_tokens(new_chars)
    # mean_resizing=False disables the 256k-vocab covariance matrix calculation that triggers OOM on T4
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    
    with torch.no_grad():
        embed = model.get_input_embeddings()
        mean_embedding = embed.weight[:-len(new_chars)].mean(dim=0)
        for i in range(len(new_chars)):
            embed.weight[-(i + 1)] = mean_embedding

# 4. Attach Fast LoRA Adapters
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

# 5. Format Prompts for Gemma 2 (Handles System role and mapping)
def format_prompts(examples):
    texts = []
    for msg_list in examples["messages"]:
        cleaned_msgs = []
        sys_text = ""
        
        for m in msg_list:
            if m["role"] == "system":
                sys_text += m["content"] + "\n\n"
            else:
                role = "model" if m["role"] == "assistant" else m["role"]
                cleaned_msgs.append({"role": role, "content": m["content"]})
        
        if sys_text and cleaned_msgs and cleaned_msgs[0]["role"] == "user":
            cleaned_msgs[0]["content"] = sys_text + cleaned_msgs[0]["content"]
            
        formatted = tokenizer.apply_chat_template(
            cleaned_msgs, 
            tokenize=False, 
            add_generation_prompt=False
        )
        texts.append(formatted)
    return { "text" : texts }

dataset = dataset.map(format_prompts, batched = True)

# 6. Training Loop Configuration
trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 3,
        learning_rate = 5e-5,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        output_dir = "masri_lora_model",
    ),
)
trainer_stats = trainer.train()

# 7. Save Fine-Tuned LoRA Weights Locally
model.save_pretrained("masri_lora_model")
tokenizer.save_pretrained("masri_lora_model")
print("✅ Local saving complete!")

# 8. Push LoRA Model to Hugging Face Hub
HF_USERNAME = "sightlake"
HUB_MODEL_ID = f"{HF_USERNAME}/masri-gemma2-2b-lora"
model.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN)
tokenizer.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN)
print(f"🚀 Model pushed successfully: https://huggingface.co/{HUB_MODEL_ID}")

# --- Native Inference Check ---
FastLanguageModel.for_inference(model)
test_prompt = "azayk ya basha 3amel eh el naharda"
messages = [{"role": "user", "content": test_prompt}]
inputs = tokenizer.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True, 
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=64,
    do_sample=True,
    temperature=0.3,
    pad_token_id=tokenizer.eos_token_id
)
response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(f"Input:  {test_prompt}")
print(f"Masri:  {response.strip()}")
