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

# 1. Config & Base Model Selection
# T4 = 16GB VRAM, single GPU only (Kaggle's 2nd T4 isn't usable by
# open-source Unsloth without manual multi-GPU setup). Starting with 2B
# to confirm the pipeline works within that budget, since full
# embed_tokens/lm_head training on Gemma 2's 256k vocab adds real memory
# on top of the LoRA adapters. Swap to "unsloth/gemma-2-9b-it-bnb-4bit"
# once this runs cleanly, if you want to try for more capacity.
max_seq_length = 512
model_name = "unsloth/gemma-2-2b-it-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    token = HF_TOKEN,
)

# 2. Load Local Dataset (loaded before touching the model so we can scan it)
dataset = load_dataset("json", data_files="masri-llm-v2/data/train.jsonl", split="train")

# 3. Auto-detect Masri characters the tokenizer doesn't already represent cleanly
#    A character is "already fine" if it round-trips as exactly one token.
#    Anything that takes 2+ tokens gets registered as a proper new token, so
#    the model learns it as a single unit instead of a fragile byte sequence.
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
    tokenizer.add_tokens(new_chars)
    model.resize_token_embeddings(len(tokenizer))
    # Seed new embedding rows with the mean of existing ones instead of pure
    # random init — gives training a head start rather than starting from noise.
    with torch.no_grad():
        embed = model.get_input_embeddings()
        mean_embedding = embed.weight[:-len(new_chars)].mean(dim=0)
        for i in range(len(new_chars)):
            embed.weight[-(i + 1)] = mean_embedding

# 4. Attach Fast LoRA Adapters
#    modules_to_save keeps embed_tokens/lm_head FULLY trainable (not LoRA-decomposed)
#    so the new token rows we just added can actually learn — this is required
#    whenever you've resized the vocabulary, otherwise the new rows never update.
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

# 5. Format Messages to ChatML
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
        per_device_train_batch_size = 2,    # halved from 4 to leave headroom for
                                              # the full embed_tokens/lm_head training
        gradient_accumulation_steps = 4,    # doubled to keep effective batch size (8) the same
        warmup_steps = 10,
        num_train_epochs = 3,               # bumped from 1: learning a new script needs more passes
        learning_rate = 2e-4,
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

# --- Inference check (run in the same session) ---
FastLanguageModel.for_inference(model)
test_prompt = "azayk ya basha 3amel eh el naharda"
formatted_input = f"<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(formatted_input, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
