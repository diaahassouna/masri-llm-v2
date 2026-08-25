#run by "command_script_opt.py"

%%writefile train_ddp.py
import os
import gc
import torch
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from huggingface_hub import login
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXX"  # Replace with your HF token
os.environ["HF_TOKEN"] = HF_TOKEN

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
        # enable_thinking=False: Qwen3 is a hybrid reasoning model whose default
        # template/behavior emits <think>...</think>. We don't want that for a
        # deterministic Franco->Masri conversion task, so we disable it explicitly
        # and consistently here AND at inference time.
        formatted = tokenizer.apply_chat_template(
            msg_list,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        texts.append(formatted)
    return { "text" : texts }

dataset = dataset.map(format_prompts, batched = True, num_proc = 4)

local_rank = int(os.getenv("LOCAL_RANK", "0"))
if local_rank == 0:
    print(f"✅ Active Dataset Size (pre-packing): {len(dataset)} examples")
    # Sanity check: confirm the formatted text actually contains both the Franco
    # input and the Masri completion in the shape we expect, before burning compute.
    print("----- Sample formatted example -----")
    print(dataset[0]["text"])
    print("-------------------------------------")

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
        packing = False,                       # train_on_responses_only's marker search is
                                                # unreliable against packed/concatenated
                                                # sequences (it masks every label to -100 -
                                                # "nothing to train on"). Unsloth's own
                                                # notebooks that use train_on_responses_only
                                                # run with packing off for this reason.
        per_device_train_batch_size = 1,       # Fits completely in VRAM (No gradient offloading)
        gradient_accumulation_steps = 4,       # Effective batch size = 8
        num_train_epochs = 1,                  # With packing off, this is computed directly
                                                # from the real example count (5,143), so it's
                                                # an honest "3 full passes" - no more guessing
                                                # steps against a packed length like before.
                                                # NOTE: without packing, each of your 5,143
                                                # examples pays the full system-prompt token
                                                # cost individually (it looks like several
                                                # hundred tokens), so this run will take
                                                # meaningfully longer wall-clock than the
                                                # packed 150-step run. If you hit a session
                                                # time limit, drop this to 2 first.
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

# Mask loss on the prompt (Franco input) tokens so the model only learns to
# predict the assistant's Masri completion, not to reproduce the user's input.
# Without this, loss collapses fast by trivially "learning" to echo/continue
# short packed sequences rather than the actual conversion task.
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

if local_rank == 0:
    import math
    batches_per_epoch = len(trainer.get_train_dataloader())
    total_steps = math.ceil(batches_per_epoch * trainer.args.num_train_epochs / trainer.args.gradient_accumulation_steps)
    print(f"Total optimizer steps for {trainer.args.num_train_epochs} epoch(s): {total_steps}")

trainer.train()

if local_rank == 0:
    model.save_pretrained("masri_qwen3_lora")
    tokenizer.save_pretrained("masri_qwen3_lora")

    HF_USERNAME = "sightlake"
    HUB_MODEL_ID = f"{HF_USERNAME}/masri-qwen3-4b-lora"
    model.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN)
    tokenizer.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN)
    print(f"🚀 Model pushed successfully: https://huggingface.co/{HUB_MODEL_ID}")
