#!/usr/bin/env python3
"""
train_lora.py — LoRA fine-tune an open base model on the Masri dataset.

Designed to run on a single Colab/free-tier T4 (7B/8B in 4-bit) or better.
Uses TRL SFTTrainer + PEFT LoRA + bitsandbytes 4-bit quantization (QLoRA).

Important:
- Keeps the dataset as a conversational `messages` dataset.
- Uses assistant_only_loss=True so loss is computed only on the assistant/Masri output.
- Prints token-length statistics before training.
- Writes token statistics to <output_dir>/token_stats.json.
- Does NOT flatten messages into a `text` field, because that would prevent
  TRL from using conversational assistant-only masking correctly.

Usage:
  python3 train_lora.py \
      --base_model Qwen/Qwen3-8B \
      --train_file ../data/train.jsonl \
      --eval_file ../data/dev.jsonl \
      --output_dir ../out/masri-lora-v4 \
      --epochs 3
"""

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base_model",
        default="Qwen/Qwen3-8B",
        help=(
            "Base model repo id. Must have solid Arabic + broad Unicode "
            "coverage (Coptic/Greek code points) in its tokenizer."
        ),
    )
    p.add_argument("--train_file", default="../data/train.jsonl")
    p.add_argument("--eval_file", default="../data/dev.jsonl")
    p.add_argument("--output_dir", default="../out/masri-lora")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help=(
            "Maximum total sequence length. Token statistics are computed "
            "before truncation so you can see how many examples exceed this."
        ),
    )
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization (use with a sufficiently large GPU).",
    )
    p.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to a checkpoint folder to resume an interrupted run.",
    )
    p.add_argument(
        "--push_to_hub",
        default=None,
        help="Optional Hugging Face repo id to push after training.",
    )
    return p.parse_args()


def _percentile(values, pct):
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    rank = (len(xs) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return float(xs[lo] + frac * (xs[hi] - xs[lo]))


def _summary(values):
    if not values:
        return {
            "count": 0,
            "min": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0,
        }
    return {
        "count": len(values),
        "min": int(min(values)),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "p90": round(_percentile(values, 0.90), 2),
        "p95": round(_percentile(values, 0.95), 2),
        "max": int(max(values)),
    }


def analyze_token_lengths(ds, tokenizer, max_seq_len, split_name):
    """
    Tokenize each conversational example using the model's chat template.

    For assistant token counts, use Transformers' assistant-token mask.
    TRL patches supported chat templates for assistant-only loss; Qwen3 is
    one of the supported families.
    """
    total_lengths = []
    assistant_lengths = []
    prompt_lengths = []
    over_limit = 0
    no_assistant_mask = 0

    for ex in ds:
        messages = ex["messages"]

        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
        except TypeError:
            # Older Transformers may not accept return_dict here.
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_assistant_tokens_mask=True,
            )
            if isinstance(encoded, dict):
                pass
            else:
                # No assistant mask available: fall back to exact total length.
                encoded = {"input_ids": encoded, "assistant_masks": None}

        input_ids = encoded["input_ids"]
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]

        total_len = len(input_ids)
        total_lengths.append(total_len)

        if total_len > max_seq_len:
            over_limit += 1

        mask = encoded.get("assistant_masks", encoded.get("assistant_token_mask"))
        if mask is None:
            # Transformers/TRL naming has varied; try the common key.
            mask = encoded.get("assistant_mask")

        if mask is None:
            no_assistant_mask += 1
            continue

        if hasattr(mask, "tolist"):
            mask = mask.tolist()
        if mask and isinstance(mask[0], list):
            mask = mask[0]

        assistant_len = int(sum(mask))
        assistant_lengths.append(assistant_len)
        prompt_lengths.append(max(total_len - assistant_len, 0))

    report = {
        "split": split_name,
        "max_seq_len": max_seq_len,
        "total_tokens": _summary(total_lengths),
        "assistant_tokens": _summary(assistant_lengths),
        "non_assistant_tokens": _summary(prompt_lengths),
        "examples_over_max_seq_len": over_limit,
        "examples_over_max_seq_len_pct": round(
            100.0 * over_limit / max(len(ds), 1), 2
        ),
        "examples_without_assistant_mask": no_assistant_mask,
        "assistant_mask_available_pct": round(
            100.0 * (len(ds) - no_assistant_mask) / max(len(ds), 1), 2
        ),
    }

    return report


def print_token_report(report):
    print()
    print("=" * 72)
    print(f"TOKEN LENGTH REPORT — {report['split']}")
    print("=" * 72)
    print(f"max_seq_len:                 {report['max_seq_len']}")
    print(
        f"total tokens:                mean={report['total_tokens']['mean']}, "
        f"median={report['total_tokens']['median']}, "
        f"p95={report['total_tokens']['p95']}, "
        f"max={report['total_tokens']['max']}"
    )
    print(
        f"assistant tokens:            mean={report['assistant_tokens']['mean']}, "
        f"median={report['assistant_tokens']['median']}, "
        f"p95={report['assistant_tokens']['p95']}, "
        f"max={report['assistant_tokens']['max']}"
    )
    print(
        f"non-assistant tokens:        mean={report['non_assistant_tokens']['mean']}, "
        f"median={report['non_assistant_tokens']['median']}, "
        f"p95={report['non_assistant_tokens']['p95']}, "
        f"max={report['non_assistant_tokens']['max']}"
    )
    print(
        f"over max_seq_len:            {report['examples_over_max_seq_len']} "
        f"({report['examples_over_max_seq_len_pct']}%)"
    )
    print(
        f"assistant mask available:    {report['assistant_mask_available_pct']}%"
    )
    print("=" * 72)
    print()


def main():
    args = parse_args()

    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Keep tokenizer/model special-token alignment explicit.
    if tokenizer.pad_token_id is not None:
        print(
            f"Tokenizer special tokens: "
            f"pad={tokenizer.pad_token_id}, "
            f"bos={tokenizer.bos_token_id}, "
            f"eos={tokenizer.eos_token_id}"
        )

    # Qwen3 is supported by TRL's training chat-template patching for
    # assistant-only loss. SFTTrainer will also handle this path when
    # assistant_only_loss=True.
    try:
        from trl.chat_template_utils import get_training_chat_template

        training_template = get_training_chat_template(tokenizer)
        if training_template:
            tokenizer.chat_template = training_template
            print("Using TRL training-compatible chat template for assistant masking.")
    except Exception as exc:
        print(f"Warning: could not pre-patch the chat template: {exc}")

    quant_config = None
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["eval"] = args.eval_file

    raw_ds = load_dataset("json", data_files=data_files)

    train_ds = raw_ds["train"]
    eval_ds = raw_ds.get("eval")

    # IMPORTANT: keep the dataset conversational.
    # Do NOT convert messages -> text. TRL needs the messages structure to
    # compute assistant-only loss.
    if "messages" not in train_ds.column_names:
        raise ValueError(
            "Training dataset must contain a 'messages' column for "
            "assistant_only_loss=True."
        )

    # Preflight token statistics before SFTTrainer preprocessing.
    train_report = analyze_token_lengths(
        train_ds, tokenizer, args.max_seq_len, "train"
    )
    print_token_report(train_report)

    if train_report["examples_without_assistant_mask"] > 0:
        raise RuntimeError(
            "Assistant-only loss cannot be safely verified: "
            f"{train_report['examples_without_assistant_mask']} training examples "
            "did not return an assistant token mask. Update TRL/Transformers or "
            "verify the model's training chat template before training."
        )

    eval_report = None
    if eval_ds is not None:
        if "messages" not in eval_ds.column_names:
            raise ValueError(
                "Evaluation dataset must contain a 'messages' column."
            )
        eval_report = analyze_token_lengths(
            eval_ds, tokenizer, args.max_seq_len, "eval"
        )
        print_token_report(eval_report)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / "token_stats.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": args.base_model,
                "train": train_report,
                "eval": eval_report,
            },
            f,
            indent=2,
        )

    # Critical change: assistant_only_loss=True.
    # For conversational datasets, TRL computes loss only on assistant messages.
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        save_total_limit=3,
        fp16=False,
        bf16=False,
        max_length=args.max_seq_len,
        assistant_only_loss=True,
        packing=False,
        optim="paged_adamw_8bit",
        report_to="none",
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(Path(args.output_dir) / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    if args.push_to_hub:
        trainer.push_to_hub()
        print(f"Pushed adapter to https://huggingface.co/{args.push_to_hub}")

    print(f"Done. LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
