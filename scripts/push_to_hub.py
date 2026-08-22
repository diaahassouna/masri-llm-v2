#!/usr/bin/env python3
"""
push_to_hub.py — Merge a trained LoRA adapter into the base model and push
the merged model (and/or the dataset) to the Hugging Face Hub.

Usage:
  huggingface-cli login   # once, paste your write token

  # Push the merged full model:
  python3 push_to_hub.py \
      --base_model Qwen/Qwen2.5-7B-Instruct \
      --adapter_dir ../out/masri-lora \
      --repo_id yourname/masri-qwen2.5-7b \
      --merge

  # Or push just the raw dataset:
  python3 push_to_hub.py --dataset_dir ../data --dataset_repo_id yourname/masri-dataset
"""
import argparse

from huggingface_hub import HfApi


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter_dir", default="../out/masri-lora")
    p.add_argument("--repo_id", default=None, help="e.g. yourname/masri-qwen2.5-7b")
    p.add_argument("--merge", action="store_true", help="Merge LoRA weights into base before push")
    p.add_argument("--dataset_dir", default=None, help="Path to data/ folder to push as a dataset repo")
    p.add_argument("--dataset_repo_id", default=None, help="e.g. yourname/masri-dataset")
    return p.parse_args()


def push_model(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)

    if args.merge:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, args.adapter_dir)
        model = model.merge_and_unload()
        model.push_to_hub(args.repo_id)
    else:
        # push the adapter only (much smaller upload, requires PEFT to load later)
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, args.adapter_dir)
        model.push_to_hub(args.repo_id)

    tokenizer.push_to_hub(args.repo_id)
    print(f"Pushed to https://huggingface.co/{args.repo_id}")


def push_dataset(args):
    api = HfApi()
    api.create_repo(args.dataset_repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=args.dataset_dir,
        repo_id=args.dataset_repo_id,
        repo_type="dataset",
    )
    print(f"Pushed dataset to https://huggingface.co/datasets/{args.dataset_repo_id}")


if __name__ == "__main__":
    args = parse_args()
    if args.repo_id:
        push_model(args)
    if args.dataset_repo_id:
        push_dataset(args)
