#!/usr/bin/env python3
"""Run this in your training environment (has network access to HF) before retraining
to confirm how many examples were truncated at the old max_seq_len=1024, and to pick
a safe new value.

Works regardless of your current working directory — data path is resolved relative
to this script's location (../data/ from scripts/).
"""
import json
from pathlib import Path
from transformers import AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-8B"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

for split in ["train", "dev"]:
    path = DATA_DIR / f"{split}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lengths = []
    for line in lines:
        ex = json.loads(line)
        text = tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        n = len(tokenizer(text)["input_ids"])
        lengths.append(n)
    over_1024 = sum(1 for n in lengths if n > 1024)
    print(f"{split}: n={len(lengths)}  min={min(lengths)}  max={max(lengths)}  "
          f"over_1024={over_1024} ({100*over_1024/len(lengths):.0f}%)")
