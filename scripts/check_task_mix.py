#!/usr/bin/env python3
"""Check what fraction of train.jsonl is actual conversion examples vs meta Q&A
about the writing system's rules. Run from anywhere; resolves data/ relative to
this script's own location."""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

for split in ["train", "dev"]:
    path = DATA_DIR / f"{split}.jsonl"
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    meta_like, convert_like = 0, 0
    for ex in lines:
        user_msg = next(m["content"] for m in ex["messages"] if m["role"] == "user")
        stripped = user_msg.strip()
        if stripped.endswith(("؟", "?")) or "بيمثل" in stripped or "إيه معنى" in stripped:
            meta_like += 1
        else:
            convert_like += 1
    total = len(lines)
    print(f"{split}: total={total}  meta/explanation-like={meta_like} "
          f"({100*meta_like/total:.0f}%)  conversion-like={convert_like} "
          f"({100*convert_like/total:.0f}%)")
