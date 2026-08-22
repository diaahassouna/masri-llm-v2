#!/usr/bin/env python3
"""
augment_with_gemini.py — Free-tier alternative to augment_with_claude.py.

Uses Google's Gemini API (free tier via AI Studio, no credit card required)
to generate more (Arabic -> Masri Tier 2) training pairs. Same purpose as
augment_with_claude.py: your hand-curated 119 examples aren't enough for a
model to reliably learn Masri's memorized exceptions.

Get a free key (no credit card): aistudio.google.com/apikey

Free tier is rate-limited (roughly 10-15 requests/minute on Flash-class
models as of 2026), so this script runs in small batches with a short pause
between calls rather than firing everything at once — expect it to take a
few minutes for a few hundred examples, not seconds.

Requires: pip install google-genai
          export GEMINI_API_KEY=...   (or set it directly in Colab, see GUIDE.md)

Usage:
  python3 augment_with_gemini.py --n 400 --out ../data/train_augmented.jsonl
"""
import argparse
import json
import os
import time
from pathlib import Path

from google import genai

SRC = Path(__file__).parent.parent / "source"
DATA = Path(__file__).parent.parent / "data"

SYSTEM_PROMPT = (SRC / "masri_tier2_system_prompt.md").read_text(encoding="utf-8")

GEN_INSTRUCTIONS = """You are helping build a fine-tuning dataset for the Masri Tier 2 writing \
system described in the system prompt above, created by Diaa Hassouna (CC BY 4.0).

Generate {batch_size} NEW, natural Urban Cairene Egyptian Arabic sentences (in Arabic script), \
each paired with its correct Masri Tier 2 transliteration, following every rule in the system \
prompt exactly. Vary topic, length (mix short phrases and longer sentences), and which rules \
each example exercises — deliberately include some sentences that combine multiple rules \
(gemination + ayin + definite article in one sentence, etc.), matching the difficulty style of \
the "stacks the hardest combination" test items you'd expect in an eval set.

Do NOT reuse any of these already-covered example words/sentences verbatim: {avoid_list}

Return ONLY a JSON array, no prose, no markdown fences, in this exact shape:
[{{"arabic": "...", "masri": "...", "category": "gemination|definite_article|glottal_stop|ayin|q_hamza_merger|p_b_v_f|short_vowels|loanword|tha_merger|homograph|conversational"}}, ...]
"""


def load_avoid_list():
    avoid = []
    if (DATA / "train.jsonl").exists():
        for line in (DATA / "train.jsonl").read_text(encoding="utf-8").splitlines():
            ex = json.loads(line)
            for m in ex["messages"]:
                if m["role"] == "assistant":
                    avoid.append(m["content"][:40])
    return avoid[:60]  # keep prompt short


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="total examples to generate")
    ap.add_argument("--batch_size", type=int, default=20,
                     help="Keep this modest — free-tier RPM limits mean big batches don't help.")
    ap.add_argument("--model", default="gemini-3.6-flash",
                     help="Flash-class models are on the free tier; Pro models are not (as of 2026). "
                          "If this 404s, list your account's available models (see GUIDE.md) and pick one.")
    ap.add_argument("--out", default=str(DATA / "train_augmented.jsonl"))
    ap.add_argument("--sleep_seconds", type=float, default=5.0,
                     help="Pause between requests to stay under free-tier RPM limits.")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set GEMINI_API_KEY first. Get a free key (no card needed) at "
            "aistudio.google.com/apikey, then: import os; os.environ['GEMINI_API_KEY']='...'"
        )
    client = genai.Client(api_key=api_key)
    avoid_list = load_avoid_list()

    all_rows = []
    n_batches = (args.n + args.batch_size - 1) // args.batch_size
    for i in range(n_batches):
        prompt = SYSTEM_PROMPT + "\n\n" + GEN_INSTRUCTIONS.format(
            batch_size=args.batch_size, avoid_list=avoid_list
        )
        try:
            resp = client.models.generate_content(model=args.model, contents=prompt)
            raw = (resp.text or "").strip().strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            batch = json.loads(raw)
        except Exception as e:
            print(f"[batch {i+1}/{n_batches}] failed ({e}), skipping")
            time.sleep(args.sleep_seconds)
            continue

        all_rows.extend(batch)
        print(f"[batch {i+1}/{n_batches}] +{len(batch)} examples (total {len(all_rows)})")
        time.sleep(args.sleep_seconds)  # stay under free-tier RPM limits

    with open(args.out, "w", encoding="utf-8") as f:
        for row in all_rows:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"حوّل الجملة دي للمصرية (Tier 2): {row['arabic']}"},
                {"role": "assistant", "content": row["masri"]},
            ]
            f.write(json.dumps({"messages": messages, "category": row.get("category", "augmented")},
                                ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_rows)} augmented examples to {args.out}")
    print("REVIEW THESE BEFORE TRAINING — especially tha_merger, gemination, and loanword rows,")
    print("since those rules are memorized closed lists, not derivable patterns, and a generator")
    print("model can confidently produce a plausible-looking but wrong spelling.")


if __name__ == "__main__":
    main()
