#!/usr/bin/env python3
"""
augment_from_corpus.py — Convert real Egyptian Arabic corpus sentences
(from fetch_corpus.py) into Masri Tier 2 training pairs, using Gemini.

This differs from augment_with_gemini.py in a key way: that script asks
Gemini to INVENT Arabic sentences from scratch. This script instead feeds
it REAL sentences pulled from arz.wikipedia, and asks Gemini to:

  1. Produce the correct Masri Tier 2 transliteration (applying every rule
     in the system prompt exactly, and normalizing toward natural Cairene
     phrasing if the source sentence leans MSA-ish).
  2. Produce a natural FRANCO/Arabizi rendering of the same sentence.

Point (2) exists specifically to fix a measured gap: testing showed digits
5 (خ) and 8 (غ) are almost completely absent from the existing Franco
training data (7 occurrences and 0 occurrences respectively, out of 5,143
examples), while 2/3/7 are extremely common — because the instruction-only
generator never produced real Franco input text at all. This script's
prompt explicitly instructs the model to use 2/3/5/7/8 wherever the
corresponding phoneme genuinely occurs, not just the easy/common ones.

For every corpus sentence, this writes out TWO training examples — one
with an Arabic-script instruction wrapper, one with a Franco instruction
wrapper — rotating across the instruction phrasings already used in
train.jsonl, so instruction-wording variety doesn't collapse to one fixed
string.

Requires: pip install google-genai
          export GEMINI_API_KEY=...

Usage:
  python3 augment_from_corpus.py \
      --corpus ../data/corpus_sentences.jsonl \
      --out ../data/train_corpus_augmented.jsonl \
      --n 3000
"""
import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from google import genai

SRC = Path(__file__).parent.parent / "source"
DATA = Path(__file__).parent.parent / "data"

SYSTEM_PROMPT = (SRC / "masri_tier2_system_prompt.md").read_text(encoding="utf-8")

# Instruction phrasings actually observed in the existing train.jsonl (kept
# consistent so this augmented data doesn't introduce yet another unseen
# template) — one Arabic-script set, one Franco set.
ARABIC_INSTRUCTION_TEMPLATES = [
    "حوّل الجملة دي للمصرية (Tier 2): {text}",
    "اكتبلي الجملة دي بالأبجدية المصرية: {text}",
    "حوّل دي للمصرية: {text}",
]
FRANCO_INSTRUCTION_TEMPLATES = [
    "7awel el gomla dee lel masreya (Tier 2): {text}",
    "eh el masri el sa7 lel gomla dee: {text}",
    "7awel dee lel masreya: {text}",
]

GEN_INSTRUCTIONS = """You are helping build a fine-tuning dataset for the Masri Tier 2 writing \
system described in the system prompt above, created by Diaa Hassouna (CC BY 4.0).

Below are {batch_size} REAL sentences pulled from Egyptian Arabic Wikipedia. For EACH numbered \
sentence, produce a JSON object with three fields:

1. "arabic": the sentence, lightly normalized toward natural spoken Cairene phrasing if the \
   original leans formal/MSA (don't rewrite the meaning, just make the wording something a Cairene \
   speaker would actually say out loud).
2. "masri": the correct Masri Tier 2 transliteration of that normalized sentence, following every \
   rule in the system prompt exactly.
3. "franco": a natural Arabizi/Franco typing of that SAME sentence, the way a Cairene would casually \
   type it on their phone. IMPORTANT: use digit-substitutes 2 (hamza/ق), 3 (ع), 5 (خ), 7 (ح), and 8 \
   (غ) EVERY time the corresponding sound genuinely occurs in the sentence — do not avoid 5 or 8 just \
   because they're less common in casual typing; if the sentence contains a خ or غ sound anywhere, the \
   franco field MUST represent it with 5 or 8 respectively, not with a plain Latin approximation.

Sentences:
{numbered_sentences}

Return ONLY a JSON array of {batch_size} objects in that exact shape, no prose, no markdown fences, \
in the same order as the input sentences.
"""


def load_existing_pairs():
    """Load existing train.jsonl for optional dedup checking (best-effort, not required)."""
    seen = set()
    train_path = DATA / "train.jsonl"
    if train_path.exists():
        for line in train_path.read_text(encoding="utf-8").splitlines():
            ex = json.loads(line)
            for m in ex["messages"]:
                if m["role"] == "assistant":
                    seen.add(m["content"].strip())
    return seen


def flag_for_review(arabic, masri, franco):
    """Heuristic flags so a human reviewer can prioritize which rows to check first —
    rule-sensitive constructs are exactly where an LLM generator is most likely to be
    confidently wrong (per augment_with_gemini.py's own warning)."""
    flags = []
    if re.search(r"(?<![0-9])5(?![0-9])", franco):
        flags.append("digit_5_khaa")
    if re.search(r"(?<![0-9])8(?![0-9])", franco):
        flags.append("digit_8_ghain")
    if "ث" in arabic:
        flags.append("tha_merger")
    if re.search(r"\b(el|ال)[a-zA-Zء-ي]", masri.replace(" ", "")) and "-" in franco:
        flags.append("possible_assimilated_article")
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DATA / "corpus_sentences.jsonl"),
                     help="Output of fetch_corpus.py")
    ap.add_argument("--n", type=int, default=3000, help="how many corpus sentences to process")
    ap.add_argument("--batch_size", type=int, default=10,
                     help="Keep modest — free-tier RPM limits mean big batches don't help.")
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--out", default=str(DATA / "train_corpus_augmented.jsonl"))
    ap.add_argument("--sleep_seconds", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set GEMINI_API_KEY first. Get a free key (no card needed) at "
            "aistudio.google.com/apikey"
        )
    client = genai.Client(api_key=api_key)
    random.seed(args.seed)

    existing_outputs = load_existing_pairs()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise SystemExit(f"Corpus file not found: {corpus_path}. Run fetch_corpus.py first.")
    corpus_sentences = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            corpus_sentences.append(json.loads(line)["arabic"])
    corpus_sentences = corpus_sentences[:args.n]

    all_rows = []
    review_flags_count = {}
    n_batches = (len(corpus_sentences) + args.batch_size - 1) // args.batch_size

    for i in range(n_batches):
        batch_sentences = corpus_sentences[i * args.batch_size:(i + 1) * args.batch_size]
        numbered = "\n".join(f"{j+1}. {s}" for j, s in enumerate(batch_sentences))
        prompt = SYSTEM_PROMPT + "\n\n" + GEN_INSTRUCTIONS.format(
            batch_size=len(batch_sentences), numbered_sentences=numbered
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

        for row in batch:
            if row.get("masri", "").strip() in existing_outputs:
                continue  # skip near-duplicates of what's already in train.jsonl
            flags = flag_for_review(row.get("arabic", ""), row.get("masri", ""), row.get("franco", ""))
            row["review_flags"] = flags
            for f in flags:
                review_flags_count[f] = review_flags_count.get(f, 0) + 1
            all_rows.append(row)

        print(f"[batch {i+1}/{n_batches}] +{len(batch)} examples (total {len(all_rows)})")
        time.sleep(args.sleep_seconds)

    with open(args.out, "w", encoding="utf-8") as f:
        for row in all_rows:
            arabic, masri, franco = row["arabic"], row["masri"], row.get("franco", "")

            arabic_template = random.choice(ARABIC_INSTRUCTION_TEMPLATES)
            arabic_example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": arabic_template.format(text=arabic)},
                    {"role": "assistant", "content": masri},
                ],
                "category": "corpus_augmented_arabic",
                "review_flags": row["review_flags"],
            }
            f.write(json.dumps(arabic_example, ensure_ascii=False) + "\n")

            if franco:
                franco_template = random.choice(FRANCO_INSTRUCTION_TEMPLATES)
                franco_example = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": franco_template.format(text=franco)},
                        {"role": "assistant", "content": masri},
                    ],
                    "category": "corpus_augmented_franco",
                    "review_flags": row["review_flags"],
                }
                f.write(json.dumps(franco_example, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_rows)} source pairs (~{len(all_rows) * 2} training examples "
          f"counting both Arabic and Franco instruction variants) to {args.out}")
    print("\nReview flags summary (prioritize these rows before merging into train.jsonl):")
    for flag, count in sorted(review_flags_count.items(), key=lambda x: -x[1]):
        print(f"  {flag}: {count}")
    print("\nAs with augment_with_gemini.py: REVIEW BEFORE TRAINING, especially flagged rows —")
    print("a generator model can produce confidently wrong spellings for memorized-exception rules")
    print("(tha_merger, gemination) and for the newly-targeted digit_5/digit_8 substitutions, since")
    print("those were nearly unseen in prior training data and the generator has little to imitate.")


if __name__ == "__main__":
    main()
