# Training an AI to write Masri — step-by-step Hugging Face guide

This folder is a complete pipeline: it turns your four source files
(`alphabet.json`, `tier2-rules.json`, `masri_tier2_system_prompt.md`,
`masri_tier2_eval_set.json`) into a Hugging Face dataset, fine-tunes an open
model on it, and scores the result against your own eval set.

```
masri-llm/
├── source/                  your 4 original files (copied in)
├── data/                    generated: train.jsonl, dev.jsonl, eval_held_out.jsonl, system_prompt.txt
├── scripts/
│   ├── build_dataset.py     source files -> train/eval JSONL  (already run once, see below)
│   ├── augment_with_claude.py   optional: scale train.jsonl up with the Claude API
│   ├── train_lora.py        QLoRA fine-tune
│   ├── evaluate.py          score against eval_held_out.jsonl, per-category
│   ├── push_to_hub.py       merge + upload to your HF account
│   └── inference.py         quick interactive chat with the result
├── requirements.txt
└── GUIDE.md                 this file
```

`data/` already contains 119 training examples and a 53-item held-out eval
set generated from your files. Read the **"the honest caveat"** section below
before you spend GPU hours on this — it matters for a project like Masri.

---

## 0. The honest caveat

119 examples is enough to prove the pipeline works, not enough for a model to
reliably learn Masri's exception-heavy spelling — things like which direction
ث merges (memorized per word), which words geminate word-finally (memorized,
not derivable), or which loanwords take irregular Arabic-style plurals. Those
are exactly the categories your own `masri_tier2_eval_set.json` is designed
to catch.

Two honest paths forward, and you can do both:

1. **Scale the data first.** Run `scripts/augment_with_claude.py` to
   generate a few hundred more (Arabic → Masri) pairs grounded in your rule
   set, review a sample by hand — especially the memorized-exception
   categories — then fine-tune on the combined set. This is the path most
   likely to produce a model that actually holds up against your eval set.
2. **Skip fine-tuning, use retrieval/prompting instead.** For a system this
   rule-based, a strong base model given `masri_tier2_system_prompt.md` as a
   system prompt (which is already written as excellent few-shot grounding)
   may outperform a LoRA trained on <1,000 examples, at zero training cost.
   Worth benchmarking both before committing GPU time.

---

## 1. Create a Hugging Face account and get a token

1. Sign up at [huggingface.co/join](https://huggingface.co/join) (free).
2. Go to **Settings → Access Tokens** → **New token** → role **Write**.
3. Copy it. On any machine you'll train from:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   # paste the token when prompted
   ```

## 2. Pick where you'll run training

You need a GPU. Options, cheapest first:

- **Google Colab (free tier)** — a T4 GPU (16GB), enough for a 7-8B model in
  4-bit QLoRA. Good for this project's scale. Upload this whole folder or
  `git clone` it into a Colab notebook, `pip install -r requirements.txt`,
  then run the scripts as shown below.
- **Hugging Face Spaces / Jobs** with a GPU (paid, per-hour) — good if you
  want it running unattended, or if Colab keeps disconnecting.
- **Your own GPU** if you have one with ≥16GB VRAM.

## 3. Push the dataset to the Hub (optional but recommended)

This gives you a permanent, versioned, shareable copy — and lets you load it
with `datasets.load_dataset()` from any machine instead of re-uploading
files.

```bash
cd scripts
python3 push_to_hub.py \
  --dataset_dir ../data \
  --dataset_repo_id YOURNAME/masri-tier2-dataset
```

Or via the web UI: **huggingface.co/new-dataset** → drag in the `data/`
folder's contents.

## 4. Choose a base model

Masri's Tier 2 script mixes Latin, Coptic (Ϩ ϣ Ⲵ), Greek (Θ Ɣ), and Latin
Extended (Ð Ṣ Ḍ Ṭ Ẓ Ɐ) code points on top of Egyptian Arabic content and
grammar. That means the base model needs **two** things at once: real
Egyptian-Arabic fluency, and a tokenizer that doesn't shred the Coptic/Greek
letters into meaningless byte fragments.

| Model | Why | Size / hardware |
|---|---|---|
| **Qwen/Qwen3-8B** *(default in `train_lora.py`)* | Apache 2.0, strong multilingual + Arabic support, broad-coverage tokenizer, huge fine-tuning ecosystem/documentation | ~16GB VRAM in 4-bit — fits a free Colab T4 |
| **tiiuae/Falcon3-Arabic-7B-Instruct** | Arabic-native (trained specifically for Arabic), worth A/B testing against Qwen3-8B on your eval set | similar footprint |
| **Qwen/Qwen3-4B** | Same family, smaller — use if Colab T4 is tight or training is slow | ~8GB in 4-bit |
| **Qwen/Qwen3-235B-A22B** or **Jais 30B** (Core42) | If you later want a stronger ceiling and have access to bigger/multi-GPU hardware or a hosted training job | not free-tier |

Before committing, run a quick manual check: paste
`masri_tier2_system_prompt.md` as a system prompt into the base model (via
its HF Space demo or API) and try 3-4 items from `eval_held_out.jsonl`
un-fine-tuned. That tells you your fine-tuning starting point and whether the
tokenizer is mangling the Coptic letters before you spend any GPU time.

## 5. (Optional, recommended) Scale up the training data

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cd scripts
python3 augment_with_claude.py --n 400 --out ../data/train_augmented.jsonl
# review the output by hand, then:
cat ../data/train_augmented.jsonl >> ../data/train.jsonl
```

## 6. Fine-tune

```bash
cd scripts
python3 train_lora.py \
  --base_model Qwen/Qwen3-8B \
  --train_file ../data/train.jsonl \
  --eval_file ../data/dev.jsonl \
  --output_dir ../out/masri-qwen3-8b-lora \
  --epochs 3
```

This is QLoRA (4-bit base + LoRA adapters) via `transformers` + `peft` +
`trl`, so it trains a small adapter (tens of MB), not the full model. Expect
somewhere in the range of 20-60 minutes on a T4 for this data size —
watch the `eval_loss` printed each epoch; if it's rising while `train_loss`
falls, you're overfitting the small dataset and should stop earlier
(`--epochs 1` or `2`) or get more data (step 5).

## 7. Evaluate against your own eval set

```bash
python3 evaluate.py \
  --model ../out/masri-qwen3-8b-lora \
  --adapter_of Qwen/Qwen3-8B
```

This prints a per-category pass table (EXACT / ACCEPTABLE / FAIL / PASS%) —
exactly the breakdown `masri_tier2_eval_set.json`'s own `how_to_use` notes
recommend, so you can see e.g. "tha_merger is failing consistently → add
more `tha_merger`-category training data" rather than one opaque overall
score. Full per-item outputs land in `eval_results.json`.

## 8. Push the trained model to the Hub

```bash
python3 push_to_hub.py \
  --base_model Qwen/Qwen3-8B \
  --adapter_dir ../out/masri-qwen3-8b-lora \
  --repo_id YOURNAME/masri-qwen3-8b \
  --merge
```

`--merge` bakes the LoRA weights into the base model and pushes a
self-contained full model (bigger upload, no `peft` needed to load it later).
Drop `--merge` to push just the adapter (much smaller, ~50-200MB, but
whoever uses it needs the same base model + `peft` installed).

Once pushed, your model has a page at
`huggingface.co/YOURNAME/masri-qwen3-8b` with an inference widget people can
try in the browser, no code required.

## 9. Talk to it

```bash
python3 inference.py --model YOURNAME/masri-qwen3-8b
```

or, for a raw un-merged adapter:

```bash
python3 inference.py --model ../out/masri-qwen3-8b-lora --adapter_of Qwen/Qwen3-8B
```

---

## Iterating

The eval categories map directly onto specific rules in
`masri_tier2_system_prompt.md`. If `evaluate.py` shows a category
consistently failing:

1. Open `eval_results.json`, find that category's rows, read `model_output`
   vs `expected` vs `tests_for`.
2. Write more `build_dataset.py`-style or `augment_with_claude.py`-style
   examples specifically targeting that rule.
3. Re-train, re-evaluate.

This loop is the actual product here — the eval set your earlier work
already built is what makes it possible to know whether a training run
helped or not, rather than eyeballing outputs.
