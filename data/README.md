---
license: cc-by-4.0
language:
- ar
tags:
- egyptian-arabic
- constructed-script
- transliteration
- masri
pretty_name: Masri Tier 2 Writing System Dataset
---

# Masri Tier 2 Dataset

Training data for **El Abgadeyya El Maṣreyya (Masri Writing System)**, a constructed
orthography for Urban Cairene Egyptian Arabic created by Diaa Hassouna
([github.com/diaahassouna/masri-writing](https://github.com/diaahassouna/masri-writing)),
CC BY 4.0.

## Files

| File | Purpose |
|---|---|
| `train.jsonl` | SFT training examples, chat format (`system`/`user`/`assistant`) |
| `dev.jsonl` | Small in-training dev slice (same distribution as train, held out of it) |
| `eval_held_out.jsonl` | **Never trained on.** 1:1 derived from `masri_tier2_eval_set.json`, with `expected` + `accepted_variants` + `tests_for` per item, for scoring |
| `system_prompt.txt` | The grounding system prompt used for every example (from `masri_tier2_system_prompt.md`) |
| `dataset_stats.json` | Category counts |

## Format

```json
{"messages": [
  {"role": "system", "content": "<Masri Tier 2 grounding prompt>"},
  {"role": "user", "content": "حوّل الكلمة دي للمصرية (Tier 2): عارف"},
  {"role": "assistant", "content": "Ⲵaref"}
]}
```

## Important limitation

`train.jsonl` currently has ~120 examples, generated deterministically from
`alphabet.json` and `tier2-rules.json` (letter inventory drills, rule
explanations, the ayin examples, standardized word list, loanword policy
examples, and the stress-test sentences). **This is enough to sanity-check a
training pipeline but not enough on its own for a model to reliably learn
Masri's exception-heavy rules** (word-final gemination is memorized per word,
ث-merger direction is memorized per word, irregular loanword plurals are
memorized per word). See `scripts/augment_with_claude.py` for a way to scale
this up before a real training run, and review augmented data before using it
— generated data can look fluent while getting a memorized exception wrong.

## Eval categories (from `masri_tier2_eval_set.json`)

`formatting`, `gemination`, `definite_article`, `glottal_stop`, `ayin`,
`q_hamza_merger`, `p_b_v_f`, `short_vowels`, `loanword_english_kept`,
`loanword_egyptianized`, `loanword_irregular_plural`, `tha_merger`, `homograph`

## License

CC BY 4.0, matching the source Masri project. Attribute Diaa Hassouna and the
Masri Writing System project.
