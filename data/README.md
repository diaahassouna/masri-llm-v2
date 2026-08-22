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
| `train.jsonl` | SFT training examples, chat format (`system`/`user`/`assistant`) — **1,389 examples** (662 Arabic-script, 725 Franco, 2 mixed Arabic+English) |
| `dev.jsonl` | Small in-training dev slice (20 examples, same distribution as train, held out of it — confirmed no overlap with `train.jsonl`) |
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

## How `train.jsonl` is built

`scripts/build_dataset.py` turns the deterministic sources
(`alphabet.json`, `tier2-rules.json`, `mixed_script_examples.json`,
`numeral_disambiguation_examples.json`) into training rows — letter
inventory drills, rule explanations, ayin examples, the standardized word
list, loanword policy examples, stress-test sentences, plus each Arabic
example's Franco/Arabizi counterpart in several distinct spellings
(`franco_variation.py`, so و/غ/ي etc. show up as more than one Franco
convention, not a single fixed mapping).

On top of that, `train.jsonl` folds in ~1,060 template-generated sentences:
59 common nouns (a mix of the project's own vetted loanwords and plain,
morphologically-regular Egyptian Arabic words with no memorized exceptions)
dropped into 9 gender-neutral sentence frames ("I want the X", "where's the
X?", "I don't have the X", etc.), each rendered in Arabic script, one Franco
spelling, and the corresponding canonical Masri Tier 2 output. This is what
pushes the category breakdown (see `dataset_stats.json`) from ~20 fixed
sources up to 1,389 examples without inventing spellings for anything on the
system prompt's memorized/closed-list rules.

## Important limitation

**This still isn't the same as a large, linguistically diverse corpus.**
The template sentences are deliberately simple and avoid grammatical-gender
adjective agreement, and none of them touch the genuinely hard,
exception-heavy categories — word-final gemination is memorized per word,
ث-merger direction is memorized per word, irregular loanword plurals are
memorized per word (`numeral_disambiguation`, `mixed_script`, and a handful
of loanword/ayin categories above are the only rows that exercise those at
all, and they're small: single digits per category). For a model to reliably
learn those, run `scripts/augment_with_claude.py` to generate more natural,
varied sentences grounded in the rule set — and review the tha_merger,
gemination, and loanword_irregular_plural rows by hand before training,
since generated data can look fluent while getting a memorized exception
wrong.

## Eval categories (from `masri_tier2_eval_set.json`)

`formatting`, `gemination`, `definite_article`, `glottal_stop`, `ayin`,
`q_hamza_merger`, `p_b_v_f`, `short_vowels`, `loanword_english_kept`,
`loanword_egyptianized`, `loanword_irregular_plural`, `tha_merger`, `homograph`

## License

CC BY 4.0, matching the source Masri project. Attribute Diaa Hassouna and the
Masri Writing System project.
