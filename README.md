# Masri-LLM

A many-to-one orthographic canonicalization system for Egyptian Arabic:
Arabic script, Franco/Arabizi, non-canonical Masri, and Arabic-English
code-switched input all normalize to a single canonical **Masri Tier 2**
orthography — a 35-letter system using restored Coptic, Greek, and
Latin-extended characters to represent Egyptian Arabic phonology.

Masri-LLM is not a transliteration function (character-for-character
substitution) or a grammar corrector. It recovers the underlying Egyptian
Arabic word/meaning from whatever script or spelling it was typed in, then
renders that meaning in canonical Masri — preserving word order, register,
and code-switching intent rather than rewriting them.

## Repository layout

```
source/                     — the specification (source of truth — read canonicalization_ontology.json first)
  canonicalization_ontology.json — formal token/lexical-status/normalization-operation
                                    taxonomy + decision procedures. Read this before adding
                                    new categories anywhere else in the repo.
  masri_tier2_system_prompt.md   — the grounding doc: rules (incl. Rule 0, digit policy),
                                    exceptions, register
  alphabet.json                  — the 35-letter Tier 2 alphabet + borrowed orthography
  tier2-rules.json               — spelling rules, loanword examples, standardized words
  lexicon.json                   — hand-maintained dictionary (Part-5 schema: canonical_masri,
                                    arabic_variants, loanword_status, confidence, etc.) —
                                    THE source of truth for lexicon entries; edit this directly
  masri_tier2_eval_set.json      — hand-curated eval items (Arabic script + Franco)
  mixed_script_examples.json     — code-switch / mixed Arabic+English gold examples
  numeral_disambiguation_examples.json — Arabizi numeral vs. real-number contrast pairs
  changelog.md                   — proposed/adopted rule changes, with justification
  lexicon.sqlite                 — GENERATED, gitignored. Fast-lookup index built from
                                    lexicon.json by scripts/build_lexicon.py. Build on demand,
                                    don't hand-edit, don't commit.

scripts/
  build_dataset.py            — generates data/train.jsonl, dev.jsonl, dataset_stats.json
                                  from source/. Enforces Rule 0 (see below) before writing.
  franco_variation.py         — controlled multi-variant Arabic->Franco/Arabizi rendering
                                  engine used by build_dataset.py (see its own docstring)
  build_lexicon.py            — generates source/lexicon.sqlite from source/lexicon.json
                                  (optional/on-demand — see "Lexicon workflow" below)
  augment_with_claude.py      — LLM-assisted dataset scaling (arabic + franco + masri triples)
  augment_with_gemini.py
  validate_output.py          — deterministic, TOKEN-AWARE structural checker for Masri
                                  output (script leakage, invalid chars, closed-list
                                  mismatches, Rule 0 digit-leak detection that doesn't
                                  false-flag genuine English code-switch like "Web3"/"G7")
  evaluate.py                 — scores a trained model against the eval set, split by
                                  category and input_type (arabic_script vs. franco)
  train_lora.py, inference.py, push_to_hub.py
  check_output.py, check_task_mix.py, check_truncation.py, eval_score.py

data/                        — generated training/eval data (regenerate with build_dataset.py,
                                 don't hand-edit)

GUIDE.md                     — practical notes on running the pipeline
requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# regenerate training/eval data from source/ (Rule 0 is enforced automatically;
# the build fails loudly if a digit-as-phoneme leaks into a Masri target string)
python3 scripts/build_dataset.py

# fine-tune (see train_lora.py for config)
python3 scripts/train_lora.py

# evaluate, with a breakdown by input type
python3 scripts/evaluate.py
python3 scripts/evaluate.py --filter_input_type franco

# check a single output string against the deterministic rules
python3 scripts/validate_output.py --text "El naharda el gaw Ϩelw."
```

## Lexicon workflow

`source/lexicon.json` is what you edit. It's a plain, readable JSON array —
open it, add or correct an entry, save. `source/lexicon.sqlite` is a
generated index for fast inference-time lookup; build it only when you
actually need fast lookup (e.g. wiring up an inference-time dictionary
pre-pass), not as part of routine lexicon editing:

```bash
python3 scripts/build_lexicon.py   # regenerates source/lexicon.sqlite from lexicon.json
```

If `lexicon.sqlite` and `lexicon.json` ever disagree, `lexicon.json` wins —
regenerate the sqlite, don't hand-patch it (it's gitignored for this reason).

## Rule 0 — digits never survive as letters in output

Stated formally in `masri_tier2_system_prompt.md`'s "Rule 0" section and in
`source/canonicalization_ontology.json`'s `rule_zero` block. Digits may
appear in Franco/Arabizi *input* as phoneme substitutes (2/3/5/7/8) — that's
expected and `franco_variation.py` generates that naturally. They must never
survive into canonical Masri *output* except when they represent a genuine
number. Two mechanical checks enforce this:
- `build_dataset.py` refuses to write `data/` if any assistant/target string
  contains a digit-as-letter pattern (`_assert_rule_zero`).
- `validate_output.py` flags the same pattern in model output at
  inference/eval time, token-aware enough not to flag genuine code-switch
  digits (Web3, G7) as Rule 0 violations.

## Before generating much more training data

Read `source/canonicalization_ontology.json` first. It's the formal
definition of every token class, lexical-status class, and normalization
operation the rest of the repo uses informally — the category names in
`build_dataset.py`, `augment_with_claude.py`'s generation prompt, and
`lexicon.json`'s `loanword_status` field should all trace back to an id
defined there. Introducing a new ad-hoc category name in a script or a
generated data file without adding it to the ontology first is exactly how
`dataset_stats.json` drifted out of sync with reality before this repo was
cleaned up — don't repeat that.

## A note on `data/` and generated files

`data/*.jsonl`, `data/dataset_stats.json`, and `data/system_prompt.txt` are
all build artifacts — always reproducible by re-running the relevant script
against `source/`. Treat `source/` as the only hand-edited ground truth; if a
generated file and `source/` ever disagree, regenerate rather than
hand-patching the generated file.

## License

CC BY 4.0 — see [LICENSE](LICENSE). Same license as the Masri Writing System
Development Framework doc.
