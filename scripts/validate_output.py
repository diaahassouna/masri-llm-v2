#!/usr/bin/env python3
"""
validate_output.py — Deterministic post-generation validator for Masri Tier 2
output. This did not exist anywhere in the repo before the Aug 2026 audit;
evaluate.py's exact/substring string match against a fixed `expected` answer
is the only current signal, which can't catch a fluent-looking WRONG answer
that isn't already in the eval set, and gives no structural diagnosis at all.

This is not a replacement for evaluate.py's correctness scoring — it's a
cheap, rule-based backstop that runs on ANY output (even unlabeled/production
output with no `expected` to compare against) and catches the specific
failure modes the spec (masri_tier2_system_prompt.md, tier2-rules.json) rules
out categorically:
  - leftover Arabic-script characters (defeats the whole point of
    canonicalization)
  - Arabizi numerals surviving as letters (2/3/5/7 standing in for
    hamza/ain/kha/ha instead of being converted) — Rule 0 in
    masri_tier2_system_prompt.md
  - characters outside the 35-letter Tier 2 alphabet + borrowed C/J
  - "el" fused or hyphenated to the following word (rule 2)
  - missing sentence-initial capitalization (rule 8)
  - closed-list word mismatches against a small hand-curated lexicon of
    memorized exceptions (tha_merger / word-final-gemination / irregular
    loanword plurals) pulled from tier2-rules.json

TOKEN-AWARENESS (added after the person's request to stop false-rejecting
genuine English code-switching): the alphabet check itself was already fine
— alphabet.json's core 35-letter set already covers every plain ASCII letter
A-Z (Q/V/X are core letters, only C/J are the separately-"borrowed,
code-switch-only" ones) — so a plain English word was never going to trip
`invalid_character`. The real false-positive risk was the GLUED-DIGIT check:
a digit glued to letters is exactly what an un-decoded Arabizi leak looks
like (Rule 0 violation), but it's *also* what ordinary tech/brand code-switch
looks like (Web3, iPhone7, G7). Both look identical at the character level;
they only differ by token shape and position. `classify_token()` below tells
them apart with a few heuristics (ALL-CAPS acronym, digit-as-suffix-after-a-
vowelled-word, presence of the code-switch-only letters C/J, a small known-
codeswitch lookup) — documented as heuristics, not a certainty, since this
can't be solved perfectly without full lexical knowledge of every possible
brand name. Known remaining gap: a digit in the MIDDLE of an otherwise
plausible code-switch word (b2b) isn't distinguished from a genuine Arabizi
leak by these heuristics — flagged here rather than silently over-claimed.

Usage:
  python3 validate_output.py --text "El naharda el gaw Ϩelw."
  python3 validate_output.py --eval_results eval_results.json --out validated_results.json
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

SRC = Path(__file__).parent.parent / "source"

with open(SRC / "alphabet.json", encoding="utf-8") as f:
    _ALPHABET = json.load(f)
with open(SRC / "tier2-rules.json", encoding="utf-8") as f:
    _RULES = json.load(f)

# --- Build the valid Tier 2 character set -----------------------------------
_VALID_CHARS = set()
for letter in _ALPHABET["alphabet"]:
    _VALID_CHARS.add(letter["letter_upper"])
    _VALID_CHARS.add(letter["letter_lower"])
for b in _ALPHABET.get("borrowed_orthography", []):
    _VALID_CHARS.add(b["letter_upper"])
    _VALID_CHARS.add(b["letter_lower"])
# vowel-length marks, common punctuation/diacritics used in the spec's own examples
_VALID_CHARS.update("āēīōūáéíóúàèìòùâêîôûäëïöü'’ʿ")
_VALID_CHARS.update(" \t\n.,!?;:\"'()-–—…،؛؟/%0123456789&@")

_MASRI_SPECIAL_CHARS = set("ΘθϨϩXxÐðƔɣϢϣṢṣḌḍṬṭẒẓⲴⲵƐɐ")  # letters with no plain-ASCII overlap with English

_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")
# a digit that is glued to Latin letters (no surrounding space) is suspicious —
# genuine numbers are normally space/punctuation-delimited tokens. Scoped to
# the {2,3,5,7,8} set Rule 0 actually names as phoneme-substitutes; other
# digits (0,1,4,6,9) are just numbers even when glued (Covid19, GPT4).
_GLUED_DIGIT_RE = re.compile(r"[A-Za-zÀ-ſϢϣϨϩⲴⲵɣƔ][23578][A-Za-zÀ-ſϢϣϨϩⲴⲵɣƔ]|[A-Za-zÀ-ſϢϣϨϩⲴⲵɣƔ][23578]\b|\b[23578][A-Za-zÀ-ſϢϣϨϩⲴⲵɣƔ]")
_EL_FUSED_RE = re.compile(
    r"\bel[-\u2010\u2011][A-Za-zϢϣϨϩⲴⲵɣƔÐðṢṣḌḍṬṭẒẓɐƐ]|\bel[A-Za-zΘθGϨϩXDÐðRZSϢϣṢṣḌḍṬṭẒẓⲴⲵɣƔFVQKLMNHOUWEIYɐƐ]",
    re.IGNORECASE,
)

# A handful of code-switch words seen in this project's own examples
# (mixed_script_examples.json) plus common tech/interjection terms. Not
# meant to be exhaustive — extend from source/lexicon.json's code_switch
# entries once that's populated; this is the always-on fallback list.
_KNOWN_CODESWITCH_WORDS = {
    "save", "file", "wow", "ok", "okay", "share", "email", "app", "web",
    "pixel", "piano", "pizza", "escript", "script",
}

# --- Closed-list lexicon (memorized exceptions, hand-pulled from tier2-rules.json
# and source/lexicon.json — see build_lexicon.py for the fuller sqlite index) --
# Not exhaustive by design (source/lexicon.sqlite, built by build_lexicon.py, is
# for at-scale lookup) — this is the small always-on checklist the system
# prompt itself calls out as "memorized, not derivable."
_CLOSED_LIST = {}
for lw in _RULES.get("loanword_examples", []):
    if lw.get("tier1"):
        _CLOSED_LIST[lw["arabic"]] = lw["tier1"]
for w in _RULES.get("standardized_word_list", []):
    _CLOSED_LIST[w["arabic"]] = w["tier2"]
try:
    with open(SRC / "lexicon.json", encoding="utf-8") as f:
        _lexicon_doc = json.load(f)
    for entry in _lexicon_doc.get("entries", []):
        for ar in entry.get("arabic_variants", []):
            _CLOSED_LIST[ar] = entry["canonical_masri"]
except FileNotFoundError:
    pass

VALID, VALID_WITH_WARNING, INVALID = "VALID", "VALID_WITH_WARNING", "INVALID"

ERROR_TAXONOMY = {
    "script_id_error": "Arabic-script characters present in output",
    "invalid_character": "character(s) outside the Tier 2 alphabet",
    "numeral_as_letter": "digit (2/3/5/7/8) appears glued to letters in a Masri-context token — possible unconverted Arabizi (Rule 0)",
    "el_merge_error": "'el' fused or hyphenated to the following word (rule 2 violation)",
    "capitalization_error": "sentence does not start with a capital letter (rule 8 violation)",
}

_TOKEN_RE = re.compile(r"\S+")


def classify_token(raw_token: str) -> str:
    """Best-effort per-token classification: 'masri', 'codeswitch', or
    'ambiguous'. Heuristic, not certain — see module docstring."""
    token = raw_token.strip(".,!?;:\"'()،؛؟…")
    if not token:
        return "other"
    lower = token.lower()

    if lower in _KNOWN_CODESWITCH_WORDS:
        return "codeswitch"
    if any(ch in _MASRI_SPECIAL_CHARS for ch in token):
        return "masri"  # uses a letterform with no plain-English overlap -> intentionally Masri
    if len(token) > 1 and token.isupper():
        return "codeswitch"  # ALL-CAPS acronym/interjection (WOW, GPT, G7)
    if any(ch in "CJcj" for ch in token):
        return "codeswitch"  # C/J are borrowed_orthography, code-switch-only per alphabet.json
    # digit-as-suffix pattern (Web3, iPhone7): letters (containing a vowel,
    # so it reads as a real word, not a bare consonant string) followed
    # directly by digit(s) at the END of the token.
    m = re.match(r"^([A-Za-z]{2,})([23578]+)$", token)
    if m and re.search(r"[aeiouAEIOU]", m.group(1)):
        return "codeswitch"
    return "ambiguous"  # plain Latin, no strong signal either way — treated leniently


def validate(text: str) -> dict:
    """Returns {"verdict": ..., "errors": [...], "warnings": [...]}"""
    errors = []
    warnings = []

    if _ARABIC_SCRIPT_RE.search(text):
        errors.append("script_id_error")

    bad_chars = sorted({ch for ch in text if ch not in _VALID_CHARS and not unicodedata.category(ch).startswith("M")})
    if bad_chars:
        errors.append("invalid_character")

    # Token-aware glued-digit check: only run against tokens NOT classified
    # as codeswitch, so Web3/iPhone7/G7/WOW-style tokens don't trip Rule 0
    # enforcement meant for un-decoded Arabizi leaks in Masri-context text.
    token_classes = {}
    flagged_digit_tokens = []
    for raw_tok in _TOKEN_RE.findall(text):
        cls = classify_token(raw_tok)
        token_classes[raw_tok] = cls
        if cls != "codeswitch" and _GLUED_DIGIT_RE.search(raw_tok):
            flagged_digit_tokens.append(raw_tok)
    if flagged_digit_tokens:
        warnings.append("numeral_as_letter")

    if _EL_FUSED_RE.search(text):
        errors.append("el_merge_error")

    stripped = text.strip()
    if stripped and stripped[0].isalpha() and not stripped[0].isupper() and stripped[0].lower() != stripped[0].upper():
        warnings.append("capitalization_error")

    if errors:
        verdict = INVALID
    elif warnings:
        verdict = VALID_WITH_WARNING
    else:
        verdict = VALID

    return {
        "verdict": verdict,
        "errors": errors,
        "warnings": warnings,
        "bad_characters": bad_chars,
        "flagged_digit_tokens": flagged_digit_tokens,
        "token_classification": token_classes,
    }


def check_closed_list(arabic_source: str, output: str) -> dict | None:
    """If arabic_source is a known closed-list word, check output matches exactly."""
    expected = _CLOSED_LIST.get(arabic_source.strip())
    if expected is None:
        return None
    ok = expected in output
    return {
        "closed_list_match": ok,
        "expected_form": expected,
        "error": None if ok else "dictionary_violation",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None, help="Validate a single string")
    ap.add_argument("--eval_results", default=None, help="Path to eval_results.json (from evaluate.py)")
    ap.add_argument("--out", default="validated_results.json")
    args = ap.parse_args()

    if args.text:
        result = validate(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.eval_results:
        rows = json.loads(Path(args.eval_results).read_text(encoding="utf-8"))
        n_invalid = n_warning = 0
        for row in rows:
            v = validate(row.get("model_output", ""))
            row["validation"] = v
            if v["verdict"] == INVALID:
                n_invalid += 1
            elif v["verdict"] == VALID_WITH_WARNING:
                n_warning += 1
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{len(rows)} rows validated: {n_invalid} INVALID, {n_warning} VALID_WITH_WARNING, "
              f"{len(rows) - n_invalid - n_warning} VALID")
        print(f"Written to {args.out}")
        return

    ap.error("pass either --text or --eval_results")


if __name__ == "__main__":
    main()
