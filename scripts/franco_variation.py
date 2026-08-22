"""
franco_variation.py — Controlled Arabic-script -> Franco/Arabizi variation
engine. Replaces the old single-mapping `arabizi_of()` in build_dataset.py.

WHY THIS EXISTS (Aug 2026 audit follow-up): the old `arabizi_of()` was a
fixed, deterministic per-character map — every Arabic source string produced
exactly ONE Franco spelling, always the same one. Real Franco writers don't
converge on one convention: و is "w" for some people and "o"/"u" for others,
غ is "gh" for some and "8" for others, ي alternates "y"/"ee"/"i" depending on
vowel length and personal habit. A model trained on a single fixed mapping
learns that ONE convention, not robustness to Arabizi variation generally —
exactly the failure mode the project's own spec (masri_tier2_system_prompt.md
Core principle) warns against ("silently identify the underlying word... not
the Latin characters as typed" only works if training saw more than one
typing of that word).

DESIGN — "controlled," not random noise:
- Each Arabic letter maps to a small, closed, real-world-attested set of
  Franco realizations, each with a relative weight (roughly: how common that
  spelling is among Cairene Franco writers). This is a judgment call, marked
  provisional — see NOTES below — not a linguistically verified frequency
  study.
- Selection is done with a seeded RNG (`random.Random(seed)`), so a given
  (arabic_text, seed) pair always reproduces the same output — builds stay
  reproducible even though the mapping is no longer 1:1.
- `n_variants(arabic_text, n, base_seed)` generates up to `n` DISTINCT
  Franco renderings of the same source string by drawing with different
  seeds and de-duplicating — this is what build_dataset.py uses to put more
  than one Franco spelling of the same underlying sentence into training
  data, instead of one.
- Digits used for phoneme-substitution letters (2/3/5/7/8) are one option
  among several per letter, never the only option — so training data also
  contains non-digit Franco spellings (kh, gh, etc.), keeping the model from
  learning "digit = only valid Franco spelling of this sound."
- Diacritics (tashkil) are stripped before mapping — matches the old
  behavior, still correct: Franco writers don't type Arabic vowel marks.

RULE ZERO INTERACTION: this module only ever produces FRANCO INPUT text (the
user-turn side of a training example). It must never be used to produce or
alter a Masri TARGET/assistant-turn string — those come only from the
hand-curated `masri`/`tier2` fields in source/*.json, which already satisfy
Rule Zero (masri_tier2_system_prompt.md, Rule 0) by construction, since no
digit-as-phoneme spelling is ever hand-authored into a canonical Masri
answer. build_dataset.py's `_assert_rule_zero()` (see bottom of that file)
checks this mechanically across the whole built dataset, not just here.

NOTES / provisional judgment calls (flag for native-speaker review):
- ح: kept single-option ("7") rather than adding "h" as an alternate,
  because "h" is also the default realization of ه — allowing both would
  make the two phonemes collide in Franco input, which would then require
  guessing which one was meant with no signal. Revisit if real corpus data
  shows "h" for ح is common enough to be worth the added ambiguity.
- ق/ء: single option ("2") — no widely-used non-digit Franco alternative
  exists for this merger in casual Cairene typing, unlike gh/kh which do
  have letter alternatives.
- ة (ta marbuta): kept as "a" only (not varied) — its Franco realization is
  fairly stable in practice; flag for revisit if evidence says otherwise.

KNOWN LIMITATION (inherited from the old arabizi_of(), not introduced here):
Arabic script doesn't spell short vowels, so a character-level map has
nothing to draw a short vowel from. القمر (qamar, "the moon") renders as
"el2mr" here, not the "el2amar" a real Franco writer would type from knowing
the word's pronunciation. Fixing this properly needs a pronunciation
lexicon/predictor, not more variation rules — out of scope for this module.
Short training examples built from single vocabulary words are least
affected; longer sentences with many consonant-only-in-script words are most
affected. Flagged here rather than silently shipped.
"""
import random

_ARABIC_DIACRITICS = "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670"

# Multi-character sequences must be checked before single-character ones.
# Each entry: arabic sequence -> list of (spelling, weight).
_MULTI_VARIANTS = {
    "ال": [("el", 10)],                      # definite article prefix, Franco-conventional
    "ث": [("th", 6), ("s", 2), ("t", 1)],      # thā — merges variably; th is the typed default
    "ش": [("sh", 9), ("$", 1)],                # $ is a rarer but real casual-typing convention
    "خ": [("5", 6), ("kh", 4)],
    "ذ": [("z", 6), ("th", 2)],
    "غ": [("gh", 5), ("8", 5)],
}

_SINGLE_VARIANTS = {
    "ا": [("a", 10)], "أ": [("a", 10)], "إ": [("a", 10)], "آ": [("aa", 10)],
    "ء": [("2", 10)],
    "ب": [("b", 10)], "پ": [("b", 10)],
    "ت": [("t", 10)], "ة": [("a", 10)],
    "ج": [("g", 8), ("j", 2)],           # Cairene hard-g default, "j" used by some writers
    "ح": [("7", 10)],
    "د": [("d", 10)], "ر": [("r", 10)], "ز": [("z", 10)],
    "س": [("s", 10)], "ص": [("s", 8), ("9", 2)],
    "ض": [("d", 8), ("9'", 1), ("d'", 1)],
    "ط": [("t", 7), ("6", 3)],
    "ظ": [("z", 8), ("6'", 2)],
    "ع": [("3", 10)],
    "ف": [("f", 10)], "ڤ": [("v", 10)],
    "ق": [("2", 10)],
    "ك": [("k", 10)], "ل": [("l", 10)], "م": [("m", 10)], "ن": [("n", 10)],
    "ه": [("h", 10)],
    "و": [("w", 6), ("o", 2), ("u", 2)],
    "ي": [("y", 6), ("i", 2), ("ee", 2)],
    "ى": [("a", 10)], "ئ": [("2", 10)],
}


def _weighted_choice(rng: random.Random, options):
    total = sum(w for _, w in options)
    r = rng.uniform(0, total)
    upto = 0
    for spelling, w in options:
        upto += w
        if r <= upto:
            return spelling
    return options[-1][0]


def arabizi_of(arabic_text: str, seed: int = 0) -> str:
    """Render one controlled, reproducible Franco variant of arabic_text.
    Same (arabic_text, seed) always returns the same string."""
    rng = random.Random(seed)
    text = "".join(ch for ch in arabic_text if ch not in _ARABIC_DIACRITICS)

    out = []
    i = 0
    while i < len(text):
        matched = False
        for seq, options in _MULTI_VARIANTS.items():
            if text.startswith(seq, i):
                out.append(_weighted_choice(rng, options))
                i += len(seq)
                matched = True
                break
        if matched:
            continue
        ch = text[i]
        if ch in _SINGLE_VARIANTS:
            out.append(_weighted_choice(rng, _SINGLE_VARIANTS[ch]))
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def n_variants(arabic_text: str, n: int, base_seed: int = 0) -> list[str]:
    """Up to n DISTINCT Franco renderings of the same source string.
    Draws with increasing seeds and de-duplicates; returns fewer than n if
    the variation space for this particular string is smaller than n
    (e.g. short words with no varying letters only ever produce 1 form)."""
    seen = []
    seed = base_seed
    attempts = 0
    while len(seen) < n and attempts < n * 6:
        variant = arabizi_of(arabic_text, seed=seed)
        if variant not in seen:
            seen.append(variant)
        seed += 1
        attempts += 1
    return seen
