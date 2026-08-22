#!/usr/bin/env python3
"""
build_dataset.py — Turns Diaa Hassouna's Masri source-of-truth files
(alphabet.json, tier2-rules.json, masri_tier2_system_prompt.md,
masri_tier2_eval_set.json) into Hugging Face-ready SFT datasets.

Outputs (into ../data/):
  train.jsonl        — SFT training examples (chat format: system/user/assistant)
  eval_held_out.jsonl — eval-set-derived examples, kept OUT of train.jsonl on purpose
                         (this is masri_tier2_eval_set.json — use it only for scoring,
                         never for training, or your eval numbers become meaningless)
  dataset_stats.json — counts per category, for sanity-checking coverage

Run:
  python3 build_dataset.py
"""
import json
import random
from pathlib import Path

from franco_variation import n_variants

random.seed(42)

SRC = Path(__file__).parent.parent / "source"
OUT = Path(__file__).parent.parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

with open(SRC / "alphabet.json", encoding="utf-8") as f:
    alphabet = json.load(f)
with open(SRC / "tier2-rules.json", encoding="utf-8") as f:
    rules = json.load(f)
with open(SRC / "masri_tier2_system_prompt.md", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()
with open(SRC / "masri_tier2_eval_set.json", encoding="utf-8") as f:
    eval_set = json.load(f)

examples = []  # list of dicts: {"messages": [...], "category": str, "source": str}


def add(user, assistant, category, source, input_type="arabic_script"):
    examples.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "category": category,
            "source": source,
            "input_type": input_type,
        }
    )


_franco_seed_counter = [1000]  # avoids reusing seeds across unrelated words


def add_franco(make_user, assistant, category, source, arabic_source):
    """Adds FRANCO_VARIANTS_PER_EXAMPLE distinct Franco-input training rows
    for the same underlying arabic_source/assistant pair, using the
    controlled variation engine (franco_variation.py) instead of one fixed
    spelling. make_user(variant_text) builds the user-turn string."""
    _franco_seed_counter[0] += 37  # arbitrary stride, just needs to vary
    for v in n_variants(arabic_source, FRANCO_VARIANTS_PER_EXAMPLE, base_seed=_franco_seed_counter[0]):
        add(make_user(v), assistant, category, source, input_type="franco")


# ---------------------------------------------------------------------------
# Arabizi/Franco generation now lives in franco_variation.py -- a controlled
# multi-variant engine (see that file's docstring), replacing the old
# single fixed-mapping arabizi_of() that used to live here. Every Arabic-
# script example below gets N distinct Franco spellings via n_variants(),
# not one, so training data teaches robustness to real Franco spelling
# variation (w/o/u for waw, gh/8 for ghain, etc.) instead of one fixed
# convention.
FRANCO_VARIANTS_PER_EXAMPLE = 2



# ---------------------------------------------------------------------------
# 1. Alphabet grounding — letter <-> Arabic <-> IPA <-> name, both directions.
#    This teaches the base model the actual symbol inventory before it ever
#    has to use the symbols in running text.
# ---------------------------------------------------------------------------
letter_q_templates_ar_to_masri = [
    "إيه الحرف بتاع {ar} في الأبجدية المصرية؟",
    "اكتبلي حرف {ar} بالأبجدية المصرية.",
]
letter_q_templates_masri_to_ar = [
    "الحرف {up}/{lo} في الأبجدية المصرية بيمثل إيه في العربي؟",
]

for letter in alphabet["alphabet"]:
    up, lo, ar, name, ipa = (
        letter["letter_upper"],
        letter["letter_lower"],
        letter["arabic"],
        letter["name"],
        letter["ipa"],
    )
    notes_ar = letter.get("notes_ar", "")
    q1 = random.choice(letter_q_templates_ar_to_masri).format(ar=ar)
    a1 = f"حرف {ar} في الأبجدية المصرية (الفئة الثانية) بيتكتب {up}/{lo} ({name}). {notes_ar}".strip()
    add(q1, a1, "alphabet", "alphabet.json")

    q2 = random.choice(letter_q_templates_masri_to_ar).format(up=up, lo=lo)
    a2 = f"{up}/{lo} ({name}, {ipa}) بيمثل حرف {ar} في العربي. {notes_ar}".strip()
    add(q2, a2, "alphabet", "alphabet.json")

for b in alphabet["borrowed_orthography"]:
    q = f"إمتى بستخدم حرف {b['letter_upper']}/{b['letter_lower']} في المصرية؟"
    a = f"{b['letter_upper']}/{b['letter_lower']} ({b['name']}) {b['notes_ar']}."
    add(q, a, "alphabet_borrowed", "alphabet.json")

# ---------------------------------------------------------------------------
# 2. Rule explanations — ask the model to state a rule, in Masri Tier 2 itself,
#    so the model practices *generating* Masri prose, not just converting into it.
# ---------------------------------------------------------------------------
RULE_EXPLANATIONS_MASRI = {
    "gemination": "El Ϩorouf el metⲴaddeda betetketeb metⲴadda fel kalema, ϨaⲴalāla ⲴaN maB6 el shadda maktouba weLLa la'. Zayy 'geddan' aw 'Ϩobb'.",
    "definite_article": "'El' betetketeb daayman keda, kelma leϨaaha, meϣ metdammega maⲴ elli baⲴdaha — meϣ 'elnaharda' walla 'el-naharda', laakenn 'el naharda'.",
    "glottal_stop": "Ɐ betetketeb bass fe nosṣ el kelma walla fe axerha, mesh fe awwelha. El hamza fe awwel el kelma dayman saakta we metmaktobaash.",
    "ayin": "Ⲵ (el Ⲵayn el Ⱳobṭeya el adeema) howa el Ϩarf elli beyeⲴber Ⲵan ص ع fel Ⲵarabi, fel Tier 2. Da axtar Ϩarf, laazem yetsahheb feeh koll marra.",
    "q_hamza_merger": "El Ⲵaammeya el Ɐaahereyya betⲴaamel el ق zayy el hamza — betwaddeeh Ɐ. Bass el q betfḍal lel formeyya aw el fosⲴa bass.",
}

add(
    "اشرحلي في المصرية إيه قاعدة الـgemination.",
    RULE_EXPLANATIONS_MASRI["gemination"],
    "rule_explanation",
    "tier2-rules.json + system prompt",
)
add(
    "قوللي بالمصري إمتى بنستخدم Ɐ.",
    RULE_EXPLANATIONS_MASRI["glottal_stop"],
    "rule_explanation",
    "tier2-rules.json",
)
add(
    "إيه أهم قاعدة في كتابة حرف العين بالمصرية؟ رد بالمصري.",
    RULE_EXPLANATIONS_MASRI["ayin"],
    "rule_explanation",
    "tier2-rules.json ayin_rule",
)

# ---------------------------------------------------------------------------
# 3. Ayin examples (both files) — direct conversion drills.
# ---------------------------------------------------------------------------
for ex in alphabet["ayin_rule"]["examples"]:
    add(
        f"حوّل الكلمة دي للمصرية (Tier 2): {ex['arabic']}",
        ex["tier2"],
        "ayin",
        "alphabet.json ayin_rule",
    )
    add_franco(
        lambda v: f"7awel el kelma dee lel masreya (Tier 2): {v}",
        ex["tier2"],
        "ayin",
        "alphabet.json ayin_rule",
        ex["arabic"],
    )

for ex in rules["ayin_rule"]["examples"]:
    add(
        f"حوّل الكلمة دي للمصرية (Tier 2): {ex['arabic']} (معناها: {ex['meaning']})",
        ex["tier2"],
        "ayin",
        "tier2-rules.json ayin_rule",
    )
    add_franco(
        lambda v, ex=ex: f"7awel el kelma dee lel masreya (Tier 2): {v} (meaning: {ex['meaning']})",
        ex["tier2"],
        "ayin",
        "tier2-rules.json ayin_rule",
        ex["arabic"],
    )

# ---------------------------------------------------------------------------
# 4. Spelling-rule worked examples (rule 1-6 blocks in tier2-rules.json)
# ---------------------------------------------------------------------------
for rule in rules["spelling_rules"]:
    for ex in rule.get("examples", []):
        if "→" not in ex:
            continue
        src, tgt = [p.strip() for p in ex.split("→", 1)]
        cat = f"rule_{rule['id']}_{rule['name'].lower().replace(' ', '_')}"
        add(
            f"حوّل الكلمة/الجملة دي للمصرية (Tier 2) وطبّق قاعدة '{rule['name']}': {src}",
            tgt,
            cat,
            "tier2-rules.json spelling_rules",
        )
        add_franco(
            lambda v, rule=rule: f"7awel el kelma/gomla dee lel masreya (Tier 2), we tabba2 2a3edet '{rule['name']}': {v}",
            tgt,
            cat,
            "tier2-rules.json spelling_rules",
            src,
        )

# ---------------------------------------------------------------------------
# 5. Standardized high-frequency words — memorized closed list.
# ---------------------------------------------------------------------------
for w in rules["standardized_word_list"]:
    add(
        f"إزاي بتتكتب '{w['arabic']}' بالمصرية (Tier 2)؟",
        w["tier2"],
        "standardized_word",
        "tier2-rules.json standardized_word_list",
    )
    add_franco(
        lambda v: f"ezay betetketeb '{v}' bel masreya (Tier 2)?",
        w["tier2"],
        "standardized_word",
        "tier2-rules.json standardized_word_list",
        w["arabic"],
    )

# ---------------------------------------------------------------------------
# 6. Loanword policy examples (correct vs wrong spelling contrast)
# ---------------------------------------------------------------------------
for lw in rules["loanword_examples"]:
    correct = lw.get("tier1")
    wrong = lw.get("wrong")
    if not correct:
        continue
    if wrong:
        answer = f"الصح: {correct}. الغلط الشائع: {wrong} — لازم نفرّق بين B/P عشان دول حروف مستقلة في المصرية."
        add(
            f"إيه الصح والغلط في كتابة '{lw['arabic']}' بالمصرية؟ ({lw.get('notes','')})",
            answer,
            "loanword_p_b_v_f",
            "tier2-rules.json loanword_examples",
        )
        add_franco(
            lambda v, answer=answer, lw=lw: f"eh el sa7 wel ghala6 fe ketabet '{v}' bel masreya? ({lw.get('notes','')})",
            answer,
            "loanword_p_b_v_f",
            "tier2-rules.json loanword_examples",
            lw["arabic"],
        )
    else:
        add(
            f"إزاي بتتكتب '{lw['arabic']}' بالمصرية؟ ({lw.get('source','')})",
            correct,
            "loanword",
            "tier2-rules.json loanword_examples",
        )
        add_franco(
            lambda v, lw=lw: f"ezay betetketeb '{v}' bel masreya? ({lw.get('source','')})",
            correct,
            "loanword",
            "tier2-rules.json loanword_examples",
            lw["arabic"],
        )

# ---------------------------------------------------------------------------
# 7. Stress-test sentences — pure Masri monolingual text, used as
#    "continue/respond in Masri" conversational turns so the model learns
#    natural running prose, not just isolated word conversions.
# ---------------------------------------------------------------------------
conversation_openers = [
    "قوللي حاجة عن مصر بالمصري.",
    "اكتبلي جملة طويلة بالمصري Tier 2 تجرب فيها أكتر من قاعدة.",
    "عايز مثال جملة مصرية معقدة تستخدم فيها حروف قبطية ويونانية.",
    "احكيلي عن يومك بالمصري.",
    "قوللي رأيك في حاجة بالمصري.",
]
for i, sent in enumerate(rules["sample_texts"]["stress_test_sentences_tier2"]):
    opener = conversation_openers[i % len(conversation_openers)]
    add(opener, sent, "conversational_masri", "tier2-rules.json stress_test_sentences_tier2")

add(
    "احكيلي عن مصر والقهوة المصرية بالمصري.",
    rules["sample_texts"]["tier2_academic"],
    "conversational_masri",
    "tier2-rules.json sample_texts",
)
add(
    f"حوّل النص ده للمصرية Tier 2: {rules['sample_texts']['arabic_script']}",
    rules["sample_texts"]["tier2_academic"],
    "full_text_conversion",
    "tier2-rules.json sample_texts",
)
add_franco(
    lambda v: f"7awel el nass da lel masreya Tier 2: {v}",
    rules["sample_texts"]["tier2_academic"],
    "full_text_conversion",
    "tier2-rules.json sample_texts",
    rules["sample_texts"]["arabic_script"],
)

# ---------------------------------------------------------------------------
# 8. Loanword phonology micro-rules called out in the system prompt but not
#    present as structured JSON — hand-encoded here since they're small and
#    high-value (epenthesis, French/Greek é, cinema/cima judgment call).
# ---------------------------------------------------------------------------
extra_prompt_examples = [
    ("حوّل دي للمصرية: أسانسير", "asansēr", "loanword_egyptianized_french"),
    ("حوّل دي للمصرية: طرابيزة", "ṭarabéza", "loanword_egyptianized_greek"),
    ("حوّل دي للمصرية: كلاكس", "kalaks", "loanword_epenthesis"),
    ("حوّل دي للمصرية: بيانو", "piano", "loanword_visual_spelling"),
    ("حوّل دي للمصرية: سينما (استخدم أي شكل مقبول)", "cinema", "loanword_judgment_call"),
]
for u, a, cat in extra_prompt_examples:
    add(u, a, cat, "masri_tier2_system_prompt.md")
    # Franco companion: strip the "حوّل دي للمصرية:" instruction and re-render
    # the Arabic target word in Arabizi, keeping the instruction in Franco too.
    ar_word = u.split(":", 1)[1].strip()
    add_franco(
        lambda v: f"7awel dee lel masreya: {v}",
        a,
        cat,
        "masri_tier2_system_prompt.md",
        ar_word,
    )

# ---------------------------------------------------------------------------
# Rule Zero enforcement (masri_tier2_system_prompt.md, Rule 0): a digit must
# never survive in a canonical Masri (assistant-turn) string as a letter-
# substitute. This checks every assistant string actually built above, not
# just the franco_variation.py module in isolation — catches the case where
# some other part of this script accidentally puts a phoneme-digit into a
# target string. A digit glued to letters (23al, 5al ...) is presumed to be
# a phoneme-digit leak; a standalone digit token (real numbers/quantities,
# which the spec explicitly allows to survive) is not flagged.
# ---------------------------------------------------------------------------
import re as _re

_GLUED_DIGIT_RE = _re.compile(r"[A-Za-zÀ-ſϢϣϨϩⲴⲵɣƔ][2357][A-Za-zÀ-ſϢϣϨϩⲴⲵɣƔ]|[A-Za-zÀ-ſϢϣϨϩⲴⲵɣƔ][2357]\b|\b[2357][A-Za-zÀ-ſϢϣϨϩⲴⲵɣƔ]")


def _assert_rule_zero(built_examples):
    violations = []
    for ex in built_examples:
        assistant_text = ex["messages"][2]["content"]
        if _GLUED_DIGIT_RE.search(assistant_text):
            violations.append((ex["category"], ex["source"], assistant_text))
    if violations:
        print(f"\nRULE ZERO VIOLATION: {len(violations)} assistant/masri-target string(s) "
              "contain a digit glued to letters (possible un-decoded phoneme-digit leak):")
        for cat, src, text in violations[:10]:
            print(f"  [{cat} / {src}] {text!r}")
        raise SystemExit(
            "Refusing to write data/ with Rule Zero violations present. "
            "Fix the offending example(s) in source/ or the generation code above, then re-run."
        )
    print(f"\nRule Zero check: {len(built_examples)} assistant/masri-target strings, 0 violations.")


_assert_rule_zero(examples)

# ---------------------------------------------------------------------------
# Shuffle, split off a small in-train dev slice, write files
# ---------------------------------------------------------------------------
random.shuffle(examples)
n_dev = max(20, int(0.05 * len(examples)))
dev = examples[:n_dev]
train = examples[n_dev:]

def _row(ex):
    # Metadata (category/source/input_type) is now written alongside messages
    # (previously dropped at write time, which is why dataset_stats.json could
    # silently go stale relative to the code that produced it — see audit).
    # Training/eval loaders should select on "messages" and ignore the rest.
    return {
        "messages": ex["messages"],
        "category": ex["category"],
        "source": ex["source"],
        "input_type": ex["input_type"],
    }


with open(OUT / "train.jsonl", "w", encoding="utf-8") as f:
    for ex in train:
        f.write(json.dumps(_row(ex), ensure_ascii=False) + "\n")

with open(OUT / "dev.jsonl", "w", encoding="utf-8") as f:
    for ex in dev:
        f.write(json.dumps(_row(ex), ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# Eval set: derived STRICTLY from masri_tier2_eval_set.json.
# Kept as a SEPARATE file, never merged into train.jsonl, so pass-rate numbers
# mean something. Two rows per item when a franco variant exists.
# ---------------------------------------------------------------------------
eval_rows = []
for item in eval_set["items"]:
    base = {
        "id": item["id"],
        "category": item["category"],
        "expected": item["expected"],
        "accepted_variants": item.get("accepted_variants", []),
        "tests_for": item["tests_for"],
    }
    row = dict(base)
    row["input"] = item["arabic_script"]
    row["input_type"] = "arabic_script"
    eval_rows.append(row)
    if item.get("franco"):
        row2 = dict(base)
        row2["input"] = item["franco"]
        row2["input_type"] = "franco"
        eval_rows.append(row2)

with open(OUT / "eval_held_out.jsonl", "w", encoding="utf-8") as f:
    for row in eval_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# also save the system prompt alongside the data, since eval/inference need it
with open(OUT / "system_prompt.txt", "w", encoding="utf-8") as f:
    f.write(SYSTEM_PROMPT)

# stats
from collections import Counter

stats = {
    "train_examples": len(train),
    "dev_examples": len(dev),
    "eval_examples": len(eval_rows),
    # input_type breakdown is the direct check for the Aug 2026 audit finding
    # (train had 0% franco while eval had 28%). Regenerated every build, never
    # hand-edited, so it can't go stale relative to this script again.
    "train_by_input_type": dict(Counter(e["input_type"] for e in train)),
    "eval_by_input_type": dict(Counter(r["input_type"] for r in eval_rows)),
    "train_by_category": dict(Counter(e["category"] for e in train)),
    "eval_by_category": dict(Counter(r["category"] for r in eval_rows)),
}
with open(OUT / "dataset_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print(json.dumps(stats, ensure_ascii=False, indent=2))

train_franco_pct = 100 * stats["train_by_input_type"].get("franco", 0) / len(train)
eval_franco_pct = 100 * stats["eval_by_input_type"].get("franco", 0) / len(eval_rows)
print(f"\ntrain franco%={train_franco_pct:.0f}  eval franco%={eval_franco_pct:.0f}", end="  ")
if abs(train_franco_pct - eval_franco_pct) > 15:
    print("[WARNING: train/eval franco coverage mismatch >15pts]")
else:
    print("[OK: within 15pts]")
