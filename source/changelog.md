# Masri Tier 2 — Rule Change Log

Template for every proposed rule change (per the audit's Part 14 pattern):
current rule -> identified problem -> proposed rule -> justification -> examples
-> backward-compat implications -> training implications. Status is one of
PROVISIONAL (needs more evidence), ADOPTED (confirmed by author), or REJECTED.

---

## 2026-08-20 — Bound Egyptian morphology on unadapted foreign stems

- **Current rule:** loanword_examples / the code-switch vs. loanword decision
  framework (audit Part 4a) has two branches: genuine code-switch (preserve
  original spelling) or integrated Egyptianized loanword (Egyptianize the
  spelling). Morphological attachment is treated as evidence for the second
  branch.
- **Identified problem:** "أsave" (Egyptian first-person prefix أ- fused
  directly onto the English stem "save") doesn't fit either branch cleanly —
  Egyptian morphology is attached (branch-2 evidence) but the stem spelling
  stays unadapted English, not Egyptianized (branch-1 behavior).
- **Proposed rule:** add a third category — "bound morphology on unadapted
  stem" — where Egyptian affixes attach directly (no space) to a foreign stem
  that keeps its original spelling. Distinguish from full Egyptianization
  (bitza, biano) where the stem itself is respelled.
- **Examples:** أsave -> asave ("I'll save"/"[I] save"); by extension likely
  applies to other verb-prefixed English tech verbs (أshare, هsave, etc.) —
  not yet tested.
- **Backward compatibility:** additive, doesn't change any existing rule's
  output.
- **Training implications:** needs its own eval category and a handful of
  training examples; currently zero coverage (see
  `source/mixed_script_examples.json`, item mix-01).
- **Status:** PROVISIONAL — one confirmed instance, validated conversationally
  with the project author, not yet formally adopted into
  `masri_tier2_system_prompt.md`.

---

## 2026-08-20 — Productive Arabic-style plural on unlisted English loanwords

- **Current rule:** system prompt's loanword policy lists three specific
  fully-assimilated English words taking Arabic-style broken plurals
  (bonṭ/bonṭat, matϣ/matϣat, gon/egoaan) as a closed, memorized set — "when in
  doubt about a specific English loanword's plural, ask rather than inventing
  a pattern."
- **Identified problem:** "escript" -> "escriptat" (a plural not in the
  closed list) was produced and looked correct by the same pattern as
  bonṭ/bonṭat — suggesting the -at plural may be more productively
  generalizable for a subset of loanwords than "closed list, ask when unsure"
  implies.
- **Proposed rule:** none yet — flagging the tension for the author's
  judgment rather than resolving it. Two options: (a) keep the closed-list
  framing and treat "escriptat" as an acceptable one-off extension pending
  explicit approval per instance, or (b) identify the phonological/semantic
  property that makes -at productive (e.g. monosyllabic/disyllabic loan nouns
  ending in a consonant) and promote it to a derivable rule.
- **Backward compatibility:** N/A — no change made.
- **Training implications:** N/A — no change made.
- **Status:** PROVISIONAL — open question, not a proposed rule change.

---

## 2026-08-20 — Future clitic "ha-" spacing (open, unresolved)

- **Current rule:** none exists. The definite article "el" has an explicit
  always-separate-word rule (rule 2); the future clitic "ha-" has no
  equivalent rule anywhere in `masri_tier2_system_prompt.md` or
  `tier2-rules.json`.
- **Identified problem:** "ha tekteb" (two words) and "hatekteb" (fused) are
  both currently defensible readings of the spec — this is a genuine gap, not
  a violation of an existing rule.
- **Proposed rule:** treat consistently with "el" (separate word), for
  consistency of the "grammatical clitics are separate words" pattern — but
  this is a suggestion, not a justified linguistic conclusion.
- **Backward compatibility:** would need auditing against any existing
  "ha-"-prefixed forms already in loanword_examples/standardized_word_list
  (spot check: none found as of this entry).
- **Training implications:** none yet, since no rule exists to train toward.
- **Status:** PROVISIONAL — needs the author's decision before any data or
  code changes.
