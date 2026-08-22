#!/usr/bin/env python3
"""
build_lexicon.py — Builds source/lexicon.sqlite from source/lexicon.json.

source/lexicon.json is the SOURCE OF TRUTH — hand-edit that file directly to
add or correct dictionary entries. This script does not read tier2-rules.json
loanword tables directly anymore (it used to; lexicon.json was migrated from
them once, then became independently editable) — it only reads lexicon.json.

lexicon.sqlite is a disposable, regeneratable index, built ON DEMAND for
fast inference-time lookup (indexed by arabic/masri form). Running this
script is optional for day-to-day lexicon editing — you only need it before
an inference run that actually queries the sqlite index, not every time you
add a word to lexicon.json. Don't hand-edit the .sqlite file; don't commit
it as if it were authoritative — if it and lexicon.json ever disagree,
lexicon.json wins and the .sqlite should be regenerated.

Usage:
  python3 build_lexicon.py
"""
import json
import sqlite3
from pathlib import Path

SRC = Path(__file__).parent.parent / "source"
DB_PATH = SRC / "lexicon.sqlite"

SCHEMA = """
CREATE TABLE lexicon (
    id TEXT PRIMARY KEY,
    canonical_masri TEXT NOT NULL,
    canonical_masri_variants TEXT,   -- JSON array
    arabic_variants TEXT,            -- JSON array
    arabizi_variants TEXT,           -- JSON array
    ipa TEXT,
    part_of_speech TEXT,
    gender TEXT,
    source_language TEXT,
    loanword_status TEXT,            -- established | recognized_variant | code_switch
    code_switch_form TEXT,
    confidence TEXT,                 -- established | recognized_variant | uncertain
    attestation_source TEXT,
    notes TEXT
);
CREATE INDEX idx_arabic ON lexicon(arabic_variants);
CREATE INDEX idx_masri ON lexicon(canonical_masri);
CREATE INDEX idx_status ON lexicon(loanword_status);
"""


def main():
    with open(SRC / "lexicon.json", encoding="utf-8") as f:
        lexicon_doc = json.load(f)

    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)

    rows = [
        (
            e["id"], e["canonical_masri"],
            json.dumps(e.get("canonical_masri_variants", [])),
            json.dumps(e.get("arabic_variants", [])),
            json.dumps(e.get("arabizi_variants", [])),
            e.get("ipa"), e.get("part_of_speech"), e.get("gender"),
            e.get("source_language"), e.get("loanword_status"),
            e.get("code_switch_form"), e.get("confidence"),
            e.get("attestation_source"), e.get("notes"),
        )
        for e in lexicon_doc["entries"]
    ]

    conn.executemany(
        """INSERT INTO lexicon
           (id, canonical_masri, canonical_masri_variants, arabic_variants,
            arabizi_variants, ipa, part_of_speech, gender, source_language,
            loanword_status, code_switch_form, confidence, attestation_source, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()

    n = conn.execute("SELECT COUNT(*) FROM lexicon").fetchone()[0]
    print(f"Wrote {n} lexicon entries to {DB_PATH} (source: source/lexicon.json, "
          f"schema_version {lexicon_doc.get('meta', {}).get('schema_version', '?')})")
    conn.close()


if __name__ == "__main__":
    main()
