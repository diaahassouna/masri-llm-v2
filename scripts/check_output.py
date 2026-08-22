#!/usr/bin/env python3
"""
check_output.py — Print raw model predictions next to expected answers,
straight from eval_results.json (produced by eval_score.py / evaluate.py).

Use this whenever the pass-rate table looks wrong (e.g. everything failing)
to see the ACTUAL text the model produced, not just PASS/FAIL — that's what
tells you whether it's a real quality problem or a script/format bug.

Usage:
  python3 check_output.py                      # first 5 rows
  python3 check_output.py --n 10                # first 10 rows
  python3 check_output.py --category tha_merger # only one category
  python3 check_output.py --only_fails          # skip anything that passed
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--file", default="eval_results.json")
ap.add_argument("--n", type=int, default=5, help="how many rows to print (0 = all)")
ap.add_argument("--category", default=None, help="only show this category")
ap.add_argument("--only_fails", action="store_true", help="skip EXACT/ACCEPTABLE rows")
args = ap.parse_args()

rows = json.load(open(args.file, encoding="utf-8"))

if args.category:
    rows = [r for r in rows if r["category"] == args.category]
if args.only_fails:
    rows = [r for r in rows if r["verdict"] == "FAIL"]
if args.n:
    rows = rows[: args.n]

for r in rows:
    print(f"--- {r['id']} [{r['category']}] verdict={r['verdict']} ---")
    print("INPUT:   ", r.get("input", ""))
    print("EXPECTED:", repr(r["expected"]))
    print("GOT:     ", repr(r["model_output"])[:300])
    print()

print(f"Shown: {len(rows)} row(s)")
