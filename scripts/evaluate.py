#!/usr/bin/env python3
"""
evaluate.py — Score a (base or fine-tuned) model against data/eval_held_out.jsonl,
which is derived 1:1 from masri_tier2_eval_set.json. Never train on this file.

Usage:
  python3 evaluate.py --model yourname/masri-qwen2.5-7b
  python3 evaluate.py --model ../out/masri-lora --adapter_of Qwen/Qwen2.5-7B-Instruct

Outputs a per-category pass table and writes eval_results.json with every
model output next to its expected answer, so you can see exactly which rule
is failing (per masri_tier2_eval_set.json's "tests_for" field).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from validate_output import validate as validate_deterministic


def normalize(s: str) -> str:
    return " ".join(s.strip().split())


def load_model(model_id, adapter_of=None):
    tokenizer = AutoTokenizer.from_pretrained(adapter_of or model_id)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    if adapter_of:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            adapter_of, quantization_config=quant_config, dtype=torch.float16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, model_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quant_config, dtype=torch.float16, device_map="auto"
        )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, system_prompt, user_input, max_new_tokens=512):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # Qwen3 can still emit a </think> tag even with enable_thinking=False on some
    # checkpoints/adapters — strip anything up to and including it defensively.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument("--adapter_of", default=None, help="If --model is a LoRA adapter, the base model repo id")
    ap.add_argument("--eval_file", default="../data/eval_held_out.jsonl")
    ap.add_argument("--system_prompt_file", default="../data/system_prompt.txt")
    ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--filter_input_type", default=None, choices=["arabic_script", "franco"],
                     help="Only run rows of this input_type — added after the Aug 2026 audit found "
                          "franco rows were never represented in training; use this to check the "
                          "franco/arabic_script pass-rate gap specifically (Part 10 of the audit).")
    args = ap.parse_args()

    system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
    rows = [json.loads(l) for l in Path(args.eval_file).read_text(encoding="utf-8").splitlines()]
    if args.filter_input_type:
        rows = [r for r in rows if r.get("input_type") == args.filter_input_type]

    model, tokenizer = load_model(args.model, args.adapter_of)

    results = []
    tally = defaultdict(lambda: {"EXACT": 0, "ACCEPTABLE": 0, "FAIL": 0})
    tally_by_input_type = defaultdict(lambda: {"EXACT": 0, "ACCEPTABLE": 0, "FAIL": 0})

    for row in rows:
        output = generate(model, tokenizer, system_prompt, row["input"])
        norm_out = normalize(output)
        norm_expected = normalize(row["expected"])
        accepted = [normalize(v) for v in row.get("accepted_variants", [])]

        if norm_out == norm_expected:
            verdict = "EXACT"
        elif norm_out in accepted or any(norm_out in v or v in norm_out for v in accepted):
            verdict = "ACCEPTABLE"
        else:
            verdict = "FAIL"

        # Deterministic structural check (Part 9 of the audit) — runs regardless
        # of verdict, since a wrong-but-well-formed output and a wrong-and-
        # malformed output are different failure modes worth telling apart.
        validation = validate_deterministic(output)

        tally[row["category"]][verdict] += 1
        input_type = row.get("input_type", "unknown")
        tally_by_input_type[input_type][verdict] += 1
        results.append({**row, "model_output": output, "verdict": verdict, "validation": validation})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'category':<26}{'EXACT':>8}{'ACCEPT':>8}{'FAIL':>8}{'PASS%':>8}")
    total = {"EXACT": 0, "ACCEPTABLE": 0, "FAIL": 0}
    for cat, counts in sorted(tally.items()):
        n = sum(counts.values())
        passed = counts["EXACT"] + counts["ACCEPTABLE"]
        for k in total:
            total[k] += counts[k]
        print(f"{cat:<26}{counts['EXACT']:>8}{counts['ACCEPTABLE']:>8}{counts['FAIL']:>8}{100*passed/n:>7.0f}%")

    n_total = sum(total.values())
    passed_total = total["EXACT"] + total["ACCEPTABLE"]
    print("-" * 58)
    print(f"{'TOTAL':<26}{total['EXACT']:>8}{total['ACCEPTABLE']:>8}{total['FAIL']:>8}{100*passed_total/n_total:>7.0f}%")

    # input_type breakdown — the direct check for the Aug 2026 audit finding
    # (model trained with 0% franco coverage, evaluated with 28% franco).
    print(f"\n{'input_type':<26}{'EXACT':>8}{'ACCEPT':>8}{'FAIL':>8}{'PASS%':>8}")
    pass_rates = {}
    for itype, counts in sorted(tally_by_input_type.items()):
        n = sum(counts.values())
        passed = counts["EXACT"] + counts["ACCEPTABLE"]
        pass_rates[itype] = 100 * passed / n
        print(f"{itype:<26}{counts['EXACT']:>8}{counts['ACCEPTABLE']:>8}{counts['FAIL']:>8}{pass_rates[itype]:>7.0f}%")
    if "franco" in pass_rates and "arabic_script" in pass_rates:
        gap = pass_rates["arabic_script"] - pass_rates["franco"]
        flag = " [WARNING: >10pt gap]" if gap > 10 else ""
        print(f"\narabic_script - franco pass-rate gap: {gap:.0f}pts{flag}")

    n_invalid = sum(1 for r in results if r["validation"]["verdict"] == "INVALID")
    n_warn = sum(1 for r in results if r["validation"]["verdict"] == "VALID_WITH_WARNING")
    print(f"\nDeterministic validator: {n_invalid} INVALID, {n_warn} VALID_WITH_WARNING, "
          f"{len(results) - n_invalid - n_warn} VALID (see validation.errors/warnings per row)")
    print(f"\nFull per-item results written to {args.out}")


if __name__ == "__main__":
    main()
