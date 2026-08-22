#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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


def generate(model, tokenizer, system_prompt, user_input, max_new_tokens=128):
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
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter_of", default=None)
    ap.add_argument("--eval_file", default="../data/eval_held_out.jsonl")
    ap.add_argument("--system_prompt_file", default="../data/system_prompt.txt")
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
    rows = [json.loads(l) for l in Path(args.eval_file).read_text(encoding="utf-8").splitlines()]

    model, tokenizer = load_model(args.model, args.adapter_of)

    results = []
    tally = defaultdict(lambda: {"EXACT": 0, "ACCEPTABLE": 0, "FAIL": 0})

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

        tally[row["category"]][verdict] += 1
        results.append({**row, "model_output": output, "verdict": verdict})

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
    print(f"\nFull per-item results written to {args.out}")


if __name__ == "__main__":
    main()
