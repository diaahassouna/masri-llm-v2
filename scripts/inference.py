#!/usr/bin/env python3
"""
inference.py — Quick interactive chat with a fine-tuned (or base) Masri model.

Usage:
  python3 inference.py --model yourname/masri-qwen2.5-7b
  python3 inference.py --model ../out/masri-lora --adapter_of Qwen/Qwen3-8B
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapter_of", default=None)
ap.add_argument("--system_prompt_file", default="../data/system_prompt.txt")
args = ap.parse_args()

system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(args.adapter_of or args.model)
if args.adapter_of:
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(args.adapter_of, quantization_config=quant_config, dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base, args.model)
else:
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant_config, dtype=torch.float16, device_map="auto")
model.eval()

print("Masri converter ready. Type Arabic script or Franco/Arabizi. Ctrl+C to quit.\n")
history = [{"role": "system", "content": system_prompt}]

while True:
    try:
        user_input = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not user_input:
        continue
    history.append({"role": "user", "content": user_input})
    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.3, top_p=0.9)
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"masri> {reply}\n")
    history.append({"role": "assistant", "content": reply})
