#!/usr/bin/env python3
"""
Masri Writing System — inference script
=========================================

Runs the fine-tuned Masri model:
  base:    unsloth/Qwen3-4B-bnb-4bit   (== Qwen/Qwen3-4B, 4-bit)
  adapter: sightlake/masri-qwen3-4b-lora   (LoRA, trained with Unsloth)

The adapter is a LoRA, not a merged checkpoint, so it must be loaded on
top of the base model with PEFT (or Unsloth's fast loader — see the
`--fast` flag below).

The system prompt is the "Masri Tier 2" grounding document from the
masri-llm-v2 repo (data/system_prompt.txt). By default this script looks
for that file next to itself / in the repo, and falls back to downloading
it straight from GitHub raw if it isn't found locally.

--------------------------------------------------------------------
Setup
--------------------------------------------------------------------
    # Standard path (works on any CUDA box, incl. Colab/Kaggle):
    pip install -U transformers accelerate peft bitsandbytes

    # Faster load/inference path (recommended, same stack the model
    # was trained with):
    pip install -U "unsloth[colab-new]"   # see unsloth's install docs
                                            # for your specific CUDA version

--------------------------------------------------------------------
Usage
--------------------------------------------------------------------
    # One-off conversion
    python masri_infer.py --text "ezzayak, 3amel eh enaharda?"

    # Interactive chat loop
    python masri_infer.py --chat

    # Force the plain transformers+peft loader instead of Unsloth
    python masri_infer.py --chat --no-fast

    # Point at a local clone of the repo (for the system prompt file)
    python masri_infer.py --chat --repo-dir /path/to/masri-llm-v2
"""

import argparse
import os
import sys

BASE_MODEL = "unsloth/Qwen3-4B-bnb-4bit"
ADAPTER = "sightlake/masri-qwen3-4b-lora"
SYSTEM_PROMPT_RAW_URL = (
    "https://raw.githubusercontent.com/diaahassouna/masri-llm-v2/"
    "main/data/system_prompt.txt"
)


def load_system_prompt(repo_dir: str | None) -> str:
    """Load the Masri Tier 2 system prompt, local file first, else GitHub raw."""
    candidates = []
    if repo_dir:
        candidates.append(os.path.join(repo_dir, "data", "system_prompt.txt"))
    candidates.append(os.path.join(os.path.dirname(__file__), "data", "system_prompt.txt"))
    candidates.append(os.path.join(os.getcwd(), "data", "system_prompt.txt"))

    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

    # Fall back to fetching it straight from the repo on GitHub.
    try:
        import urllib.request

        with urllib.request.urlopen(SYSTEM_PROMPT_RAW_URL, timeout=15) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:  # pragma: no cover
        print(
            f"[warn] Could not find system_prompt.txt locally and could not "
            f"download it ({e}). Proceeding with no system prompt.",
            file=sys.stderr,
        )
        return ""


def load_model_fast(load_in_4bit: bool = True):
    """Load base + LoRA adapter via Unsloth's FastLanguageModel (recommended)."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER,  # Unsloth will pull the base + apply the adapter
        max_seq_length=4096,
        load_in_4bit=load_in_4bit,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)  # enables native 2x faster generation
    return model, tokenizer


def load_model_transformers(load_in_4bit: bool = True):
    """Load base + LoRA adapter via plain transformers + peft (portable path)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if not load_in_4bit else None,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER)
    model.eval()
    return model, tokenizer


# The instruction wrapper every training example's user turn actually used
# (confirmed from data/train.jsonl) — the model was never trained on raw
# Franco/Arabic text as the user turn by itself, only wrapped like this.
INSTRUCTION_TEMPLATE = "eh el masri el sa7 lel gomla dee: {text}"


def generate(model, tokenizer, system_prompt: str, user_text: str,
             max_new_tokens: int = 512, temperature: float = 0.7,
             history: list | None = None, wrap_instruction: bool = True) -> str:
    import torch

    wrapped_text = INSTRUCTION_TEMPLATE.format(text=user_text) if wrap_instruction else user_text

    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": wrapped_text})

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,  # matches train_ddp.py — Qwen3 thinking disabled at train time
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][encoded["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Masri Writing System — model inference")
    parser.add_argument("--text", type=str, default=None,
                         help="Single input to convert/chat with (Arabic script or Franco).")
    parser.add_argument("--chat", action="store_true",
                         help="Start an interactive multi-turn chat loop.")
    parser.add_argument("--repo-dir", type=str, default=None,
                         help="Path to a local clone of masri-llm-v2 (for data/system_prompt.txt).")
    parser.add_argument("--fast", dest="fast", action="store_true", default=True,
                         help="Use Unsloth's FastLanguageModel loader (default).")
    parser.add_argument("--no-fast", dest="fast", action="store_false",
                         help="Use plain transformers + peft instead of Unsloth.")
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false", default=True,
                         help="Disable 4-bit quantized loading.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-wrap", dest="wrap_instruction", action="store_false", default=True,
                         help="Send raw text as the user turn instead of the trained instruction wrapper.")
    args = parser.parse_args()

    if not args.text and not args.chat:
        parser.error("Pass --text \"...\" for a single run, or --chat for interactive mode.")

    system_prompt = load_system_prompt(args.repo_dir)

    print(f"[info] Loading base model '{BASE_MODEL}' + adapter '{ADAPTER}' "
          f"({'unsloth' if args.fast else 'transformers+peft'}, "
          f"{'4-bit' if args.load_in_4bit else 'full precision'})...", file=sys.stderr)

    if args.fast:
        try:
            model, tokenizer = load_model_fast(load_in_4bit=args.load_in_4bit)
        except ImportError:
            print("[warn] unsloth not installed, falling back to transformers+peft. "
                  "Install with: pip install -U \"unsloth[colab-new]\"", file=sys.stderr)
            model, tokenizer = load_model_transformers(load_in_4bit=args.load_in_4bit)
    else:
        model, tokenizer = load_model_transformers(load_in_4bit=args.load_in_4bit)

    print("[info] Model ready.\n", file=sys.stderr)

    if args.chat:
        history: list = []
        print("Masri chat — type 'exit' or Ctrl+C to quit.\n")
        while True:
            try:
                user_text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user_text.lower() in {"exit", "quit"}:
                break
            if not user_text:
                continue
            reply = generate(
                model, tokenizer, system_prompt, user_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                history=history,
                wrap_instruction=args.wrap_instruction,
            )
            print(f"masri> {reply}\n")
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": reply})
    else:
        reply = generate(
            model, tokenizer, system_prompt, args.text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            wrap_instruction=args.wrap_instruction,
        )
        print(reply)


if __name__ == "__main__":
    main()
