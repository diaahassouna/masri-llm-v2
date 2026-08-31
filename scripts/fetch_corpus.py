#!/usr/bin/env python3
"""
fetch_corpus.py — Pull real, natural Egyptian Arabic sentences for the
Masri training corpus, instead of relying on an LLM to invent sentences
from scratch (which is what augment_with_gemini.py / augment_with_claude.py
currently do).

Source: arz.wikipedia.org — Egyptian Arabic Wikipedia. It's written
natively in colloquial/Masri Egyptian Arabic (not MSA), and is CC BY-SA
licensed (attribution required if you republish the raw corpus; using it
as training data for a derived model is standard practice and doesn't
require per-sentence attribution, but keep a note of the source).

This uses the public MediaWiki API (no auth, no key needed):
  1. action=query&list=random         -> get a batch of random article IDs
  2. action=query&prop=extracts       -> get their plain-text content
  3. split into sentences, filter, dedupe, write out

Requires: pip install requests

Usage:
  python3 fetch_corpus.py --n 5000 --out ../data/corpus_sentences.jsonl
"""
import argparse
import json
import re
import time
from pathlib import Path

import requests

API_URL = "https://arz.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "masri-llm-v2-corpus-fetcher/1.0 (research/training data collection)"}

# Arabic sentence-ending punctuation (Arabic full stop, question mark, etc.)
# plus standard Latin punctuation that also shows up in mixed text.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟۔])\s+|\n+")
ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
# Strip residual wiki markup that sometimes survives extraction (refs, braces, pipes)
WIKI_JUNK_RE = re.compile(r"\[\d+\]|\{\{.*?\}\}|\[\[|\]\]|\|")


def fetch_random_page_ids(n, session):
    """Get n random article page IDs (namespace 0 = real articles only)."""
    ids = []
    while len(ids) < n:
        batch = min(500, n - len(ids))  # API max per request is 500
        resp = session.get(API_URL, params={
            "action": "query",
            "list": "random",
            "rnnamespace": 0,
            "rnlimit": batch,
            "format": "json",
        }, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        ids.extend(p["id"] for p in data["query"]["random"])
    return ids


def fetch_extracts(page_ids, session):
    """Get plain-text extracts for a batch of page IDs (max 20 per request for extracts)."""
    texts = []
    for i in range(0, len(page_ids), 20):
        chunk = page_ids[i:i + 20]
        resp = session.get(API_URL, params={
            "action": "query",
            "pageids": "|".join(str(p) for p in chunk),
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "plain",
            "format": "json",
        }, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            if extract:
                texts.append(extract)
    return texts


def extract_sentences(text, min_words, max_words):
    sentences = []
    for raw in SENTENCE_SPLIT_RE.split(text):
        s = WIKI_JUNK_RE.sub(" ", raw).strip()
        if not s:
            continue
        word_count = len(s.split())
        if not (min_words <= word_count <= max_words):
            continue
        # Require the sentence to be predominantly Arabic script, not a stray
        # English heading, table fragment, or numeric list left over from extraction.
        arabic_chars = len(ARABIC_CHAR_RE.findall(s))
        if arabic_chars < 0.6 * len(s.replace(" ", "")):
            continue
        sentences.append(s)
    return sentences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="target number of unique sentences")
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--max-words", type=int, default=25)
    ap.add_argument("--out", default=str(Path(__file__).parent.parent / "data" / "corpus_sentences.jsonl"))
    ap.add_argument("--sleep-seconds", type=float, default=0.5,
                     help="Pause between API calls to be a polite Wikipedia API citizen.")
    args = ap.parse_args()

    session = requests.Session()
    seen = set()
    collected = []

    print(f"[info] Target: {args.n} unique sentences ({args.min_words}-{args.max_words} words each)")

    while len(collected) < args.n:
        remaining = args.n - len(collected)
        # Over-fetch pages since many sentences per article get filtered out.
        n_pages = max(20, min(200, remaining // 3 + 10))

        try:
            page_ids = fetch_random_page_ids(n_pages, session)
            texts = fetch_extracts(page_ids, session)
        except requests.RequestException as e:
            print(f"[warn] request failed ({e}), retrying after pause...")
            time.sleep(5)
            continue

        for text in texts:
            for sentence in extract_sentences(text, args.min_words, args.max_words):
                if sentence not in seen:
                    seen.add(sentence)
                    collected.append(sentence)
                    if len(collected) >= args.n:
                        break
            if len(collected) >= args.n:
                break

        print(f"[info] collected {len(collected)}/{args.n} unique sentences "
              f"(from {n_pages} articles this round)")
        time.sleep(args.sleep_seconds)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in collected:
            f.write(json.dumps({"arabic": s, "source": "arz.wikipedia.org"}, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(collected)} sentences to {out_path}")
    print("Source: Egyptian Arabic Wikipedia (arz.wikipedia.org), CC BY-SA 4.0.")
    print("NOTE: article quality varies (some contributors write closer to MSA than")
    print("colloquial Cairene). The augment_from_corpus.py step should normalize toward")
    print("natural Cairene phrasing, not transliterate MSA-leaning sentences literally.")


if __name__ == "__main__":
    main()
