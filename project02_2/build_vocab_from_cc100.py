#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from datasets import load_dataset
from tqdm import tqdm

from morph_utils import extract_content_words
from vowel_utils import is_allowed_output_word, reading_to_vowel_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_vocab_jsonl", type=str, default="vocab_freq.jsonl")
    ap.add_argument("--out_vowel2cands", type=str, default="vowel2cands.json")
    ap.add_argument("--max_rows", type=int, default=2_000_000, help="まずは小さく回してOK。最大は cc100 全行")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_vocab", type=int, default=600_000, help="Counter pruning 上限（重いほど語彙増える）")
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--max_word_len", type=int, default=12)

    ap.add_argument("--topk_per_vowel", type=int, default=2000, help="母音ごとの候補上限（IME用）")
    args = ap.parse_args()

    random.seed(args.seed)

    ds = load_dataset("range3/cc100-ja", split="train", streaming=True)  # id,text :contentReference[oaicite:9]{index=9}

    word_counter: Counter[str] = Counter()
    word_meta: Dict[str, Tuple[str, str]] = {}  # word -> (pos1, reading)

    seen = 0
    pbar = tqdm(total=args.max_rows, desc="scan cc100-ja")

    for row in ds:
        text = row.get("text", "")
        if not text:
            continue

        items = extract_content_words(text)
        for (w, pos1, reading) in items:
            if not w:
                continue
            if len(w) > args.max_word_len:
                continue
            if not is_allowed_output_word(w):
                continue

            word_counter[w] += 1
            if w not in word_meta:
                word_meta[w] = (pos1, reading)

        seen += 1
        pbar.update(1)

        # pruning（heavy hitters 近似）
        if seen % 200_000 == 0 and len(word_counter) > int(args.max_vocab * 1.5):
            keep = dict(word_counter.most_common(args.max_vocab))
            word_counter = Counter(keep)
            # 追加: word_meta も残った語だけに絞る(活用形で語彙爆発するためメモリ枯渇を防ぐ)
            word_meta = {w: word_meta[w] for w in keep if w in word_meta}

        if seen >= args.max_rows:
            break

    pbar.close()

    # vocab 出力
    with open(args.out_vocab_jsonl, "w", encoding="utf-8") as f:
        for w, c in word_counter.most_common():
            if c < args.min_count:
                break
            pos1, reading = word_meta.get(w, ("", ""))
            vowels = reading_to_vowel_str(reading)
            if not vowels:
                continue
            rec = {
                "word": w,
                "pos1": pos1,
                "reading": reading,
                "vowels": vowels,
                "count": c,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # vowel -> candidates（頻度順）
    vowel2: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for w, c in word_counter.items():
        if c < args.min_count:
            continue
        pos1, reading = word_meta.get(w, ("", ""))
        vowels = reading_to_vowel_str(reading)
        if not vowels:
            continue
        vowel2[vowels].append((w, c))

    vowel2cands: Dict[str, List[str]] = {}
    for v, lst in vowel2.items():
        lst.sort(key=lambda x: x[1], reverse=True)
        vowel2cands[v] = [w for (w, _) in lst[: args.topk_per_vowel]]

    with open(args.out_vowel2cands, "w", encoding="utf-8") as f:
        json.dump(vowel2cands, f, ensure_ascii=False)

    print("DONE")
    print("vocab:", args.out_vocab_jsonl)
    print("vowel2cands:", args.out_vowel2cands)
    print("unique words:", len(word_counter))
    print("unique vowels:", len(vowel2cands))


if __name__ == "__main__":
    main()
