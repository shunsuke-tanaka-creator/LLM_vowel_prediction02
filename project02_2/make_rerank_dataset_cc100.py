#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import random
from typing import Dict, List, Tuple

from datasets import load_dataset
from tqdm import tqdm

from morph_utils import extract_content_words_with_pos
from vowel_utils import reading_to_vowel_str


PROMPT_CTX_NONE = "<NONE>"


def load_vocab(vocab_jsonl: str) -> Dict[str, Tuple[str, str]]:
    """
    word -> (vowels, reading)
    """
    out: Dict[str, Tuple[str, str]] = {}
    with open(vocab_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["word"]] = (r["vowels"], r.get("reading", ""))
    return out


def load_vowel2cands(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_example(ctx_text: str, vowels: str, cands: List[str], gold_word: str) -> dict:
    # gold が入ってなければ最後を差し替え
    if gold_word not in cands:
        cands = cands[:-1] + [gold_word]

    # candidates をシャッフル（位置依存を殺す）
    random.shuffle(cands)

    ans = str(cands.index(gold_word) + 1)  # 1-based
    ctx = ctx_text if ctx_text else PROMPT_CTX_NONE  # 変更: 原文そのままの CTX
    return {"ctx": ctx, "vowels": vowels, "cands": cands, "answer": ans}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_jsonl", type=str, default="vocab_freq.jsonl")
    ap.add_argument("--vowel2cands", type=str, default="vowel2cands.json")

    ap.add_argument("--out_train", type=str, default="../dataset/project02_2/train.jsonl.gz")
    ap.add_argument("--out_eval", type=str, default="../dataset/project02_2/eval.jsonl.gz")

    ap.add_argument("--max_rows", type=int, default=2_000_000)
    ap.add_argument("--ctx_len", type=int, default=4)

    ap.add_argument("--num_cands", type=int, default=32)
    ap.add_argument("--eval_ratio", type=float, default=0.002)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    vocab = load_vocab(args.vocab_jsonl)
    vowel2cands = load_vowel2cands(args.vowel2cands)

    ds = load_dataset("range3/cc100-ja", split="train", streaming=True)

    ft_train = gzip.open(args.out_train, "wt", encoding="utf-8")
    ft_eval = gzip.open(args.out_eval, "wt", encoding="utf-8")

    seen = 0
    written_train = 0
    written_eval = 0

    pbar = tqdm(total=args.max_rows, desc="make rerank dataset")

    for row in ds:
        text = row.get("text", "")
        if not text:
            continue

        # content words only（開始位置付き）
        items = extract_content_words_with_pos(text)  # 変更: 原文位置付き抽出
        words = [w for (w, _, _, _) in items]
        starts = [st for (_, _, _, st) in items]  # 追加: 各 word の原文開始位置

        if len(words) < 2:
            continue

        # 逐次：次の単語を当てる
        for i in range(1, len(words)):
            gold = words[i]
            if gold not in vocab:
                continue
            vowels, _ = vocab[gold]
            if vowels not in vowel2cands:
                continue

            cand_pool = vowel2cands[vowels]
            if len(cand_pool) < 2:
                continue

            cands = cand_pool[: args.num_cands]
            if len(cands) < args.num_cands:
                continue

            # 変更: CTX は原文そのまま。ctx_len 個前の自立語開始位置から gold 直前までを切り出す
            ctx_word_start = max(0, i - args.ctx_len)
            ctx_char_start = starts[ctx_word_start]
            ctx_char_end = starts[i]
            ctx_text = text[ctx_char_start:ctx_char_end].strip()

            ex = make_example(ctx_text, vowels, cands, gold)

            if random.random() < args.eval_ratio:
                ft_eval.write(json.dumps(ex, ensure_ascii=False) + "\n")
                written_eval += 1
            else:
                ft_train.write(json.dumps(ex, ensure_ascii=False) + "\n")
                written_train += 1

        seen += 1
        pbar.update(1)
        if seen >= args.max_rows:
            break

    pbar.close()
    ft_train.close()
    ft_eval.close()

    print("DONE")
    print("train:", args.out_train, "records:", written_train)
    print("eval :", args.out_eval, "records:", written_eval)


if __name__ == "__main__":
    main()
