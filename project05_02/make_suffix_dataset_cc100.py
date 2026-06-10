#!/usr/bin/env python3
"""CC100-ja から語尾予測用データ {stem, ctx, suffix} を抽出して train/eval.jsonl.gz を生成する。"""
from __future__ import annotations

import argparse
import gzip
import json
import random

from datasets import load_dataset
from tqdm import tqdm

from morph_suffix_utils import extract_stem_suffix

PROMPT_CTX_NONE = "<NONE>"


def build_ctx(text: str, items, idx: int, ctx_len: int) -> str:
    """idx 番目の用言の直前 ctx_len 個の用言開始位置から、その用言開始位置までを CTX とする。"""
    if idx == 0:
        return PROMPT_CTX_NONE
    ctx_word_start = max(0, idx - ctx_len)
    ctx_char_start = items[ctx_word_start].start_char
    ctx_char_end = items[idx].start_char
    ctx_text = text[ctx_char_start:ctx_char_end].strip()
    return ctx_text if ctx_text else PROMPT_CTX_NONE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_train", type=str, default="../dataset/project05_02/train.jsonl.gz")
    ap.add_argument("--out_eval", type=str, default="../dataset/project05_02/eval.jsonl.gz")
    ap.add_argument("--max_rows", type=int, default=2_000_000)
    ap.add_argument("--ctx_len", type=int, default=4)
    ap.add_argument("--max_suffix_len", type=int, default=16)  # 語尾の最大文字数(接続助詞/終助詞付きで長くなるため 12→16 に拡張)
    ap.add_argument("--none_ctx_ratio", type=float, default=0.3)  # CTX を <NONE> にする割合
    ap.add_argument("--eval_ratio", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    ds = load_dataset("range3/cc100-ja", split="train", streaming=True)

    ft_train = gzip.open(args.out_train, "wt", encoding="utf-8")
    ft_eval = gzip.open(args.out_eval, "wt", encoding="utf-8")

    seen = 0
    written_train = 0
    written_eval = 0
    pbar = tqdm(total=args.max_rows, desc="make suffix dataset")

    for row in ds:
        text = row.get("text", "")
        if not text:
            continue

        items = extract_stem_suffix(text)
        for idx, r in enumerate(items):
            if len(r.suffix) > args.max_suffix_len:
                continue

            ctx = build_ctx(text, items, idx, args.ctx_len)
            # 一定割合で CTX を落として「語幹のみ」入力にも対応できるようにする
            if random.random() < args.none_ctx_ratio:
                ctx = PROMPT_CTX_NONE

            ex = {"stem": r.stem, "ctx": ctx, "suffix": r.suffix}

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
    # fugashi(MeCab) と datasets のスレッド解放順序による終了時クラッシュを回避するため即時終了
    import os
    os._exit(0)
