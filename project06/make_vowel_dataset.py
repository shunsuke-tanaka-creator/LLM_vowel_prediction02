#!/usr/bin/env python3
"""CC-100 と字幕(OpenSubtitles/JESC)を同程度混合し、「。」区切りで文を作り、
母音列(空白区切り)→ 原文 を復元する SFT 用データ {vowels, ctx, sentence} を生成する。"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re

from datasets import load_dataset
from tqdm import tqdm

from corpus_utils import split_sentences, is_valid_sentence
from vowel_utils import text_to_vowel_str

PROMPT_CTX_NONE = "<NONE>"

# fetch_corpus.py と同じ字幕ソース定義: (HF名, split, 日本語カラム候補)
SUBS_SOURCES = {
    "opensubtitles": ("Nan-Do/OpenSubtitlesJapanese", "train", ["TEXT", "text"]),
    "jesc": ("nntsuzu/JESC", "train", ["ja"]),
}


def pick_ja(row: dict, cand_cols) -> str:
    # fetch_corpus.py から流用: 日本語カラムを取り出す
    for c in cand_cols:
        if c in row and row[c]:
            return str(row[c])
    for v in row.values():
        if isinstance(v, dict) and "ja" in v and v["ja"]:
            return str(v["ja"])
    return ""


def iter_cc100_texts(max_rows: int):
    """CC-100-ja から原文(text)を逐次 yield。max_rows 行で打ち切り。"""
    ds = load_dataset("range3/cc100-ja", split="train", streaming=True)
    seen = 0
    for row in ds:
        text = row.get("text", "")
        if text:
            yield text
        seen += 1
        if max_rows > 0 and seen >= max_rows:
            break


def iter_subs_texts(max_rows: int):
    """字幕(OpenSubtitles + JESC)から原文を逐次 yield。両ソース合算で max_rows 行。"""
    half = max_rows // 2 if max_rows > 0 else 0
    for name in ["opensubtitles", "jesc"]:
        hf_name, split, cand_cols = SUBS_SOURCES[name]
        ds = load_dataset(hf_name, split=split, streaming=True)
        seen = 0
        for row in ds:
            line = pick_ja(row, cand_cols)
            if line:
                yield line
            seen += 1
            if half > 0 and seen >= half:
                break


def text_to_examples(text: str, min_len: int, max_len: int, none_ctx_ratio: float,
                     comma_ctx_ratio: float = 0.5):
    """1つの原文を「。」区切りで文に分割し、{vowels, ctx, sentence} のリストを返す。

    「、」を含む文は comma_ctx_ratio の確率で最初の「、」で分割し、前半を ctx
    (末尾の「、」は除去)、後半を sentence として ctx あり事例を増やす(追加)。
    """
    sents = split_sentences(text)
    out = []
    prev = PROMPT_CTX_NONE  # 直前の文(原文)。チャンク先頭は <NONE>
    for s in sents:
        # 追加: 「、」を含む文は一定確率で最初の「、」で ctx/sentence に分割する
        comma_ctx = None
        body = s
        if "、" in s and random.random() < comma_ctx_ratio:
            head, tail = s.split("、", 1)  # 最初の「、」で前半/後半に分割
            head = head.rstrip("、")  # ctx 側末尾の「、」は除去
            tail = tail.lstrip("、")  # 後半先頭の「、」は除去(連続読点・先頭読点対策)
            if head and tail:  # 前半・後半ともに非空のときだけ分割を採用
                comma_ctx = head
                body = tail

        if not is_valid_sentence(body, min_len, max_len):
            prev = PROMPT_CTX_NONE  # 無効文で文脈を切る
            continue
        vowels = text_to_vowel_str(body)
        if not vowels:
            prev = PROMPT_CTX_NONE
            continue
        if comma_ctx is not None and is_valid_sentence(comma_ctx, min_len, max_len):
            ctx = comma_ctx  # 文内「、」分割で作った ctx は潰さない
        else:
            ctx = prev
            if random.random() < none_ctx_ratio:  # 一定割合で母音列のみ入力にする
                ctx = PROMPT_CTX_NONE
        sentence = body
        if random.random() < 0.5:  # 追加: 文末「。」を50%(文単位)で vowels/sentence 両方に付与
            sentence = body + "。"
            vowels = vowels + " 。"
        out.append({"vowels": vowels, "ctx": ctx, "sentence": sentence})
        prev = s  # ctx は記号なしの素の文を引き継ぐ
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_train", type=str, default="../dataset/project06/train03.jsonl.gz")
    ap.add_argument("--out_eval", type=str, default="../dataset/project06/eval03.jsonl.gz")

    ap.add_argument("--cc100_rows", type=int, default=1_000_000, help="CC-100 から読む原文行数")
    ap.add_argument("--subs_rows", type=int, default=1_000_000, help="字幕から読む原文行数(両ソース合算)")

    ap.add_argument("--min_len", type=int, default=2, help="文の最小文字数")
    ap.add_argument("--max_len", type=int, default=20, help="文の最大文字数")
    ap.add_argument("--none_ctx_ratio", type=float, default=0.5, help="CTX を <NONE> にする割合")
    ap.add_argument("--comma_ctx_ratio", type=float, default=0.5,
                    help="追加: 「、」を含む文を最初の「、」で分割して文内 ctx を作る割合")
    ap.add_argument("--eval_ratio", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    import os
    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)

    ft_train = gzip.open(args.out_train, "wt", encoding="utf-8")
    ft_eval = gzip.open(args.out_eval, "wt", encoding="utf-8")

    seen_set = set()  # sentence の重複除去
    written_train = 0
    written_eval = 0

    # CC-100 と字幕を同程度の件数で交互に混ぜる(zip で片方が尽きるまで交互)
    cc_iter = iter_cc100_texts(args.cc100_rows)
    subs_iter = iter_subs_texts(args.subs_rows)

    def interleave():
        cc_done = subs_done = False
        while not (cc_done and subs_done):
            if not cc_done:
                try:
                    yield next(cc_iter)
                except StopIteration:
                    cc_done = True
            if not subs_done:
                try:
                    yield next(subs_iter)
                except StopIteration:
                    subs_done = True

    total_rows = args.cc100_rows + (args.subs_rows // 2) * 2
    pbar = tqdm(total=total_rows if total_rows > 0 else None, desc="make vowel dataset")

    for text in interleave():
        pbar.update(1)
        for ex in text_to_examples(text, args.min_len, args.max_len, args.none_ctx_ratio,
                                   args.comma_ctx_ratio):
            if ex["sentence"] in seen_set:  # 重複文は捨てる
                continue
            seen_set.add(ex["sentence"])
            if random.random() < args.eval_ratio:
                ft_eval.write(json.dumps(ex, ensure_ascii=False) + "\n")
                written_eval += 1
            else:
                ft_train.write(json.dumps(ex, ensure_ascii=False) + "\n")
                written_train += 1

    pbar.close()
    ft_train.close()
    ft_eval.close()

    print("DONE")
    print("train:", args.out_train, "records:", written_train)
    print("eval :", args.out_eval, "records:", written_eval)


if __name__ == "__main__":
    main()
    # fugashi/datasets のスレッド解放順序による終了時クラッシュ回避のため即時終了
    import os
    os._exit(0)
