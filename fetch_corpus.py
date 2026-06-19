#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import re

from datasets import load_dataset
from tqdm import tqdm

# データセット定義: (HF名, split, 日本語カラム候補)
SOURCES = {
    "opensubtitles": ("Nan-Do/OpenSubtitlesJapanese", "train", ["TEXT", "text"]),
    "jesc": ("nntsuzu/JESC", "train", ["ja"]),
}

_RE_HAS_JA = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")


def pick_ja(row: dict, cand_cols: list[str]) -> str:
    for c in cand_cols:
        if c in row and row[c]:
            return str(row[c])
    # フォールバック: 値が dict (translation 構造) の場合
    for v in row.values():
        if isinstance(v, dict) and "ja" in v and v["ja"]:
            return str(v["ja"])
    return ""


def clean_line(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "", s)  # 字幕の途中改行・空白を詰める
    return s


def iter_source(name: str, max_rows: int):
    hf_name, split, cand_cols = SOURCES[name]
    ds = load_dataset(hf_name, split=split, streaming=True)
    seen = 0
    for row in ds:
        line = pick_ja(row, cand_cols)
        if line:
            yield line
        seen += 1
        if max_rows > 0 and seen >= max_rows:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["opensubtitles", "jesc", "both"], default="both")
    ap.add_argument("--max_rows", type=int, default=20000, help="各ソースの最大行数。0で全件")
    ap.add_argument("--out", type=str, default="corpus_ja.txt.gz")
    ap.add_argument("--min_len", type=int, default=2, help="最小文字数")
    ap.add_argument("--max_len", type=int, default=80, help="最大文字数")
    args = ap.parse_args()

    if args.source == "both":
        names = ["opensubtitles", "jesc"]
    else:
        names = [args.source]

    seen_set: set[str] = set()
    written = 0

    with gzip.open(args.out, "wt", encoding="utf-8") as fout:
        for name in names:
            pbar = tqdm(desc=f"fetch {name}", total=(args.max_rows or None))
            for raw in iter_source(name, args.max_rows):
                pbar.update(1)
                line = clean_line(raw)
                if len(line) < args.min_len or len(line) > args.max_len:
                    continue
                if not _RE_HAS_JA.search(line):  # 日本語を含まない行は捨てる
                    continue
                if line in seen_set:  # 重複除去
                    continue
                seen_set.add(line)
                fout.write(line + "\n")
                written += 1
            pbar.close()

    print("DONE")
    print("out:", args.out, "written:", written)


if __name__ == "__main__":
    main()
