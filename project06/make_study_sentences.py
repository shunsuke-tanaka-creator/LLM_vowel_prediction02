#!/usr/bin/env python3
"""実ユーザ実験(認知負荷評価)用の課題文セットを CSV 生成する。

Level 1 = 単語, Level 2 = 10文字以内の短文, Level 3 = 一般文(11〜40字)。
各課題文に対し、正解の母音列(vowel_utils で算出)も同時に出力しておく。
これにより study_app.py 側で「被験者が打った母音列」と突き合わせて
母音変換誤り率(母音化CER)を集計できる。

出力: study/sentences.csv  (level, id, text, vowels, n_chars)
"""
from __future__ import annotations

import argparse
import csv
import os

from vowel_utils import text_to_vowel_str

# 課題文はここを編集して差し替える(単語集)。母音列は自動算出。各Level 3文ずつ。
# 追加: 各文に許容表記(alts)を持たせ、どちらの表記でも正解扱いにできる。
# (text, [許容表記...]) の形式。許容表記が無ければ空リスト。
L1_WORDS = [
    ("おはよう", []), ("パソコン", []), ("電話", []),
]
L2_SENTENCES = [
    ("今から帰る", []), ("気をつけてね", ["気を付けて"]),  # 追加: 表記ゆれ許容
    ("お腹がすいた", ["お腹が空いた"]),  # 追加: 表記ゆれ許容
]
L3_SENTENCES = [
    ("来週はパソコンを買いに行く", ["来週はパソコンを、買いに行く"]),  # 追加: 読点ありも許容
    ("子供の病院に行こう", []),
    ("車で行こうか、電車で行こうか", ["車で行こうか電車で行こうか"]),  # 追加: 読点なしも許容
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="study/sentences.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    for level, items in ((1, L1_WORDS), (2, L2_SENTENCES), (3, L3_SENTENCES)):
        for i, (text, alts) in enumerate(items, start=1):
            rows.append({
                "level": level,
                "id": f"L{level}-{i:02d}",
                "text": text,
                "vowels": text_to_vowel_str(text),
                "n_chars": len(text),
                "alts": "|".join(alts),  # 追加: 許容表記を | 区切りで保存
            })

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["level", "id", "text", "vowels", "n_chars", "alts"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} sentences -> {args.out}")
    for r in rows:
        print(f"  [{r['id']}] {r['text']}  ->  {r['vowels']}")


if __name__ == "__main__":
    main()
