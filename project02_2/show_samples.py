from __future__ import annotations

import argparse
import gzip
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="../dataset/project02_2/train.jsonl.gz")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--pool", type=int, default=200000, help="先頭からこの件数だけ読みその中からランダム抽出")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    rows = []
    with gzip.open(args.path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if i + 1 >= args.pool:
                break

    if len(rows) > args.n:
        rows = random.sample(rows, args.n)

    for k, ex in enumerate(rows, start=1):
        ans_idx = int(ex["answer"]) - 1
        gold = ex["cands"][ans_idx]
        print(f"[{k}] CTX={ex['ctx']!r}")
        print(f"     VOWELS=[{ex['vowels']}]  ANSWER#{ex['answer']} -> 正解={gold!r}")
        print(f"     CANDS(先頭8)={ex['cands'][:8]}")


if __name__ == "__main__":
    main()
