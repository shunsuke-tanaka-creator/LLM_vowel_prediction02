#!/usr/bin/env python3
"""eval.jsonl.gz でビーム生成し、母音列→文 復元の精度を測る。
完全一致率(top1/top8)と文字レベル精度(正規化編集距離ベース)を出力する。"""
from __future__ import annotations

import argparse
import gzip
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from infer import predict


def char_accuracy(pred: str, gold: str) -> float:
    """1 - (編集距離 / max長) で文字レベル精度を返す(0〜1)。"""
    n, m = len(pred), len(gold)
    if n == 0 and m == 0:
        return 1.0
    # Levenshtein 距離
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if pred[i - 1] == gold[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    dist = dp[m]
    return 1.0 - dist / max(n, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="openbmb/MiniCPM4-0.5B")
    ap.add_argument("--lora_dir", default="lora_vowel_out")
    ap.add_argument("--eval_path", default="../dataset/project06/eval.jsonl.gz")
    ap.add_argument("--revision", default="2aaa97c53d")
    ap.add_argument("--max_records", type=int, default=500, help="0 なら全件")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--num_beams", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--vis", action="store_true", default=False, help="推論結果を1件ずつ表示する")
    ap.add_argument("--vis_n", type=int, default=50)
    args = ap.parse_args()

    print("Loading model:", args.base_model, "lora:", args.lora_dir)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, revision=args.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        revision=args.revision,
    )
    model = PeftModel.from_pretrained(base, args.lora_dir)
    model.eval()
    print("Ready.\n")

    total = 0
    top1_exact = 0
    topk_exact = 0
    char_acc_sum = 0.0  # top1 の文字精度合計
    vis_rows = []

    plan_total = args.max_records if args.max_records > 0 else None
    with gzip.open(args.eval_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, total=plan_total, desc="eval restore"):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            gold = ex["sentence"]
            cands = predict(model, tok, ex["vowels"], ex.get("ctx", "<NONE>"),
                            args.k, args.num_beams, args.max_new_tokens)

            total += 1
            top1 = cands[0] if cands else ""
            if top1 == gold:
                top1_exact += 1
            if gold in cands:
                topk_exact += 1
            char_acc_sum += char_accuracy(top1, gold)

            if args.vis and len(vis_rows) < args.vis_n:
                vis_rows.append((ex, gold, cands))

            if args.max_records > 0 and total >= args.max_records:
                break

    print("=" * 60)
    print(f"used records          : {total}")
    print(f"top1 exact match acc  : {top1_exact/total*100:.2f}%  ({top1_exact}/{total})")
    print(f"top{args.k} exact match acc  : {topk_exact/total*100:.2f}%  ({topk_exact}/{total})")
    print(f"top1 char accuracy    : {char_acc_sum/total*100:.2f}%  (1 - 編集距離/最大長 の平均)")
    print("=" * 60)

    if args.vis and vis_rows:
        print(f"\n--- 推論結果 {len(vis_rows)} 件 ---")
        for k, (ex, gold, cands) in enumerate(vis_rows, 1):
            mark = "○" if gold in cands else "×"
            print(f"[{k}] CTX={ex.get('ctx', '<NONE>')!r}  VOWELS=[{ex['vowels']}]")
            print(f"     正解={gold!r}  / 予測top{min(len(cands),3)}={cands[:3]}  -> {mark}")


if __name__ == "__main__":
    main()
