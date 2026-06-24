#!/usr/bin/env python3
"""project06 の LoRA モデルで eval データを推論し、論文評価用 JSONL を出力する。

出力 1行: {"input_vowels", "gold", "candidates", "scores"(null)}
これを evaluate_for_paper.py に渡すと Acc@k / KSPC が計算できる。

project06 ルート(infer.py がある場所)から実行すること:
  source /home/shunsuke/Desktop/venv/pytorch_gpu/bin/activate
  python paper/scripts/dump_predictions.py \
    --lora_dir logs/train_20260611_134432 \
    --eval_path ../dataset/project06/eval.jsonl.gz \
    --max_records 1309 --k 8 \
    --out paper/results/preds_proposed.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# project06 ルートの infer.py を使う(プロンプトを学習と完全一致させるため)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from infer import predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="openbmb/MiniCPM4-0.5B")
    ap.add_argument("--lora_dir", default=None)  # 追加: 任意化(--no_lora 時は不要)
    ap.add_argument("--no_lora", action="store_true", default=False,
                    help="追加: LoRAを適用せず素のベースモデルで評価する(LoRA有無アブレーション用)")
    ap.add_argument("--eval_path", default="../dataset/project06/eval.jsonl.gz")
    ap.add_argument("--revision", default="2aaa97c53d")
    ap.add_argument("--max_records", type=int, default=0, help="0 で全件")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--num_beams", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--use_ctx", action="store_true", default=False,
                    help="eval の ctx を使う(既定は母音列のみ評価)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, revision=args.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, revision=args.revision,
    )
    if args.no_lora:  # 追加: LoRAを当てず素のベースモデルを評価
        model = base
    else:
        if not args.lora_dir:
            raise SystemExit("--lora_dir か --no_lora のどちらかを指定してください")
        model = PeftModel.from_pretrained(base, args.lora_dir)
    model.eval()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with gzip.open(args.eval_path, "rt", encoding="utf-8") as f, \
            open(args.out, "w", encoding="utf-8") as w:
        for line in tqdm(f, desc="dump preds"):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            ctx = ex.get("ctx", "<NONE>") if args.use_ctx else ""
            cands = predict(model, tok, ex["vowels"], ctx, args.k, args.num_beams, args.max_new_tokens)
            rec = {
                "input_vowels": ex["vowels"],
                "gold": ex["sentence"],
                "candidates": cands,
                "scores": None,  # ビーム生成のため候補スコアは保存しない(順位のみ意味を持つ)
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if args.max_records > 0 and n >= args.max_records:
                break
    print("DONE", args.out, "records:", n)


if __name__ == "__main__":
    main()
