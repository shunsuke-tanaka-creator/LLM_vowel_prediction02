#!/usr/bin/env python3
"""表4用: 同一母音列に対する各条件(a-e)の推論時間(応答時間)を計測する。

条件a-dは同じLoRA重みで CTX有無・ビーム幅のみ変える。条件eは素のベース(LoRAなし)。
各条件は warmup 1回を捨て、5回計測して平均[ms]を出す。
"""
from __future__ import annotations

import argparse
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from infer import predict  # 学習と完全一致した推論本体を流用

VOWELS = "o u a e n i a a u a a"  # 今日は天気が悪かった
CTX = "今日はいい一日でした"       # 条件a/c/d 用のCTX(母音列と同じ長さの実文脈)
REPEAT = 5

# (ラベル, use_ctx, num_beams, no_lora)
CONDS = [
    ("a: CTXあり・beam8 (提案)", True, 8, False),
    ("b: CTXなし・beam8", False, 8, False),
    ("c: CTXあり・beam4", True, 4, False),
    ("d: CTXあり・beam16", True, 16, False),
    ("e: LoRAなし・beam8", False, 8, True),
]


def timed_predict(model, tok, vowels, ctx, num_beams):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    cands = predict(model, tok, vowels, ctx, k=8, num_beams=num_beams, max_new_tokens=64)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.perf_counter() - t0) * 1000.0, cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="openbmb/MiniCPM4-0.5B")
    ap.add_argument("--lora_dir", default="logs/train_20260622_045837")
    ap.add_argument("--revision", default="2aaa97c53d")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, revision=args.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, revision=args.revision,
    )
    lora_model = PeftModel.from_pretrained(base, args.lora_dir)
    lora_model.eval()
    base.eval()
    dev = next(base.parameters()).device
    print(f"device: {dev}  vowels: {VOWELS}\n")

    print(f"{'condition':28s} {'mean[ms]':>10s} {'min[ms]':>9s} {'max[ms]':>9s}   top1")
    print("-" * 80)
    for label, use_ctx, num_beams, no_lora in CONDS:
        model = base if no_lora else lora_model
        ctx = CTX if use_ctx else ""
        timed_predict(model, tok, VOWELS, ctx, num_beams)  # warmup(捨て)
        ts = []
        top1 = ""
        for _ in range(REPEAT):
            ms, cands = timed_predict(model, tok, VOWELS, ctx, num_beams)
            ts.append(ms)
            top1 = cands[0] if cands else "(なし)"
        mean = sum(ts) / len(ts)
        print(f"{label:28s} {mean:10.1f} {min(ts):9.1f} {max(ts):9.1f}   {top1}")


if __name__ == "__main__":
    main()
