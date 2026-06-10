#!/usr/bin/env python3
"""コマンドライン対話推論。サーバーを立てずに vowels(+ctx) を入力して候補を確認する。"""
from __future__ import annotations

import argparse
import json
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from morph_utils import extract_content_words_with_pos  # 原文位置付き抽出(学習データCTXと整合)

PROMPT_CTX_NONE = "<NONE>"
VALID_KEYS = {"a", "i", "u", "e", "o", "n"}


def validate_vowels(v: str) -> bool:
    toks = v.strip().split()
    return bool(toks) and all(t in VALID_KEYS for t in toks)


def normalize_ctx(ctx: str, ctx_len: int = 4) -> str:
    if not ctx:
        return PROMPT_CTX_NONE
    items = extract_content_words_with_pos(ctx)
    if not items:
        return PROMPT_CTX_NONE
    starts = [st for (_, _, _, st) in items]
    ctx_char_start = starts[max(0, len(starts) - ctx_len)]
    ctx_text = ctx[ctx_char_start:].strip()
    return ctx_text if ctx_text else PROMPT_CTX_NONE


def build_prompt(ctx_norm: str, vowels: str, cands: List[str]) -> str:
    lines = []
    lines.append("あなたは日本語IMEの予測器です。")
    lines.append("出力は候補番号のみ。")
    lines.append(f"CTX: {ctx_norm}")
    lines.append(f"VOWELS: {vowels}")
    lines.append("CANDIDATES:")
    for i, w in enumerate(cands, start=1):
        lines.append(f"{i}) {w}")
    lines.append("ANSWER:\n")
    return "\n".join(lines)


# 多桁候補番号(例 10〜32)を区別するため prompt + str(i) の対数尤度を合算するスコア
@torch.no_grad()
def score_candidate(model, tok, prompt_ids, num_str: str) -> float:
    cont_ids = tok(num_str, add_special_tokens=False)["input_ids"]
    if not cont_ids:
        return float("-inf")
    full_ids = torch.cat(
        [prompt_ids, torch.tensor([cont_ids], device=prompt_ids.device)], dim=1
    )
    out = model(input_ids=full_ids, use_cache=False)  # MiniCPM4 リモートコード互換のため use_cache=False
    logits = out.logits[0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    base = prompt_ids.shape[1] - 1
    total = 0.0
    for k, tid in enumerate(cont_ids):
        total += log_probs[base + k, tid].item()
    return total


@torch.no_grad()
def predict(model, tok, v2, ctx: str, vowels: str, k: int, num_cands: int, ctx_len: int):
    if not validate_vowels(vowels):
        print("Invalid vowel sequence. (a/i/u/e/o/n を空白区切り)")
        return
    if vowels not in v2:
        print("No candidates for:", vowels)
        return

    cands = v2[vowels][:num_cands]
    ctx_norm = normalize_ctx(ctx, ctx_len)
    prompt = build_prompt(ctx_norm, vowels, cands)

    prompt_ids = tok(prompt, return_tensors="pt").to(model.device)["input_ids"]
    scored = []
    for i in range(1, len(cands) + 1):
        s = score_candidate(model, tok, prompt_ids, str(i))
        scored.append((i, s))
    scored.sort(key=lambda x: x[1], reverse=True)

    print(f"CTX: {ctx_norm}")
    print(f"VOWELS: {vowels}")
    print("TOP:")
    for rank, (idx, sc) in enumerate(scored[:k], start=1):
        print(f"{rank}. {cands[idx-1]}  score={sc:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="openbmb/MiniCPM4-0.5B")
    ap.add_argument("--lora_dir", default="project02/lora_smoke_out")
    ap.add_argument("--vowel2cands", default="vowel2cands.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--num_cands", type=int, default=32)
    ap.add_argument("--ctx_len", type=int, default=4)
    ap.add_argument("--revision", default="2aaa97c53d")  # transformers 4.46.3 互換 rev pin
    # 単発実行用(指定すれば対話に入らず1回だけ推論)
    ap.add_argument("--ctx", default=None)
    ap.add_argument("--vowels", default=None)
    args = ap.parse_args()

    with open(args.vowel2cands, "r", encoding="utf-8") as f:
        v2 = json.load(f)

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

    # 単発モード
    if args.vowels is not None:
        predict(model, tok, v2, args.ctx or "", args.vowels, args.k, args.num_cands, args.ctx_len)
        return

    # 対話モード(モデルは1回だけロード)
    print("対話モード: ctx と vowels を入力(空入力 or quit で終了)")
    while True:
        try:
            ctx = input("ctx> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if ctx in ("quit", "exit"):
            break
        try:
            vowels = input("vowels(a i u e o n)> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not vowels or vowels in ("quit", "exit"):
            break
        predict(model, tok, v2, ctx, vowels, args.k, args.num_cands, args.ctx_len)
        print()


if __name__ == "__main__":
    main()
