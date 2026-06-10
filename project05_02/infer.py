#!/usr/bin/env python3
"""語尾予測 CLI 推論。stem(+ctx) を入力して語尾候補 TopK をビーム生成する。"""
from __future__ import annotations

import argparse
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_CTX_NONE = "<NONE>"


# train_suffix_lora.build_prompt と完全一致させること
def build_prompt(stem: str, ctx: str) -> str:
    lines = []
    lines.append("あなたは日本語IMEの語尾予測器です。")
    lines.append("語幹に続く自然な語尾だけを出力してください。")
    lines.append(f"CTX: {ctx}")
    lines.append(f"STEM: {stem}")
    lines.append("SUFFIX:")
    return "\n".join(lines)


@torch.no_grad()
def predict(model, tok, stem: str, ctx: str, k: int, num_beams: int, max_new_tokens: int) -> List[str]:
    ctx_norm = ctx.strip() if ctx and ctx.strip() else PROMPT_CTX_NONE
    prompt = build_prompt(stem, ctx_norm)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    ids = enc["input_ids"]

    out = model.generate(
        input_ids=ids,
        attention_mask=enc["attention_mask"],
        max_new_tokens=max_new_tokens,
        num_beams=max(num_beams, k),
        num_return_sequences=max(num_beams, k),
        do_sample=False,
        early_stopping=True,
        pad_token_id=tok.pad_token_id,
        use_cache=False,  # MiniCPM4 リモートコード互換のため
    )
    gen = out[:, ids.shape[1]:]
    seen = set()
    cands: List[str] = []
    for row in gen:
        text = tok.decode(row, skip_special_tokens=True).strip()
        # 改行以降(モデルが続けてしまった場合)を切る
        text = text.split("\n")[0].strip()
        if text and text not in seen:
            seen.add(text)
            cands.append(text)
    return cands[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="openbmb/MiniCPM4-0.5B")
    ap.add_argument("--lora_dir", default="lora_suffix_out")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--num_beams", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--revision", default="2aaa97c53d")  # transformers 4.46.3 互換 rev pin
    # 単発実行用(指定すれば対話に入らず1回だけ推論)
    ap.add_argument("--stem", default=None)
    ap.add_argument("--ctx", default=None)
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

    def run(stem: str, ctx: str):
        cands = predict(model, tok, stem, ctx, args.k, args.num_beams, args.max_new_tokens)
        print(f"CTX : {ctx if ctx else PROMPT_CTX_NONE}")
        print(f"STEM: {stem}")
        print("SUFFIX候補:")
        for rank, w in enumerate(cands, start=1):
            print(f"{rank}. {stem}{w}   (語尾: {w})")

    # 単発モード
    if args.stem is not None:
        run(args.stem, args.ctx or "")
        return

    # 対話モード
    print("対話モード: stem と ctx を入力(空入力 or quit で終了)")
    while True:
        try:
            stem = input("stem> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not stem or stem in ("quit", "exit"):
            break
        try:
            ctx = input("ctx(任意)> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        run(stem, ctx)
        print()


if __name__ == "__main__":
    main()
