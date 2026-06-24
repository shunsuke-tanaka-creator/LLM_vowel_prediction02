#!/usr/bin/env python3
"""母音列→文 復元 CLI 推論。vowels(+ctx) を入力して復元文候補 TopK をビーム生成する。"""
from __future__ import annotations

import argparse
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_vowel_lora import build_prompt, PROMPT_CTX_NONE  # 学習と完全一致させるため共通利用


@torch.no_grad()
def predict(model, tok, vowels: str, ctx: str, k: int, num_beams: int, max_new_tokens: int) -> List[str]:
    ctx_norm = ctx.strip() if ctx and ctx.strip() else PROMPT_CTX_NONE
    prompt = build_prompt(vowels, ctx_norm) + "\n"  # 学習時の prompt(SENTENCE:\n まで)と一致
    enc = tok(prompt, return_tensors="pt").to(model.device)
    ids = enc["input_ids"]

    out = model.generate(
        input_ids=ids,
        attention_mask=enc["attention_mask"],
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,  # 追加: ビーム幅をそのまま使う(アブレーションで beam<k を評価可能にする)
        num_return_sequences=min(num_beams, k),  # 追加: 返す候補数はビーム幅を超えられない
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
        text = text.split("\n")[0].strip()  # 改行以降(続けて生成した場合)を切る
        if text and text not in seen:
            seen.add(text)
            cands.append(text)
    return cands[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="openbmb/MiniCPM4-0.5B")
    ap.add_argument("--lora_dir", default="lora_vowel_out")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--num_beams", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=64)  # 40字相当
    ap.add_argument("--revision", default="2aaa97c53d")  # transformers 4.46.3 互換 rev pin
    # 単発実行用(指定すれば対話に入らず1回だけ推論)
    ap.add_argument("--vowels", default=None)
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

    def run(vowels: str, ctx: str):
        cands = predict(model, tok, vowels, ctx, args.k, args.num_beams, args.max_new_tokens)
        print(f"CTX   : {ctx if ctx else PROMPT_CTX_NONE}")
        print(f"VOWELS: {vowels}")
        print("復元文候補:")
        for rank, w in enumerate(cands, start=1):
            print(f"{rank}. {w}")

    # 単発モード
    if args.vowels is not None:
        run(args.vowels, args.ctx or "")
        return

    # 対話モード(モデルは1回だけロード)
    print("対話モード: vowels と ctx を入力(空入力 or quit で終了)")
    while True:
        try:
            vowels = input("vowels(a i u e o n)> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not vowels or vowels in ("quit", "exit"):
            break
        try:
            ctx = input("ctx(任意)> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        run(vowels, ctx)
        print()


if __name__ == "__main__":
    main()
