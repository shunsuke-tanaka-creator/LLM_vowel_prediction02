#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from morph_utils import extract_content_words_with_pos  # 変更: 原文位置付き抽出に統一(学習データCTXと整合)

PROMPT_CTX_NONE = "<NONE>"
VALID_KEYS = {"a", "i", "u", "e", "o", "n"}


def validate_vowels(v: str) -> bool:
    toks = v.strip().split()
    return all(t in VALID_KEYS for t in toks)


def normalize_ctx(ctx: str, ctx_len: int = 4) -> str:
    if not ctx:
        return PROMPT_CTX_NONE
    # 変更: 学習データ(make_rerank)と同じく原文そのまま。末尾 ctx_len 語の開始位置から原文末尾までを切り出す
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
    prompt = "\n".join(lines)
    # 追加: プロンプト可視化用のデバッグ出力
    print("===== BUILT PROMPT =====")
    print(prompt)
    print("===== END PROMPT =====")
    return prompt


@torch.no_grad()
def score(model, tok, prompt: str):
    inp = tok(prompt, return_tensors="pt").to(model.device)
    out = model(**inp, use_cache=False)  # コメントに追加: MiniCPM4 のリモートコードは新Cache形式必須。1発forwardなのでKVキャッシュ不要 → use_cache=False で回避
    return out.logits[0, -1, :]


# コメントに追加: 多桁候補番号(例: 10〜32)を正しく区別するための、候補ごとの対数尤度合計スコア。
# コメントに追加: prompt + str(i) を tokenize し、str(i) のトークン群について対数尤度を順に合算する。
@torch.no_grad()
def score_candidate(model, tok, prompt_ids, num_str: str) -> float:
    cont_ids = tok(num_str, add_special_tokens=False)["input_ids"]
    if not cont_ids:
        return float("-inf")
    full_ids = torch.cat(
        [prompt_ids, torch.tensor([cont_ids], device=prompt_ids.device)], dim=1
    )
    out = model(input_ids=full_ids, use_cache=False)
    logits = out.logits[0]  # [seq_len, vocab]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    base = prompt_ids.shape[1] - 1  # prompt最後の位置の logits が次トークンの分布
    total = 0.0
    for k, tid in enumerate(cont_ids):
        total += log_probs[base + k, tid].item()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="openbmb/MiniCPM4-0.5B")
    ap.add_argument("--lora_dir", default="project02/lora_out_minicpm4")
    ap.add_argument("--vowel2cands", default="vowel2cands.json")
    ap.add_argument("--ctx", default="今日は冬だ。")
    ap.add_argument("--vowels", default="a u i")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--num_cands", type=int, default=32)
    # コメントに追加: MiniCPM4-0.5B 最新 rev は transformers>=4.50 が必須なので、4.46.3 環境では古い rev を pin
    ap.add_argument("--revision", default="2aaa97c53d")
    args = ap.parse_args()

    if not validate_vowels(args.vowels):
        print("Invalid vowel sequence.")
        return

    with open(args.vowel2cands, "r", encoding="utf-8") as f:
        v2 = json.load(f)

    if args.vowels not in v2:
        print("No candidates.")
        return

    cands = v2[args.vowels][: args.num_cands]

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, revision=args.revision)  # コメントに追加: MiniCPM4 用 trust_remote_code + rev pin

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,  # コメントに追加: MiniCPM4 用
        revision=args.revision,  # コメントに追加: rev pin
    )
    model = PeftModel.from_pretrained(base, args.lora_dir)
    model.eval()

    ctx_norm = normalize_ctx(args.ctx)
    prompt = build_prompt(ctx_norm, args.vowels, cands)

    # コメントに追加: 多桁候補に対応した対数尤度合計スコアで全候補をスコアリング
    prompt_ids = tok(prompt, return_tensors="pt").to(model.device)["input_ids"]
    scored = []
    for i in range(1, len(cands) + 1):
        s = score_candidate(model, tok, prompt_ids, str(i))
        scored.append((i, s))

    scored.sort(key=lambda x: x[1], reverse=True)

    print("TOP:")
    for rank, (idx, sc) in enumerate(scored[: args.k], start=1):
        print(f"{rank}. {cands[idx-1]}  score={sc:.3f}")


if __name__ == "__main__":
    main()
