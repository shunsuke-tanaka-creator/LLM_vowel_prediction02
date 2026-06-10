from __future__ import annotations

import json
from typing import List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from morph_utils import extract_content_words_with_pos  # 変更: 原文位置付き抽出(学習データCTXと整合)

PROMPT_CTX_NONE = "<NONE>"

app = FastAPI()


class Req(BaseModel):
    ctx: Optional[str] = None
    vowels: str
    k: int = 5
    num_cands: int = 32
    ctx_len: int = 4


class Resp(BaseModel):
    best: str
    candidates: List[str]


def normalize_ctx(ctx: Optional[str], ctx_len: int) -> str:
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
    return "\n".join(lines)


# --- globals (load once) ---
MODEL = None
TOK = None
V2 = None


# コメントに追加: MiniCPM4 用に trust_remote_code + revision pin (transformers 4.46.3 互換のため 2aaa97c53d を pin)
def load_all(base_model: str, lora_dir: str, vowel2cands_path: str, revision: str = "2aaa97c53d"):
    global MODEL, TOK, V2
    with open(vowel2cands_path, "r", encoding="utf-8") as f:
        V2 = json.load(f)

    TOK = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, revision=revision)
    if TOK.pad_token is None:
        TOK.pad_token = TOK.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        revision=revision,
    )
    MODEL = PeftModel.from_pretrained(base, lora_dir)
    MODEL.eval()


# コメントに追加: 多桁候補番号(例 10〜32)を区別するため、prompt + str(i) のトークン列の対数尤度を合算するスコア。infer_ime.py から移植。
@torch.no_grad()
def score_candidate(prompt_ids, num_str: str) -> float:
    cont_ids = TOK(num_str, add_special_tokens=False)["input_ids"]
    if not cont_ids:
        return float("-inf")
    full_ids = torch.cat(
        [prompt_ids, torch.tensor([cont_ids], device=prompt_ids.device)], dim=1
    )
    out = MODEL(input_ids=full_ids, use_cache=False)  # コメントに追加: MiniCPM4 リモートコードと現 transformers の互換性確保のため use_cache=False
    logits = out.logits[0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    base = prompt_ids.shape[1] - 1
    total = 0.0
    for k, tid in enumerate(cont_ids):
        total += log_probs[base + k, tid].item()
    return total


@torch.no_grad()
def topk_words(ctx_norm: str, vowels: str, k: int, num_cands: int) -> List[str]:
    if vowels not in V2:
        return []
    cands = V2[vowels][:num_cands]
    prompt = build_prompt(ctx_norm, vowels, cands)

    prompt_ids = TOK(prompt, return_tensors="pt").to(MODEL.device)["input_ids"]
    scored = []
    for i in range(1, len(cands) + 1):
        s = score_candidate(prompt_ids, str(i))
        scored.append((i, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = [cands[i - 1] for (i, _) in scored[:k]]
    return top


@app.on_event("startup")
def _startup():
    load_all(
        base_model="openbmb/MiniCPM4-0.5B",
        lora_dir="project02/lora_smoke_out",
        vowel2cands_path="vowel2cands.json",
    )


@app.post("/predict", response_model=Resp)
def predict(req: Req):
    ctx_norm = normalize_ctx(req.ctx, req.ctx_len)
    cands = topk_words(ctx_norm, req.vowels, req.k, req.num_cands)
    if not cands:
        return Resp(best="", candidates=[])
    return Resp(best=cands[0], candidates=cands)
