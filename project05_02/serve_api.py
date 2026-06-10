from __future__ import annotations

import os
from typing import List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_CTX_NONE = "<NONE>"

app = FastAPI()


class Req(BaseModel):
    stem: str
    ctx: Optional[str] = None
    k: int = 8
    num_beams: int = 8
    max_new_tokens: int = 12


class Resp(BaseModel):
    stem: str
    candidates: List[str]  # 語尾候補(語幹は含まない)
    words: List[str]       # 語幹+語尾の結合形


# train_suffix_lora.build_prompt と完全一致させること
def build_prompt(stem: str, ctx: str) -> str:
    lines = []
    lines.append("あなたは日本語IMEの語尾予測器です。")
    lines.append("語幹に続く自然な語尾だけを出力してください。")
    lines.append(f"CTX: {ctx}")
    lines.append(f"STEM: {stem}")
    lines.append("SUFFIX:")
    return "\n".join(lines)


MODEL = None
TOK = None


# MiniCPM4 用に trust_remote_code + revision pin (transformers 4.46.3 互換のため 2aaa97c53d を pin)
def load_all(base_model: str, lora_dir: str, revision: str = "2aaa97c53d"):
    global MODEL, TOK
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


@torch.no_grad()
def topk_suffix(stem: str, ctx: str, k: int, num_beams: int, max_new_tokens: int) -> List[str]:
    ctx_norm = ctx.strip() if ctx and ctx.strip() else PROMPT_CTX_NONE
    prompt = build_prompt(stem, ctx_norm)
    enc = TOK(prompt, return_tensors="pt").to(MODEL.device)

    out = MODEL.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=max_new_tokens,
        num_beams=max(num_beams, k),
        num_return_sequences=max(num_beams, k),
        do_sample=False,
        early_stopping=True,
        pad_token_id=TOK.pad_token_id,
        use_cache=False,  # MiniCPM4 リモートコード互換のため
    )
    gen = out[:, enc["input_ids"].shape[1]:]
    seen = set()
    cands: List[str] = []
    for row in gen:
        text = TOK.decode(row, skip_special_tokens=True).strip().split("\n")[0].strip()
        if text and text not in seen:
            seen.add(text)
            cands.append(text)
    return cands[:k]


@app.on_event("startup")
def _startup():
    load_all(
        base_model=os.environ.get("BASE_MODEL", "openbmb/MiniCPM4-0.5B"),
        lora_dir=os.environ.get("LORA_DIR", "lora_suffix_out"),
    )


@app.post("/predict", response_model=Resp)
def predict(req: Req):
    cands = topk_suffix(req.stem, req.ctx, req.k, req.num_beams, req.max_new_tokens)
    words = [req.stem + c for c in cands]
    return Resp(stem=req.stem, candidates=cands, words=words)
