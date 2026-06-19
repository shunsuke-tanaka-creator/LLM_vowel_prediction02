from __future__ import annotations

import os
from typing import List, Optional

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # 追加: ブラウザGUI(project07)からの直接アクセス用
from fastapi.responses import FileResponse  # 追加: GET / で GUI(index.html) を配信
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_vowel_lora import build_prompt, PROMPT_CTX_NONE  # 学習と完全一致させるため共通利用

app = FastAPI()

# 追加: project07 の単一HTML GUI からブラウザ直アクセスできるよう CORS を全許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Req(BaseModel):
    vowels: str
    ctx: Optional[str] = None
    k: int = 8
    num_beams: int = 8
    max_new_tokens: int = 64


class Resp(BaseModel):
    vowels: str
    candidates: List[str]  # 復元文候補


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
def topk_sentences(vowels: str, ctx: str, k: int, num_beams: int, max_new_tokens: int) -> List[str]:
    ctx_norm = ctx.strip() if ctx and ctx.strip() else PROMPT_CTX_NONE
    prompt = build_prompt(vowels, ctx_norm) + "\n"  # 学習時の prompt(SENTENCE:\n まで)と一致
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
        lora_dir=os.environ.get("LORA_DIR", "lora_vowel_out"),
    )


# 追加: project07 の GUI(単一HTML)へのパス(serve_api.py からの相対)
GUI_HTML = os.path.join(os.path.dirname(__file__), "..", "project07", "index.html")


@app.get("/")  # 追加: ブラウザでサーバーURLを開くと GUI を返す(別PCからも同一オリジンでアクセス可)
def index():
    return FileResponse(GUI_HTML)


@app.post("/predict", response_model=Resp)
def predict(req: Req):
    cands = topk_sentences(req.vowels, req.ctx, req.k, req.num_beams, req.max_new_tokens)
    return Resp(vowels=req.vowels, candidates=cands)
