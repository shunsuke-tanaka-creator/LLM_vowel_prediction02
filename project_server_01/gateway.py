from __future__ import annotations

import os
from typing import List, Optional

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

# 02(母音→単語) と 05(語幹→語尾) の転送先。環境変数で上書き可能。
WORD_API_URL = os.environ.get("WORD_API_URL", "http://127.0.0.1:8001")
SUFFIX_API_URL = os.environ.get("SUFFIX_API_URL", "http://127.0.0.1:8000")

app = FastAPI()


class WordReq(BaseModel):
    vowels: str
    ctx: Optional[str] = None
    k: int = 32


class WordResp(BaseModel):
    best: str
    candidates: List[str]


class SuffixReq(BaseModel):
    stem: str
    ctx: Optional[str] = None
    k: int = 32


class SuffixResp(BaseModel):
    stem: str
    candidates: List[str]
    words: List[str]


@app.post("/predict_word", response_model=WordResp)
def predict_word(req: WordReq):
    # 02 の /predict は {ctx, vowels, k} を受け取り {best, candidates} を返す
    payload = {"vowels": req.vowels, "ctx": req.ctx, "k": req.k}
    r = httpx.post(f"{WORD_API_URL}/predict", json=payload, timeout=60.0)
    r.raise_for_status()
    return r.json()


@app.post("/predict_suffix", response_model=SuffixResp)
def predict_suffix(req: SuffixReq):
    # 05 の /predict は {stem, ctx, k} を受け取り {stem, candidates, words} を返す
    payload = {"stem": req.stem, "ctx": req.ctx, "k": req.k}
    r = httpx.post(f"{SUFFIX_API_URL}/predict", json=payload, timeout=60.0)
    r.raise_for_status()
    return r.json()


@app.get("/health")
def health():
    # 02/05 双方の疎通を確認(到達できれば up)
    status = {}
    for name, url in (("word", WORD_API_URL), ("suffix", SUFFIX_API_URL)):
        try:
            httpx.get(f"{url}/docs", timeout=5.0)
            status[name] = "up"
        except Exception as e:
            status[name] = f"down: {e}"
    return status
