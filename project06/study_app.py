#!/usr/bin/env python3
"""実ユーザ実験(認知負荷評価)用の計測 Web アプリ。

スマホ/PC のブラウザから LAN 経由でアクセスし、以下を1つの画面で行う:
  - 名前入力
  - 4モード: IMEモード(Level1-3) / 母音モード(Level1-3) / IMEアンケート / 母音アンケート
  - 課題文を一定時間視認 -> 隠す -> 入力(記憶保持負荷を課す)
  - 各試行の客観指標(初動時間/総入力時間/打鍵数/Backspace数/打鍵間隔/母音化CER 等)を CSV 保存
  - アンケート(母音7項目 / IME3項目)を CSV 保存

母音モードの候補生成は project06 の LoRA モデルを使う(GPU 必要)。
モデルが読み込めない環境でも IME モード・アンケート・計測は動くよう、
母音候補 API はモデル未ロード時にエラーを返すだけにしている(デバッグ用)。

起動:
  source /home/shunsuke/Desktop/venv/pytorch_gpu/bin/activate
  LORA_DIR=logs/train_20260611_134432 uvicorn study_app:app --host 0.0.0.0 --port 8001
  # 母音モデルを使わない(IME/アンケートのみ)場合: STUDY_NO_MODEL=1 を付ける
"""
from __future__ import annotations

import csv
import datetime
import os
import re
import threading
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from vowel_utils import text_to_vowel_str  # 母音化CER算出 / 正解母音列の提示に使用

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.join(HERE, "study")
SENTENCES_CSV = os.path.join(STUDY_DIR, "sentences.csv")

_write_lock = threading.Lock()  # 追加: 複数端末から同時POSTされてもCSVが壊れないように


def subject_dir(name: str, session: str = "") -> str:
    """測定セッションごとに study/<セッション日時>_<名前>/ を作って返す(測定ごとにログ分離)。追加。
    session(フロントがページを開いた時刻 YYYY-MM-DD_HH-MM)があればそれを使い、
    無ければ現在時刻(分まで)を使う。"""
    safe = re.sub(r"[^\w\-]", "_", (name or "anon").strip()) or "anon"
    sess = re.sub(r"[^\w\-]", "_", (session or "").strip())  # 追加: セッション識別子の安全化
    if not sess:
        sess = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    d = os.path.join(STUDY_DIR, f"{sess}_{safe}")
    os.makedirs(d, exist_ok=True)
    return d

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---- モデル(母音モードの候補生成。IME/アンケートだけなら未ロードでも可) ----
MODEL = None
TOK = None


def _try_load_model():
    """母音モードの候補生成モデルをロード。失敗しても起動は続行する。"""
    global MODEL, TOK
    if os.environ.get("STUDY_NO_MODEL"):
        print("[study_app] STUDY_NO_MODEL set -> 母音候補APIは無効(IME/アンケートのみ)")
        return
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_model = os.environ.get("BASE_MODEL", "openbmb/MiniCPM4-0.5B")
        lora_dir = os.environ.get("LORA_DIR", "lora_vowel_out")
        revision = os.environ.get("REVISION", "2aaa97c53d")
        print(f"[study_app] loading model base={base_model} lora={lora_dir}")
        TOK = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, revision=revision)
        if TOK.pad_token is None:
            TOK.pad_token = TOK.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, revision=revision,
        )
        MODEL = PeftModel.from_pretrained(base, lora_dir)
        MODEL.eval()
        print("[study_app] model ready")
    except Exception as e:  # デバッグ: ロード失敗の理由を残す
        print(f"[study_app] model load skipped: {e}")


@app.on_event("startup")
def _startup():
    os.makedirs(STUDY_DIR, exist_ok=True)
    _try_load_model()


# ---- 課題文 ----
def load_sentences() -> List[dict]:
    if not os.path.exists(SENTENCES_CSV):
        return []
    rows = []
    with open(SENTENCES_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["level"] = int(r["level"])
            r["n_chars"] = int(r["n_chars"])
            r["alts"] = r.get("alts", "")  # 追加: 許容表記(| 区切り)。無ければ空。
            rows.append(r)
    return rows


@app.get("/api/sentences")
def api_sentences(level: int):
    rows = [r for r in load_sentences() if r["level"] == level]
    if not rows:
        raise HTTPException(404, f"level {level} の課題文がありません(make_study_sentences.py を実行)")
    return {"level": level, "sentences": rows}


# ---- 母音候補生成(母音モード) ----
class PredictReq(BaseModel):
    vowels: str
    ctx: Optional[str] = None
    k: int = 8


@app.post("/api/predict")
def api_predict(req: PredictReq):
    if MODEL is None or TOK is None:
        raise HTTPException(503, "母音モデル未ロード(LORA_DIR を指定して起動してください)")
    import torch
    from train_vowel_lora import build_prompt, PROMPT_CTX_NONE

    ctx_norm = req.ctx.strip() if req.ctx and req.ctx.strip() else PROMPT_CTX_NONE
    prompt = build_prompt(req.vowels, ctx_norm) + "\n"
    enc = TOK(prompt, return_tensors="pt").to(MODEL.device)
    with torch.no_grad():
        out = MODEL.generate(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=64, num_beams=max(8, req.k), num_return_sequences=max(8, req.k),
            do_sample=False, early_stopping=True, pad_token_id=TOK.pad_token_id, use_cache=False,
        )
    gen = out[:, enc["input_ids"].shape[1]:]
    seen, cands = set(), []
    for row in gen:
        text = TOK.decode(row, skip_special_tokens=True).strip().split("\n")[0].strip()
        if text and text not in seen:
            seen.add(text)
            cands.append(text)
    return {"vowels": req.vowels, "candidates": cands[:req.k]}


# ---- 試行ログ保存 ----
class TrialReq(BaseModel):
    name: str
    mode: str            # "ime" or "vowel"
    level: int
    sentence_id: str
    gold: str            # 提示した課題文(正解)
    typed: str           # 被験者が確定した文(IME) / 選んだ候補(母音)
    typed_vowels: str = ""   # 母音モードで被験者が打った母音列
    t_init_ms: float = 0     # 課題文を隠した後〜最初の打鍵まで(母音変換負荷の代理)
    t_total_ms: float = 0    # 入力開始〜確定まで
    keystrokes: int = 0      # 打鍵数
    backspaces: int = 0      # 修正回数
    mean_iki_ms: float = 0   # 平均打鍵間隔(記憶保持負荷の代理)
    selected_rank: int = -1  # 母音モードで選んだ候補の順位(1始まり, -1=該当なし)
    success: int = 0         # typed == gold か
    session: str = ""        # 追加: 測定セッション識別子(フロントがページを開いた時刻)


TRIAL_FIELDS = [
    "timestamp", "name", "mode", "level", "sentence_id", "gold", "typed",
    "typed_vowels", "gold_vowels", "vowel_cer",
    "t_init_ms", "t_total_ms", "keystrokes", "backspaces", "mean_iki_ms",
    "selected_rank", "success",
]


def _char_cer(hyp: str, ref: str) -> float:
    """母音列(空白区切りトークン列)の編集距離 / 参照長。母音変換誤り率に使う。"""
    a, b = hyp.split(), ref.split()
    n, m = len(a), len(b)
    if m == 0:
        return 0.0 if n == 0 else 1.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m] / m


@app.post("/api/trial")
def api_trial(req: TrialReq):
    gold_vowels = text_to_vowel_str(req.gold)
    vowel_cer = _char_cer(req.typed_vowels, gold_vowels) if req.mode == "vowel" else ""
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "name": req.name, "mode": req.mode, "level": req.level,
        "sentence_id": req.sentence_id, "gold": req.gold, "typed": req.typed,
        "typed_vowels": req.typed_vowels, "gold_vowels": gold_vowels, "vowel_cer": vowel_cer,
        "t_init_ms": round(req.t_init_ms, 1), "t_total_ms": round(req.t_total_ms, 1),
        "keystrokes": req.keystrokes, "backspaces": req.backspaces,
        "mean_iki_ms": round(req.mean_iki_ms, 1),
        "selected_rank": req.selected_rank, "success": req.success,
    }
    with _write_lock:
        path = os.path.join(subject_dir(req.name, req.session), "results_trials.csv")
        new = not os.path.exists(path)
        with open(path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRIAL_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
    return {"ok": True}


# ---- アンケート保存 ----
class SurveyReq(BaseModel):
    name: str
    mode: str               # "vowel"(7項目) or "ime"(3項目)
    answers: dict           # {"q1":5, ...} 7段階
    session: str = ""       # 追加: 測定セッション識別子


# 母音7項目 / IME3項目の質問文(論文・集計と対応)
VOWEL_QUESTIONS = {
    "q1_vowel_convert": "提示された文章を母音列に変換するのは難しかった",
    "q2_memory": "母音列を覚えながら入力するのは難しかった",
    "q3_operation": "入力操作そのものは難しかった",
    "q4_frustration": "入力中に混乱やストレスを感じた",
    "q5_fatigue": "入力後に疲れを感じた",
    "q6_usefulness": "通常の日本語入力より効率的だと感じた",
    "q7_intention": "日常的に利用したいと思った",
}
IME_QUESTIONS = {
    "q3_operation": "入力操作そのものは難しかった",
    "q4_frustration": "入力中に混乱やストレスを感じた",
    "q5_fatigue": "入力後に疲れを感じた",
}


@app.post("/api/survey")
def api_survey(req: SurveyReq):
    keys = list((VOWEL_QUESTIONS if req.mode == "vowel" else IME_QUESTIONS).keys())
    fields = ["timestamp", "name", "mode"] + keys
    row = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
           "name": req.name, "mode": req.mode}
    for k in keys:
        row[k] = req.answers.get(k, "")
    with _write_lock:
        # mode ごとに列が違うので被験者フォルダ内で mode 別ファイルに分ける
        path = os.path.join(subject_dir(req.name, req.session), f"results_survey_{req.mode}.csv")
        new = not os.path.exists(path)
        with open(path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)
    return {"ok": True}


@app.get("/api/questions")
def api_questions():
    return {"vowel": VOWEL_QUESTIONS, "ime": IME_QUESTIONS}


# ---- 解析(標準IME vs 母音(推論なし) の t_total_ms 平均比較) ----
@app.get("/api/analyze")
def api_analyze(name: str, session: str = ""):
    """被験者の results_trials.csv から ime と vowel_noinfer の success=1 の t_total_ms 平均を返す。"""
    path = os.path.join(subject_dir(name, session), "results_trials.csv")
    metrics = ["t_total_ms", "t_init_ms", "mean_iki_ms"]  # 追加: t_init_ms, mean_iki_ms も集計
    sums = {m: {"ime": 0.0, "vowel_noinfer": 0.0} for m in metrics}
    counts = {"ime": 0, "vowel_noinfer": 0}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["mode"] in counts and r["success"] == "1":
                    for m in metrics:
                        sums[m][r["mode"]] += float(r[m])
                    counts[r["mode"]] += 1

    def mean(m, mode):
        return sums[m][mode] / counts[mode] if counts[mode] else 0.0

    return {
        "ime_n": counts["ime"], "vowel_n": counts["vowel_noinfer"],
        "ime_mean": mean("t_total_ms", "ime"), "vowel_mean": mean("t_total_ms", "vowel_noinfer"),
        "ime_init_mean": mean("t_init_ms", "ime"), "vowel_init_mean": mean("t_init_ms", "vowel_noinfer"),
        "ime_iki_mean": mean("mean_iki_ms", "ime"), "vowel_iki_mean": mean("mean_iki_ms", "vowel_noinfer"),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(STUDY_DIR, "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(500, "study/index.html がありません")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8001")))
