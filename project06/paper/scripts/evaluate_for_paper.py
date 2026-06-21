#!/usr/bin/env python3
"""論文用の評価スクリプト。

project06 (母音列 -> 文 の生成SFT) の推論結果を JSONL で受け取り、
候補提示精度(Acc@k)と入力効率(KSPC)を計算する。

KSPC は提案手法(母音入力)に加え、既存IME方式(ローマ字=訓令式 / かな入力 / フリック)の
理想KSPC(誤りなしの最小打鍵数 / 正解文字数)も同時に算出し、横並び比較できるようにする。
分母はどの方式も素の正解文字数 len(gold) で共通。数字・英字を含む文は除外する(追加)。

入力 JSONL の想定形式(1行1サンプル):
  {
    "input_vowels": "a i a o u",        # 母音列(空白区切り)
    "gold": "ありがとう",               # 正解文(または正解読み)
    "candidates": ["ありがとう", ...],  # モデルが提示した候補(順位順、上位ほど先頭)
    "scores": [0.91, 0.12, 0.08]        # (任意)各候補のスコア。無くても順序で評価可
  }

project06 の infer.py の predict() は候補文の list を順位順で返すため、
それを上記形式に整形して保存すれば本スクリプトで評価できる。
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys

# 追加: 既存IMEのKSPC算出に gold の読み(ひらがな)が要るので project06 ルートの vowel_utils を使う
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from vowel_utils import _kks  # noqa: E402  (text_to_vowel_str と同じ pykakasi 経路)


# ---------- 入出力 ----------

def open_maybe_gz(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_records(path: str):
    with open_maybe_gz(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_record(rec: dict) -> dict:
    """複数の出力形式を吸収して {input_vowels, gold, candidates, scores} に揃える。"""
    vowels = rec.get("input_vowels", rec.get("vowels", ""))
    gold = rec.get("gold", rec.get("sentence", rec.get("answer", "")))
    cands = rec.get("candidates", rec.get("cands", rec.get("preds", [])))
    scores = rec.get("scores", None)
    # scores があれば降順に並べ替えて candidates の順位を確定させる
    if scores is not None and len(scores) == len(cands):
        order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
        cands = [cands[i] for i in order]
        scores = [scores[i] for i in order]
    return {"input_vowels": vowels, "gold": gold, "candidates": list(cands), "scores": scores}


# ---------- KSPC ----------

def count_vowel_keys(vowels: str) -> int:
    """母音列 (空白区切り a/i/u/e/o/n) のキー打鍵数 = トークン数。"""
    return len([t for t in vowels.split() if t])


# 数字・英字を含む文は母音化で脱落しKSPCが不当に低く出るため評価対象から除外する(追加)
_RE_ALNUM = re.compile(r"[0-9A-Za-z\uFF10-\uFF19\uFF21-\uFF3A\uFF41-\uFF5A]")


def has_alnum(text: str) -> bool:
    return bool(_RE_ALNUM.search(text))


def reading_hira(text: str) -> str:
    """gold の読み(ひらがな)を text_to_vowel_str と同じ pykakasi 経路で得る(追加)。"""
    return "".join(tok["hira"] for tok in _kks.convert(text))


# 既存IME各方式の理想打鍵数モデル。入力は gold の読み(ひらがな)。
# どの方式も誤りなしの最小打鍵数を数える(分母は別途 len(gold) で共通)。

_KUNREI_ROMAJI = {  # 訓令式: し->si, ち->ti, つ->tu, ふ->hu, じ->zi など
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "さ": "sa", "し": "si", "す": "su", "せ": "se", "そ": "so",
    "ざ": "za", "じ": "zi", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "た": "ta", "ち": "ti", "つ": "tu", "て": "te", "と": "to",
    "だ": "da", "ぢ": "di", "づ": "du", "で": "de", "ど": "do",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "hu", "へ": "he", "ほ": "ho",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "ゐ": "wi", "ゑ": "we", "を": "wo", "ん": "n",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
    "ゔ": "vu",
}
_KUNREI_YOON = {  # 拗音: 訓令式は kya/sya/tya/zya...
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
    "しゃ": "sya", "しゅ": "syu", "しょ": "syo",
    "じゃ": "zya", "じゅ": "zyu", "じょ": "zyo",
    "ちゃ": "tya", "ちゅ": "tyu", "ちょ": "tyo",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo",
    "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo",
    "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo",
    "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
}
_DAKUON = set("がぎぐげござじずぜぞだぢづでどばびぶべぼゔ")  # 濁音(かな入力では濁点1打鍵を加算)
_HANDAKU = set("ぱぴぷぺぽ")                                  # 半濁音(半濁点1打鍵を加算)


def count_romaji_kunrei_keys(hira: str) -> int:
    """訓令式ローマ字入力の打鍵数。促音っは次の子音重ね(+1)、長音ーは母音字(+1)。"""
    keys = 0
    i = 0
    while i < len(hira):
        pair = hira[i:i + 2]
        if pair in _KUNREI_YOON:
            keys += len(_KUNREI_YOON[pair])
            i += 2
            continue
        ch = hira[i]
        if ch == "っ":
            keys += 1  # 次の子音を重ねる1打鍵
            i += 1
            continue
        if ch == "ー":
            keys += 1  # 直前母音字をもう1打鍵
            i += 1
            continue
        r = _KUNREI_ROMAJI.get(ch)
        if r is not None:
            keys += len(r)
        i += 1
    return keys


def count_kana_keys(hira: str) -> int:
    """JISかな入力の打鍵数。かな1つ=1キー、濁音/半濁音は濁点/半濁点で+1、拗音は2キー。"""
    keys = 0
    for ch in hira:
        if ch in _DAKUON:
            keys += 2  # 清音キー + 濁点
        elif ch in _HANDAKU:
            keys += 2  # 清音キー + 半濁点
        else:
            keys += 1  # 清音/拗音の小書き/っ/ー/ん すべて1キー
    return keys


def count_flick_keys(hira: str) -> int:
    """フリック入力のタップ数。各かな1タップ(行キー＋方向フリックで1操作)、
    濁音/半濁音は濁点/半濁点キーで+1、拗音は小書きで+1。"""
    keys = 0
    for ch in hira:
        if ch in _DAKUON or ch in _HANDAKU:
            keys += 2  # 清音タップ + 濁点/半濁点キー
        else:
            keys += 1
    return keys


# トグル入力(ガラケー打ち)用: 各かなが行内で母音段の何番目か = 連打回数。
# あ段=1, い段=2, う段=3, え段=4, お段=5。vowel_utils の母音判定を流用する。
_VOWEL_TO_TAPS = {"a": 1, "i": 2, "u": 3, "e": 4, "o": 5}
# 小書き(母音マップに無い)の連打回数。小書きは「や/わ行キー」等を連打して出す想定。
_SMALL_TO_TAPS = {"ゃ": 1, "ゅ": 3, "ょ": 5, "ぁ": 1, "ぃ": 2, "ぅ": 3, "ぇ": 4, "ぉ": 5, "ゎ": 1}


def count_toggle_keys(hira: str) -> int:
    """トグル入力(12キー連打)の打鍵数。各かなを母音段の回数だけ連打する。
    あ=1,い=2,う=3,え=4,お=5回。濁音/半濁音は清音を連打後に濁点キーで+1。
    拗音の小書き(ゃゅょ)も連打で出す。ん=1, っ/ー=1。"""
    from vowel_utils import hira_to_vowels_strict  # 追加: 各かなの母音段を取得

    keys = 0
    for ch in hira:
        if ch == "ん":
            keys += 1  # 「わ」行キーの撥音(おおむね1打鍵)
            continue
        if ch in ("っ", "ー"):
            keys += 1  # 促音/長音(小書き変換・長音キーで1打鍵)
            continue
        if ch in _SMALL_TO_TAPS:
            keys += _SMALL_TO_TAPS[ch]
            continue
        vs = hira_to_vowels_strict(ch)  # 清音/濁音1文字 -> 母音1個
        if vs:
            keys += _VOWEL_TO_TAPS[vs[0]]
        if ch in _DAKUON or ch in _HANDAKU:
            keys += 1  # 濁点/半濁点キー
    return keys


# ---------- メトリクス本体 ----------

def evaluate(records, ks=(1, 3, 5, 10)):
    n = 0
    skipped_alnum = 0  # 追加: 数字英字を含み除外した件数
    acc_at = {k: 0 for k in ks}
    gold_chars = 0
    keys_sum = {"proposed": 0, "romaji_kunrei": 0, "kana": 0, "flick": 0, "toggle": 0}

    for rec in records:
        rec = normalize_record(rec)
        gold = rec["gold"]
        cands = rec["candidates"]
        vowels = rec["input_vowels"]

        # 数字・英字を含む文は母音化で脱落するため評価対象から除外(追加)
        if has_alnum(gold):
            skipped_alnum += 1
            continue
        n += 1

        # rank: 正解が候補の何位か(1-based)。無ければ None
        rank = None
        for i, c in enumerate(cands, start=1):
            if c == gold:
                rank = i
                break

        for k in ks:
            if rank is not None and rank <= k:
                acc_at[k] += 1

        # KSPC: 分母は全方式 len(gold) 共通、分子は各方式の理想打鍵数
        gold_chars += len(gold)
        hira = reading_hira(gold)
        keys_sum["proposed"] += count_vowel_keys(vowels)
        keys_sum["romaji_kunrei"] += count_romaji_kunrei_keys(hira)
        keys_sum["kana"] += count_kana_keys(hira)
        keys_sum["flick"] += count_flick_keys(hira)
        keys_sum["toggle"] += count_toggle_keys(hira)

    if n == 0:
        raise SystemExit("no records")

    def kspc(name):
        return keys_sum[name] / gold_chars if gold_chars else float("nan")

    metrics = {
        "N": n,
        "skipped_alnum": skipped_alnum,
        **{f"Acc@{k}": acc_at[k] / n for k in ks},
        "KSPC_proposed": kspc("proposed"),          # 母音キー数 / 正解文字数
        "KSPC_romaji_kunrei": kspc("romaji_kunrei"),  # 訓令式ローマ字IME
        "KSPC_kana": kspc("kana"),                    # JISかな入力
        "KSPC_flick": kspc("flick"),                  # フリック入力
        "KSPC_toggle": kspc("toggle"),                # トグル入力(12キー連打)
    }
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_path", required=True, help="推論結果 JSONL(.gz可)")
    ap.add_argument("--out_dir", default="../results", help="メトリクスの出力先")
    ap.add_argument("--ks", default="1,3,5,10")
    ap.add_argument("--tag", default="proposed", help="手法名(出力ファイル名に付与)")
    args = ap.parse_args()

    ks = tuple(int(x) for x in args.ks.split(","))
    os.makedirs(args.out_dir, exist_ok=True)

    records = list(iter_records(args.pred_path))
    metrics = evaluate(records, ks=ks)

    # メトリクス JSON
    metrics_path = os.path.join(args.out_dir, f"metrics_{args.tag}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:30s}: {v:.4f}")
        else:
            print(f"{k:30s}: {v}")
    print("=" * 60)
    print("metrics ->", metrics_path)


if __name__ == "__main__":
    main()
