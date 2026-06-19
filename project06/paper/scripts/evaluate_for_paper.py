#!/usr/bin/env python3
"""論文用の評価スクリプト。

project06 (母音列 -> 文 の生成SFT) の推論結果を JSONL で受け取り、
候補生成・順位付け性能(Acc@k, MRR, EM, CER, WER)と入力効率(KSPC, 入力削減率)を計算する。

入力 JSONL の想定形式(1行1サンプル):
  {
    "input_vowels": "a i a o u",        # 母音列(空白区切り)
    "gold": "ありがとう",               # 正解文(または正解読み)
    "candidates": ["ありがとう", ...],  # モデルが提示した候補(順位順、上位ほど先頭)
    "scores": [0.91, 0.12, 0.08]        # (任意)各候補のスコア。無くても順序で評価可
  }

project06 の eval_restore.py / infer.py の predict() は候補文の list を順位順で返すため、
それを上記形式に整形して保存すれば本スクリプトで評価できる。

CER/WER は jiwer が入っていれば使用し、無ければ内蔵の編集距離で計算する(MeCab等は不要)。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from typing import List, Optional


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


# ---------- 編集距離 / CER / WER ----------

def edit_distance(a: List[str], b: List[str]):
    """Levenshtein 距離と S/D/I の内訳を返す。a=hyp, b=ref。"""
    n, m = len(a), len(b)
    # dp[i][j] = (cost, S, D, I)
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, 0, i)  # 全削除(hyp 余り = Insertion 扱い)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, j, 0)  # 全挿入(ref 余り = Deletion 扱い)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                c_sub = dp[i - 1][j - 1][0] + 1
                c_del = dp[i][j - 1][0] + 1      # ref を消費 = Deletion
                c_ins = dp[i - 1][j][0] + 1       # hyp を消費 = Insertion
                best = min(c_sub, c_del, c_ins)
                if best == c_sub:
                    pc, s, d, ins = dp[i - 1][j - 1]
                    dp[i][j] = (best, s + 1, d, ins)
                elif best == c_del:
                    pc, s, d, ins = dp[i][j - 1]
                    dp[i][j] = (best, s, d + 1, ins)
                else:
                    pc, s, d, ins = dp[i - 1][j]
                    dp[i][j] = (best, s, d, ins + 1)
    return dp[n][m]


def cer(hyp: str, ref: str) -> float:
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    dist, _, _, _ = edit_distance(list(hyp), list(ref))
    return dist / len(ref)


def wer(hyp: str, ref: str) -> float:
    # 日本語は空白区切りが無いので、ここでは文字 n-gram ではなく素朴に空白分割(英数字混在時に意味を持つ)。
    # 純日本語文では WER==CER に近くなる点に注意(本文でも明記)。
    ref_w = ref.split()
    hyp_w = hyp.split()
    if len(ref_w) <= 1 and len(hyp_w) <= 1:
        # 空白が無い純日本語: 文字単位にフォールバック
        ref_w = list(ref)
        hyp_w = list(hyp)
    if len(ref_w) == 0:
        return 0.0 if len(hyp_w) == 0 else 1.0
    dist, _, _, _ = edit_distance(hyp_w, ref_w)
    return dist / len(ref_w)


# ---------- KSPC ----------

def count_vowel_keys(vowels: str) -> int:
    """母音列 (空白区切り a/i/u/e/o/n) のキー打鍵数 = トークン数。"""
    return len([t for t in vowels.split() if t])


def romaji_keystrokes_estimate(reading_hira: Optional[str]) -> Optional[int]:
    """ローマ字入力の打鍵数のおおまかな推定。
    読み(ひらがな)が与えられればモーラ数 x 約2打鍵で概算する。
    無ければ None を返す(本文では [要確認] / 実測KSPCで差し替え)。"""
    if not reading_hira:
        return None
    return len(reading_hira) * 2  # TODO: 実測ベースに差し替え推奨


# ---------- メトリクス本体 ----------

def evaluate(records, ks=(1, 3, 5, 10)):
    n = 0
    acc_at = {k: 0 for k in ks}
    rr_sum = 0.0
    em = 0
    cer_sum = 0.0
    wer_sum = 0.0
    vowel_keys_sum = 0
    in_set = 0  # 正解が候補集合に含まれた件数
    error_rows = []  # エラー分析用

    for rec in records:
        rec = normalize_record(rec)
        gold = rec["gold"]
        cands = rec["candidates"]
        vowels = rec["input_vowels"]
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
        if rank is not None:
            rr_sum += 1.0 / rank
            in_set += 1
        # EM = top1 完全一致
        top1 = cands[0] if cands else ""
        if top1 == gold:
            em += 1
        # CER/WER は top1 候補に対して計算
        c = cer(top1, gold)
        w = wer(top1, gold)
        cer_sum += c
        wer_sum += w
        vowel_keys_sum += count_vowel_keys(vowels)

        # エラー分析: 正解が候補集合に無い / 低順位 / top1 不一致 を記録
        if rank is None:
            etype = "not_in_candidates"
        elif rank > 5:
            etype = "low_rank"
        elif rank > 1:
            etype = "rank2-5"
        elif top1 != gold:
            etype = "top1_mismatch"
        else:
            etype = "correct"
        error_rows.append({
            "vowels": vowels,
            "gold": gold,
            "top1": top1,
            "rank": rank if rank is not None else -1,
            "cer_top1": round(c, 4),
            "wer_top1": round(w, 4),
            "error_type": etype,
            "n_chars_gold": len(gold),
        })

    if n == 0:
        raise SystemExit("no records")

    gold_chars = sum(r["n_chars_gold"] for r in error_rows)
    kspc = vowel_keys_sum / gold_chars if gold_chars else float("nan")

    metrics = {
        "N": n,
        **{f"Acc@{k}": acc_at[k] / n for k in ks},
        "MRR": rr_sum / n,
        "ExactMatch": em / n,
        "CER": cer_sum / n,
        "WER": wer_sum / n,
        "KSPC_proposed": kspc,  # 母音キー数 / 正解文字数
        "answer_in_candidate_set_rate": in_set / n,
    }
    return metrics, error_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_path", required=True, help="推論結果 JSONL(.gz可)")
    ap.add_argument("--out_dir", default="../results", help="メトリクス/CSVの出力先")
    ap.add_argument("--ks", default="1,3,5,10")
    ap.add_argument("--tag", default="proposed", help="手法名(出力ファイル名に付与)")
    args = ap.parse_args()

    ks = tuple(int(x) for x in args.ks.split(","))
    os.makedirs(args.out_dir, exist_ok=True)

    records = list(iter_records(args.pred_path))
    metrics, error_rows = evaluate(records, ks=ks)

    # メトリクス JSON
    metrics_path = os.path.join(args.out_dir, f"metrics_{args.tag}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # エラー分析 CSV
    csv_path = os.path.join(args.out_dir, f"errors_{args.tag}.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(error_rows[0].keys()))
        writer.writeheader()
        writer.writerows(error_rows)

    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:30s}: {v:.4f}")
        else:
            print(f"{k:30s}: {v}")
    print("=" * 60)
    print("metrics ->", metrics_path)
    print("errors  ->", csv_path)


if __name__ == "__main__":
    main()
