#!/usr/bin/env python3
"""論文用の図を生成する。

paper/results/metrics_*.json が存在すればそれを読み、無ければ
スクリプト内の SAMPLE_* (仮値) を使って動作確認用の図を出す。
本文中の数値は XXX のままにし、確定後に図を差し替える運用。

生成図(paper/figures/ に PNG):
  fig_topk_accuracy.png      Top-k Accuracy 比較棒グラフ
  fig_cer_wer.png            CER / WER 比較棒グラフ
  fig_mrr.png                MRR 比較棒グラフ
  fig_kspc.png               KSPC 比較棒グラフ
  fig_reduction.png          入力削減率比較棒グラフ
  fig_screen_occupancy.png   画面占有率比較棒グラフ
  fig_rank_accuracy.png      候補順位ごとの正解率分布
  fig_error_types.png        エラータイプ別割合
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 日本語フォントが無い環境でも文字化けを避けるため英語ラベルを使う
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# TODO: replace with actual scores
# 実スコアが出たら paper/results/metrics_*.json から自動で読む。
# 下記はあくまで動作確認用の仮値(本文では XXX とすること)。
# ============================================================
SAMPLE_METHODS = ["Proposed (vowel+LoRA)", "Base LLM (no LoRA)", "Freq baseline", "Random"]
SAMPLE_ACC = {  # TODO: replace with actual scores
    "Acc@1": [0.42, 0.18, 0.25, 0.05],
    "Acc@3": [0.66, 0.31, 0.40, 0.13],
    "Acc@5": [0.74, 0.39, 0.48, 0.20],
}
SAMPLE_CER = [0.28, 0.55, 0.50, 0.85]   # TODO: replace with actual scores
SAMPLE_WER = [0.35, 0.62, 0.58, 0.90]   # TODO: replace with actual scores
SAMPLE_MRR = [0.55, 0.26, 0.34, 0.10]   # TODO: replace with actual scores

# 入力効率(キー数/KSPC/削減率/画面占有率)。KSPC・キー数は設計上の概算、削減率は相対値。
SAMPLE_INPUT_METHODS = ["Proposed (5-vowel)", "Flick (12-key)", "Romaji/QWERTY", "Kana (50-key)"]
SAMPLE_KSPC = [1.0, 2.0, 2.2, 1.0]              # TODO: replace with measured KSPC
SAMPLE_REDUCTION = [0.50, 0.0, 0.0, 0.0]        # TODO: replace with actual reduction (vs baseline)
SAMPLE_SCREEN = [0.10, 0.30, 0.45, 0.50]        # TODO: replace with measured screen occupancy

SAMPLE_RANK_ACC = [0.42, 0.16, 0.10, 0.05, 0.03]  # rank1..5 ごとの「その順位に正解が出た割合」 TODO
SAMPLE_ERROR_TYPES = {  # TODO: replace with actual error analysis (errors_*.csv 集計)
    "correct": 0.42,
    "rank2-5": 0.24,
    "low_rank": 0.08,
    "not_in_candidates": 0.18,
    "top1_mismatch": 0.08,
}


def load_metrics(results_dir: str):
    """results ディレクトリの metrics_*.json を {tag: metrics} で返す。無ければ空。"""
    out = {}
    if not os.path.isdir(results_dir):
        return out
    for fn in os.listdir(results_dir):
        if fn.startswith("metrics_") and fn.endswith(".json"):
            tag = fn[len("metrics_"):-len(".json")]
            with open(os.path.join(results_dir, fn), encoding="utf-8") as f:
                out[tag] = json.load(f)
    return out


def bar(ax, labels, values, ylabel, title, ylim=None):
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.3)


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="../results")
    ap.add_argument("--fig_dir", default="../figures")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    real = load_metrics(args.results_dir)
    if real:
        print("loaded real metrics:", list(real.keys()))
        # TODO: 複数手法の metrics_*.json が揃ったら、SAMPLE_* の代わりに real から組み立てる
    else:
        print("WARNING: no metrics_*.json found -> using SAMPLE values (TODO: replace with actual scores)")

    # 1. Top-k Accuracy
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.25
    for i, (k, vals) in enumerate(SAMPLE_ACC.items()):
        ax.bar([x + i * width for x in range(len(SAMPLE_METHODS))], vals, width, label=k)
    ax.set_xticks([x + width for x in range(len(SAMPLE_METHODS))])
    ax.set_xticklabels(SAMPLE_METHODS, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Top-k Accuracy (SAMPLE - TODO replace)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, os.path.join(args.fig_dir, "fig_topk_accuracy.png"))

    # 2. CER / WER
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    ax.bar([x - width / 2 for x in range(len(SAMPLE_METHODS))], SAMPLE_CER, width, label="CER")
    ax.bar([x + width / 2 for x in range(len(SAMPLE_METHODS))], SAMPLE_WER, width, label="WER")
    ax.set_xticks(range(len(SAMPLE_METHODS)))
    ax.set_xticklabels(SAMPLE_METHODS, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Error rate")
    ax.set_title("CER / WER (SAMPLE - TODO replace)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, os.path.join(args.fig_dir, "fig_cer_wer.png"))

    # 3. MRR
    fig, ax = plt.subplots(figsize=(6, 4))
    bar(ax, SAMPLE_METHODS, SAMPLE_MRR, "MRR", "MRR (SAMPLE - TODO replace)", ylim=(0, 1))
    save(fig, os.path.join(args.fig_dir, "fig_mrr.png"))

    # 4. KSPC
    fig, ax = plt.subplots(figsize=(6, 4))
    bar(ax, SAMPLE_INPUT_METHODS, SAMPLE_KSPC, "KSPC", "KSPC (SAMPLE - TODO replace)")
    save(fig, os.path.join(args.fig_dir, "fig_kspc.png"))

    # 5. Reduction
    fig, ax = plt.subplots(figsize=(6, 4))
    bar(ax, SAMPLE_INPUT_METHODS, SAMPLE_REDUCTION, "Reduction", "Input Reduction (SAMPLE - TODO replace)", ylim=(0, 1))
    save(fig, os.path.join(args.fig_dir, "fig_reduction.png"))

    # 6. Screen occupancy
    fig, ax = plt.subplots(figsize=(6, 4))
    bar(ax, SAMPLE_INPUT_METHODS, SAMPLE_SCREEN, "Screen occupancy", "Screen Occupancy (SAMPLE - TODO replace)", ylim=(0, 1))
    save(fig, os.path.join(args.fig_dir, "fig_screen_occupancy.png"))

    # 7. rank-wise accuracy
    fig, ax = plt.subplots(figsize=(6, 4))
    ranks = [f"rank{r}" for r in range(1, len(SAMPLE_RANK_ACC) + 1)]
    bar(ax, ranks, SAMPLE_RANK_ACC, "P(correct at rank)", "Rank-wise Correct Rate (SAMPLE - TODO replace)")
    save(fig, os.path.join(args.fig_dir, "fig_rank_accuracy.png"))

    # 8. error types
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(SAMPLE_ERROR_TYPES.keys())
    vals = list(SAMPLE_ERROR_TYPES.values())
    ax.pie(vals, labels=labels, autopct="%1.0f%%", textprops={"fontsize": 8})
    ax.set_title("Error Types (SAMPLE - TODO replace)")
    save(fig, os.path.join(args.fig_dir, "fig_error_types.png"))


if __name__ == "__main__":
    main()
