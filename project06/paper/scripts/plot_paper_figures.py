#!/usr/bin/env python3
"""論文用の図を生成する。

paper/results/metrics_*.json が存在すればそれを読み、無ければ
スクリプト内の SAMPLE_* (仮値) を使って動作確認用の図を出す。
本文中の数値は XXX のままにし、確定後に図を差し替える運用。

生成図(paper/figures/ に PNG):
  fig_topk_accuracy.png      Top-k Accuracy 比較棒グラフ
  fig_kspc.png               KSPC 比較棒グラフ(metrics_*.json に KSPC_* があれば実値)
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 追加: 図中の文字を Times New Roman にする
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

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

# 入力効率(KSPC)の仮値。順は [Proposed, Romaji(kunrei), Kana(JIS)/Flick, Toggle]。
# kana(JIS) と flick は同一の理想 KSPC になるため 1 本にまとめて表示する。
SAMPLE_KSPC = [1.0, 1.9, 1.3, 2.5]              # TODO: replace with computed KSPC


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

    # 1. Top-k Accuracy: metrics_*.json があれば提案手法の実値、無ければ SAMPLE
    acc_keys = ["Acc@1", "Acc@3", "Acc@5"]
    real_acc = None
    for tag, m in real.items():
        if all(k in m for k in acc_keys):
            real_acc = [m[k] for k in acc_keys]
            print(f"Acc from metrics_{tag}.json:", dict(zip(acc_keys, real_acc)))
            break
    fig, ax = plt.subplots(figsize=(6, 4))
    if real_acc is not None:
        # baseline 未実装のため提案手法のみの実値棒グラフ(追加)
        ax.bar(range(len(acc_keys)), real_acc, color="tab:blue")
        ax.set_xticks(range(len(acc_keys)))
        ax.set_xticklabels(acc_keys)
        for x, v in enumerate(real_acc):
            ax.text(x, v, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_title("Top-k Accuracy (Proposed)")
        ax.grid(True, axis="y", alpha=0.3)
    else:
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

    # 2. KSPC: metrics_*.json に KSPC_* が揃っていれば実値、無ければ SAMPLE
    # kana(JIS) と flick は同一の理想 KSPC になるため 1 本にまとめて表示する(変更)。
    kspc_labels = ["Proposed (vowel)", "Romaji (kunrei)", "Kana (JIS) / Flick", "Toggle"]
    kspc_keys = ["KSPC_proposed", "KSPC_romaji_kunrei", "KSPC_kana", "KSPC_toggle"]
    kspc_vals = None
    for tag, m in real.items():
        if all(k in m for k in kspc_keys):
            kspc_vals = [m[k] for k in kspc_keys]
            print(f"KSPC from metrics_{tag}.json:", dict(zip(kspc_labels, kspc_vals)))
            break
    title = "KSPC by input method"
    if kspc_vals is None:
        kspc_vals = SAMPLE_KSPC
        title = "KSPC (SAMPLE - TODO replace)"
    fig, ax = plt.subplots(figsize=(6, 4))
    bar(ax, kspc_labels, kspc_vals, "KSPC", title)
    save(fig, os.path.join(args.fig_dir, "fig_kspc.png"))


if __name__ == "__main__":
    main()
