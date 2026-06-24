#!/usr/bin/env python3
"""論文の概念図(概要図・処理フロー)を matplotlib で生成する。
日本語フォントが無い環境を想定し、図中ラベルは英語/ローマ字主体にする。
学習曲線は実ログ curve.png を流用するため、ここでは flow 図の右側にプレースホルダを描く。
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 追加: 図中の文字を Times New Roman にする
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False  # 追加

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def box(ax, x, y, w, h, text, fc="#eef3fb"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                       linewidth=1.2, edgecolor="#33506e", facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.2, color="#33506e"))


def overview():
    # 追加: 学習時(上段)と推論時(下段)の2経路に分けて描く
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 8); ax.axis("off")

    # 追加: 学習時(オフライン) 経路 — コーパス文を母音化して(母音列,文)対でLLMを学習
    ax.text(0.2, 7.5, "Training (offline)", fontsize=9, fontweight="bold")
    box(ax, 0.2, 5.2, 2.2, 1.6, "Corpus\nsentences", fc="#fdf0e6")
    box(ax, 2.9, 5.2, 2.4, 1.6, "Vowelizer\n$f_{vowel}$\n(pykakasi)")
    box(ax, 5.8, 5.2, 2.9, 1.6, "(vowel, text)\npairs")
    box(ax, 9.2, 5.2, 3.0, 1.6, "LLM fine-tune\nMiniCPM4-0.5B\n+ LoRA", fc="#e8f5ec")
    arrow(ax, 2.4, 6.0, 2.9, 6.0)
    arrow(ax, 5.3, 6.0, 5.8, 6.0)
    arrow(ax, 8.7, 6.0, 9.2, 6.0)

    # 追加: 推論時(ユーザ利用) 経路 — ユーザが母音+nを直接入力(母音化部を通らない)
    ax.text(0.2, 3.5, "Inference (user)", fontsize=9, fontweight="bold")
    box(ax, 0.2, 1.2, 2.6, 1.6, "User input\n5 vowels (+n)\n'a i a o u'", fc="#fdf0e6")
    box(ax, 5.8, 1.2, 2.9, 1.6, "LLM candidate gen.\n(trained model)")
    box(ax, 9.2, 0.8, 3.0, 2.4, "Top-k\ncandidates\n1. ...\n2. ...\n3. ...", fc="#e8f5ec")
    arrow(ax, 2.8, 2.0, 5.8, 2.0)
    arrow(ax, 8.7, 2.0, 9.2, 2.0)
    box(ax, 5.8, 3.1, 2.9, 0.6, "optional context c", fc="#f3f3f3")
    arrow(ax, 7.25, 3.1, 7.25, 2.8)

    fig.savefig(os.path.join(FIG_DIR, "fig_overview.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_overview.png")


def flow():
    # 変更: 図1と被る処理フロー(左)を削除し、学習曲線のみの図にする
    fig, ax2 = plt.subplots(figsize=(4.2, 3.0))
    ax2.axis("off")
    ax2.text(0.5, 0.5,
             "Training curve\n(TBD: insert real\nloss curve, loss -> ~1.2)",
             ha="center", va="center", fontsize=9,
             bbox=dict(boxstyle="round", fc="#ffff66", ec="#888"))
    fig.savefig(os.path.join(FIG_DIR, "fig_flow.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_flow.png")


if __name__ == "__main__":
    overview()
    flow()
