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
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.2, 1.2, 2.2, 1.6, "User input\n5 vowels (+n)\n'a i a o u'", fc="#fdf0e6")
    box(ax, 2.9, 1.2, 2.4, 1.6, "Vowelizer\n$f_{vowel}$\n(pykakasi)")
    box(ax, 5.7, 1.2, 2.9, 1.6, "LLM candidate gen.\nMiniCPM4-0.5B\n+ LoRA")
    box(ax, 9.0, 0.8, 2.8, 2.4, "Top-k\ncandidates\n1. ...\n2. ...\n3. ...", fc="#e8f5ec")
    arrow(ax, 2.4, 2.0, 2.9, 2.0)
    arrow(ax, 5.3, 2.0, 5.7, 2.0)
    arrow(ax, 8.6, 2.0, 9.0, 2.0)
    box(ax, 5.7, 3.1, 2.9, 0.6, "optional context c", fc="#f3f3f3")
    arrow(ax, 7.15, 3.1, 7.15, 2.8)
    fig.savefig(os.path.join(FIG_DIR, "fig_overview.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_overview.png")


def flow():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax = axes[0]
    ax.set_xlim(0, 6); ax.set_ylim(0, 8); ax.axis("off")
    steps = ["Input text / reading x", "Reading (kana) via pykakasi",
             "Vowel string v", "Beam search over P(y|v)", "Dedup -> Top-k"]
    y = 7.0
    for i, s in enumerate(steps):
        box(ax, 0.5, y, 5.0, 0.9, s)
        if i < len(steps) - 1:
            arrow(ax, 3.0, y, 3.0, y - 0.6)
        y -= 1.5
    ax.set_title("Processing flow", fontsize=9)

    # 右: 学習曲線は実ログ curve.png を使う旨のプレースホルダ
    ax2 = axes[1]
    ax2.axis("off")
    ax2.text(0.5, 0.5,
             "Training curve:\nuse real log\nlogs/train_*/curve.png\n(loss -> ~1.2)",
             ha="center", va="center", fontsize=9,
             bbox=dict(boxstyle="round", fc="#f3f3f3", ec="#888"))
    fig.savefig(os.path.join(FIG_DIR, "fig_flow.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_flow.png")


if __name__ == "__main__":
    overview()
    flow()
