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


def box(ax, x, y, w, h, text, fc="#eef3fb", fontsize=9):  # 変更: 文字サイズを引数化
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                       linewidth=1.2, edgecolor="#33506e", facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.2, color="#33506e"))


def overview():
    # 変更: Inference 経路を削除し、Training(offline) のみを左から右へ並べる横長レイアウトにする
    fig, ax = plt.subplots(figsize=(12.0, 3.2))
    ax.set_xlim(0, 20); ax.set_ylim(0, 5); ax.axis("off")

    # 追加: 学習時(オフライン) 経路 — 左から右へ並べる
    ax.text(0.3, 4.4, "Training (offline)", fontsize=18, fontweight="bold")
    box(ax, 0.5, 1.4, 3.6, 1.8, "Corpus\nsentences", fc="#fdf0e6", fontsize=18)
    box(ax, 5.2, 1.4, 3.6, 1.8, "Vowelizer\n$f_{vowel}$\n(pykakasi)", fontsize=18)
    box(ax, 9.9, 1.4, 3.6, 1.8, "(vowel, text)\npairs", fontsize=18)
    box(ax, 14.6, 1.4, 4.6, 1.8, "LLM fine-tune\nMiniCPM4-0.5B\n+ LoRA", fc="#e8f5ec", fontsize=18)
    arrow(ax, 4.1, 2.3, 5.2, 2.3)
    arrow(ax, 8.8, 2.3, 9.9, 2.3)
    arrow(ax, 13.5, 2.3, 14.6, 2.3)

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
