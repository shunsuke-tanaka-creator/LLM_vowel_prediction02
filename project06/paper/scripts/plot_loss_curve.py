#!/usr/bin/env python3
"""学習ログ(trainer_state.json)から train/eval loss の学習曲線を描く。

logs/train_*/checkpoint-*/trainer_state.json の log_history を読み、
train loss と eval loss を step 軸で重ねた図を paper/figures/fig_loss_curve.png に出す。
生データは paper/results/loss_curve.csv にも書き出す。
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="../../logs/train_20260622_140822/checkpoint-46876/trainer_state.json")
    ap.add_argument("--fig_dir", default="../figures")
    ap.add_argument("--results_dir", default="../results")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    with open(args.state, encoding="utf-8") as f:
        hist = json.load(f)["log_history"]

    train = [(h["step"], h["loss"]) for h in hist if "loss" in h]
    ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    print(f"train points: {len(train)}, eval points: {len(ev)}")

    # 生データを CSV に保存
    csv_path = os.path.join(args.results_dir, "loss_curve.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "split", "loss"])
        for s, l in train:
            w.writerow([s, "train", l])
        for s, l in ev:
            w.writerow([s, "eval", l])
    print("saved", csv_path)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([s for s, _ in train], [l for _, l in train],
            color="tab:blue", lw=1.0, alpha=0.8, label="Train loss")
    ax.plot([s for s, _ in ev], [l for _, l in ev],
            color="tab:red", marker="o", ms=4, lw=1.5, label="Eval loss")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Training and evaluation loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out = os.path.join(args.fig_dir, "fig_loss_curve.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
