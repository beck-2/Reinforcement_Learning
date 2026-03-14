"""
Plot learning curves and baseline comparison for SR-RNN training.
"""

import argparse
import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ssr_config import SSRConfig


def load_log(path: str) -> dict:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty log file: {path}")
    data = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    return data


def smooth(x: np.ndarray, window: int = 50) -> np.ndarray:
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_training(log_path: str, out_path: str):
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    data = load_log(log_path)
    steps = data["steps"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("SR-RNN Training — Figure-8 Alternation Task", fontsize=14, y=0.98)
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    def add_panel(ax, y, title, ylabel, color):
        ax.plot(steps, y, alpha=0.25, color=color, linewidth=0.8)
        if len(y) >= 50:
            s = smooth(y)
            ax.plot(steps[len(steps) - len(s):], s, color=color, linewidth=1.8, label="smoothed")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Steps", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    add_panel(fig.add_subplot(gs[0, 0]), data["mean_return"], "Episode Return", "Return", "#4CAF50")
    add_panel(fig.add_subplot(gs[0, 1]), data["mean_episode_length"], "Episode Length", "Steps", "#00BCD4")
    add_panel(fig.add_subplot(gs[1, 0]), data["policy_loss"], "Policy Loss", "Loss", "#FF5722")
    add_panel(fig.add_subplot(gs[1, 1]), data["value_loss"], "Value Loss", "Loss", "#9C27B0")
    add_panel(fig.add_subplot(gs[2, 0]), data["entropy"], "Policy Entropy", "Entropy", "#FF9800")
    add_panel(fig.add_subplot(gs[2, 1]), data["sr_loss"], "SR TD Loss", "MSE", "#607D8B")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}  ({len(steps):,} rollouts, {int(steps[-1]):,} steps)")


def plot_baseline(trained_path: str, baseline_path: str, out_path: str):
    if not os.path.exists(trained_path) or not os.path.exists(baseline_path):
        print("Missing eval summaries for baseline plot.")
        return

    with open(trained_path) as f:
        trained = json.load(f)
    with open(baseline_path) as f:
        baseline = json.load(f)

    labels = ["Trained", "Random"]
    acc = [trained["accuracy"], baseline["accuracy"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, acc, color=["#4CAF50", "#9E9E9E"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Alternation Accuracy")
    ax.set_title("Alternation Accuracy Comparison")
    for i, v in enumerate(acc):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None)
    parser.add_argument("--eval", default=None)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--baseline-out", default=None)
    args = parser.parse_args()

    cfg = SSRConfig()

    log_path = args.log or cfg.training_log_path
    out_path = args.out or cfg.training_curves_path
    plot_training(log_path, out_path)

    eval_path = args.eval or cfg.eval_summary_path
    baseline_path = args.baseline or cfg.baseline_summary_path
    baseline_out = args.baseline_out or cfg.baseline_plot_path
    plot_baseline(eval_path, baseline_path, baseline_out)


if __name__ == "__main__":
    main()
