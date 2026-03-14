"""
Visualize training progress from training_log.csv.

Usage:
    .venv/bin/python3 plot_training.py
    .venv/bin/python3 plot_training.py --log my_run.csv --out results.png
"""

import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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


def plot(log_path: str = "training_log.csv", out_path: str = "training_curves.png"):
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    data = load_log(log_path)
    steps = data["steps"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("A2C Training — Figure-8 Alternation Task", fontsize=14, y=0.98)
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    def add_panel(ax, y, title, ylabel, color, random_baseline=None):
        ax.plot(steps, y, alpha=0.25, color=color, linewidth=0.8)
        if len(y) >= 50:
            s = smooth(y)
            ax.plot(steps[len(steps) - len(s):], s, color=color, linewidth=1.8, label="smoothed")
        if random_baseline is not None:
            ax.axhline(random_baseline, color="gray", linestyle="--",
                       linewidth=1.0, label=f"random (~{random_baseline:.0%})")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Steps", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    add_panel(fig.add_subplot(gs[0, 0]),
              data["mean_return"], "Episode Return",
              "Return", "#4CAF50")

    add_panel(fig.add_subplot(gs[0, 1]),
              data["mean_episode_length"] if "mean_episode_length" in data else np.zeros_like(steps),
              "Episode Length", "Steps", "#00BCD4")

    add_panel(fig.add_subplot(gs[1, 0]),
              data["policy_loss"], "Policy Loss",
              "Loss", "#FF5722")

    add_panel(fig.add_subplot(gs[1, 1]),
              data["value_loss"], "Value Loss",
              "Loss", "#9C27B0")

    add_panel(fig.add_subplot(gs[2, 0]),
              data["entropy"], "Policy Entropy",
              "Entropy", "#FF9800")

    add_panel(fig.add_subplot(gs[2, 1]),
              data["grad_norm"], "Gradient Norm",
              "‖∇‖₂", "#607D8B")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}  ({len(steps):,} rollouts, {int(steps[-1]):,} steps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="training_log.csv")
    parser.add_argument("--out", default="training_curves.png")
    args = parser.parse_args()
    plot(args.log, args.out)
