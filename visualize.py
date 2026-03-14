"""
Post-training visualizations for the Figure-8 alternation task.

Produces:
  trajectory_comparison.png  — trained vs random agent paths on the maze,
                                color-coded by trial type (left/right)
  baseline_comparison.png    — accuracy + return bar chart vs random baseline
  value_function.png         — V(s) heatmap across maze cells (optional)

Usage:
    .venv/bin/python3 visualize.py                    # all plots
    .venv/bin/python3 visualize.py --no-value         # skip value function
    .venv/bin/python3 visualize.py --checkpoint my.pt
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.distributions import Categorical

from figure8_maze_env import Figure8TMazeEnv
from model import RecurrentActorCritic
from config import Config
from train import obs_to_tensor
from constants import (
    MAZE_SIZE, STEM_X, STEM_TOP, STEM_BOTTOM,
    LEFT_WELL_LOC, RIGHT_WELL_LOC,
    LEFT_RETURN_X, RIGHT_RETURN_X,
)


# ── Maze background ────────────────────────────────────────────────────────────

def build_maze_image(env) -> np.ndarray:
    """
    Build a 2-D color image of the maze layout (walls, floor, wells, stem).
    Returns array of shape (MAZE_SIZE, MAZE_SIZE, 3) with values in [0, 1].
    """
    from minigrid.core.world_object import Wall, Floor, Goal
    from figure8_maze_env import WaterWell, MazeWall, StemFloor

    img = np.ones((MAZE_SIZE, MAZE_SIZE, 3))   # start white

    for x in range(MAZE_SIZE):
        for y in range(MAZE_SIZE):
            cell = env.grid.get(x, y)
            if cell is None or isinstance(cell, Floor):
                img[y, x] = [0.92, 0.92, 0.92]   # light grey — walkable
            elif isinstance(cell, StemFloor):
                # Stem sectors: slightly tinted to show structure
                sector = cell.sector
                tint = 0.75 + sector * 0.04
                img[y, x] = [tint, tint, 0.95]    # blue-ish for stem
            elif isinstance(cell, WaterWell):
                if (x, y) == LEFT_WELL_LOC:
                    img[y, x] = [0.2, 0.6, 1.0]   # blue — left well
                else:
                    img[y, x] = [1.0, 0.4, 0.2]   # orange — right well
            elif isinstance(cell, (Wall, MazeWall)):
                img[y, x] = [0.25, 0.25, 0.25]    # dark — wall

    return img


def draw_maze(ax, env):
    """Render maze background on a matplotlib axes."""
    img = build_maze_image(env)
    ax.imshow(img, origin="upper", extent=[-0.5, MAZE_SIZE - 0.5,
                                            MAZE_SIZE - 0.5, -0.5])
    ax.set_xlim(-0.5, MAZE_SIZE - 0.5)
    ax.set_ylim(MAZE_SIZE - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(model_or_none, config: Config, seed: int = 0):
    """
    Run one full episode.

    Args:
        model_or_none: trained RecurrentActorCritic, or None for random policy
        config:        Config instance
        seed:          random seed for reproducibility

    Returns:
        trial_history list from env (each entry has 'choice', 'correct',
        'trajectory' = list of (x, y, dir) poses)
        accuracy: float
    """
    env = Figure8TMazeEnv(
        max_trials_per_episode=config.max_trials_per_episode,
        step_cost=0.0,
        turn_cost=0.0,
    )
    obs, _ = env.reset(seed=seed)

    if model_or_none is not None:
        model_or_none.eval()
        device = next(model_or_none.parameters()).device
        hidden = model_or_none.init_hidden(device=device)

    done = False
    while not done:
        if model_or_none is None:
            action = env.action_space.sample()
        else:
            obs_t = obs_to_tensor(obs, device=device)
            with torch.no_grad():
                logits, _, hidden = model_or_none(obs_t, hidden)
            action = logits.argmax(dim=-1).item()

        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return env.trial_history, info.get("accuracy", 0.0)


# ── Trajectory comparison ──────────────────────────────────────────────────────

def plot_trajectory_comparison(trained_history, random_history, env, out_path):
    """
    Side-by-side maze plots: left-turn trials (blue) vs right-turn (red),
    for trained agent (left panel) and random agent (right panel).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    fig.suptitle("Agent Trajectories by Trial Type\n"
                 "Blue = left-turn trials   Red = right-turn trials",
                 fontsize=12)

    titles = ["Trained Agent", "Random Agent (baseline)"]
    histories = [trained_history, random_history]

    for ax, history, title in zip(axes, histories, titles):
        draw_maze(ax, env)
        ax.set_title(title, fontsize=11)

        for trial in history:
            traj = trial["trajectory"]   # list of (x, y, dir)
            if len(traj) < 2:
                continue
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            color = "#2196F3" if trial["choice"] == "left" else "#F44336"
            alpha = 0.6 if trial["correct"] else 0.25
            ax.plot(xs, ys, color=color, alpha=alpha, linewidth=1.0)
            # Mark start of trial
            ax.plot(xs[0], ys[0], "o", color=color, alpha=alpha,
                    markersize=3, markeredgewidth=0)

        # Legend
        patches = [
            mpatches.Patch(color="#2196F3", label="Left-turn trial"),
            mpatches.Patch(color="#F44336", label="Right-turn trial"),
            mpatches.Patch(color="gray",   alpha=0.3, label="Incorrect (faded)"),
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Baseline comparison bar chart ──────────────────────────────────────────────

def plot_baseline_comparison(trained_acc, random_acc, out_path):
    """Bar chart: alternation accuracy of trained vs random agent."""
    fig, ax = plt.subplots(figsize=(5, 5))

    labels = ["Random\n(baseline)", "Trained\nAgent"]
    values = [random_acc, trained_acc]
    colors = ["#90A4AE", "#42A5F5"]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white",
                  linewidth=1.5)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0,
               label="Chance (50%)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Alternation Accuracy", fontsize=11)
    ax.set_title("Trained Agent vs Random Baseline", fontsize=12)
    ax.legend(fontsize=9)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.1%}", ha="center", va="bottom", fontsize=12,
                fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Value function heatmap ─────────────────────────────────────────────────────

def plot_value_function(model, config: Config, env, out_path):
    """
    Compute V(s) at every walkable cell with zeroed hidden state,
    averaged over all 4 facing directions. Shows which locations the
    model considers most valuable.

    Note: uses h=0 so this is an approximation — recurrent context is ignored.
    """
    model.eval()
    device = next(model.parameters()).device
    h0 = model.init_hidden(device=device)

    value_grid = np.full((MAZE_SIZE, MAZE_SIZE), np.nan)

    from minigrid.core.world_object import Wall
    from figure8_maze_env import MazeWall

    for x in range(MAZE_SIZE):
        for y in range(MAZE_SIZE):
            cell = env.grid.get(x, y)
            # Skip walls
            if isinstance(cell, (Wall, MazeWall)):
                continue

            # Average value over all 4 facing directions
            vals = []
            for direction in range(4):
                pos = np.array([x, y], dtype=np.float32) / MAZE_SIZE
                dir_oh = np.zeros(4, dtype=np.float32)
                dir_oh[direction] = 1.0
                obs_vec = torch.tensor(
                    np.concatenate([pos, dir_oh]), dtype=torch.float32
                ).unsqueeze(0).to(device)

                with torch.no_grad():
                    _, value, _ = model(obs_vec, h0)
                vals.append(value.item())

            value_grid[y, x] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(7, 7))
    # Draw maze background first (low alpha)
    maze_img = build_maze_image(env)
    ax.imshow(maze_img, origin="upper",
              extent=[-0.5, MAZE_SIZE - 0.5, MAZE_SIZE - 0.5, -0.5],
              alpha=0.3)

    # Overlay value heatmap (only on walkable cells)
    masked = np.ma.masked_invalid(value_grid)
    hm = ax.imshow(masked, origin="upper",
                   extent=[-0.5, MAZE_SIZE - 0.5, MAZE_SIZE - 0.5, -0.5],
                   cmap="RdYlGn", alpha=0.8,
                   vmin=np.nanmin(value_grid), vmax=np.nanmax(value_grid))

    plt.colorbar(hm, ax=ax, fraction=0.03, label="V(s)  [h=0 approx.]")
    ax.set_title("Value Function V(s) — averaged over directions\n"
                 "(zeroed hidden state; recurrent context not included)",
                 fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--no-value", action="store_true",
                        help="Skip value function plot")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Episodes to average accuracy over")
    args = parser.parse_args()

    config = Config()

    # Load model
    model = RecurrentActorCritic(
        obs_dim=config.obs_dim,
        hidden_size=config.hidden_size,
        num_actions=config.num_actions,
    )
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"Checkpoint not found ({args.checkpoint}) — using untrained model.")

    # Build env (needed for maze background)
    env = Figure8TMazeEnv(
        max_trials_per_episode=config.max_trials_per_episode,
        step_cost=0.0, turn_cost=0.0,
    )
    env.reset()

    # ── Trajectories (single episode each for visual clarity) ──────────────
    print("\nRunning trained agent episode...")
    trained_history, trained_acc_single = run_episode(model, config, seed=42)

    print("Running random agent episode...")
    random_history, random_acc_single = run_episode(None, config, seed=42)

    plot_trajectory_comparison(
        trained_history, random_history, env,
        out_path="trajectory_comparison.png",
    )

    # ── Accuracy over multiple episodes ────────────────────────────────────
    print(f"\nEvaluating over {args.eval_episodes} episodes...")
    trained_accs, random_accs = [], []
    for seed in range(args.eval_episodes):
        _, ta = run_episode(model, config, seed=seed)
        _, ra = run_episode(None, config, seed=seed)
        trained_accs.append(ta)
        random_accs.append(ra)

    mean_trained = float(np.mean(trained_accs))
    mean_random = float(np.mean(random_accs))
    print(f"  Trained: {mean_trained:.1%}  |  Random: {mean_random:.1%}")

    plot_baseline_comparison(
        mean_trained, mean_random,
        out_path="baseline_comparison.png",
    )

    # ── Value function (optional) ───────────────────────────────────────────
    if not args.no_value:
        print("\nComputing value function...")
        plot_value_function(model, config, env, out_path="value_function.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
