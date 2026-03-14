"""
Generate example trajectories for trained SR-RNN and random baseline.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from figure8_maze_env import Figure8TMazeEnv
from ssr_config import SSRConfig
from ssr_model import SSRRecurrentActorCritic
from train_ssr import obs_to_tensor


def run_episode(model, config, greedy=True, random_policy=False):
    env = Figure8TMazeEnv(
        max_trials_per_episode=config.max_trials_per_episode,
        step_cost=0.0,
        turn_cost=0.0,
    )
    obs, _ = env.reset()
    hidden = model.init_hidden() if model is not None else None

    done = False
    while not done:
        if random_policy:
            action = env.action_space.sample()
        else:
            obs_t = obs_to_tensor(obs, config)
            with torch.no_grad():
                logits, _, _, hidden, _ = model(obs_t, hidden)
            action = logits.argmax(dim=-1).item() if greedy else torch.distributions.Categorical(logits=logits).sample().item()

        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    data = env.export_trajectory_data()
    env.close()
    return data


def plot_trajectories(data_trained, data_random, out_path):
    def plot_trials(ax, trials, title):
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 14)
        ax.set_ylim(14, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.15)

        colors = {"left": "#4CAF50", "right": "#FF5722"}
        for trial in trials[:6]:
            traj = trial["trajectory"]
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            ax.plot(xs, ys, color=colors[trial["choice"]], alpha=0.7, linewidth=1.2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_trials(axes[0], data_trained["all_trials"], "Trained SR-RNN (sample trials)")
    plot_trials(axes[1], data_random["all_trials"], "Random baseline (sample trials)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = SSRConfig()
    os.makedirs(config.output_dir, exist_ok=True)

    checkpoint = args.checkpoint or config.checkpoint_path
    model = SSRRecurrentActorCritic(
        obs_dim=config.obs_dim,
        feature_dim=config.feature_dim,
        hidden_size=config.hidden_size,
        num_actions=config.num_actions,
    )

    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("No checkpoint found — trajectories will be near-random.")

    trained_data = run_episode(model, config, greedy=True, random_policy=False)
    random_data = run_episode(None, config, random_policy=True)

    out = {
        "trained": trained_data,
        "random": random_data,
    }

    with open(config.trajectory_path, "w") as f:
        json.dump(out, f)

    plot_trajectories(trained_data, random_data, config.trajectory_plot_path)
    print(f"Saved: {config.trajectory_plot_path}")


if __name__ == "__main__":
    main()
