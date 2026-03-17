"""
Render and summarize example trajectories for trained SSR agent in Figure-8 maze.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from figure8_maze_env import Figure8TMazeEnv
from ssr_config import SSRConfig
from ssr_model import SSRRecurrentActorCritic
from test import obs_to_tensor


def eval_and_record(config: SSRConfig, checkpoint: str, episodes: int = 5, device=None):
    device = device or torch.device('cpu')

    env = Figure8TMazeEnv(
        max_trials_per_episode=config.max_trials_per_episode,
        step_cost=config.step_cost,
        turn_cost=config.turn_cost,
    )

    model = SSRRecurrentActorCritic(
        obs_dim=config.obs_dim,
        feature_dim=config.feature_dim,
        hidden_size=config.hidden_size,
        num_actions=config.num_actions,
    ).to(device)

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)  # allow old checkpoints missing auxiliary heads
    model.eval()

    all_episodes = []

    with torch.no_grad():
        for ep in range(episodes):
            obs, _ = env.reset()
            hidden = model.init_hidden(device=device)
            done = False

            while not done:
                obs_t = obs_to_tensor(obs, config, device=device)
                logits, _, _, hidden, _ = model(obs_t, hidden)

                # Greedy policy (best action)
                action = int(torch.argmax(logits, dim=-1).item())
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    done = True

            stats = env.get_trial_statistics()
            ep_data = {
                'episode': ep + 1,
                'accuracy': stats['accuracy'],
                'correct_trials': stats['correct'],
                'incorrect_trials': stats['incorrect'],
                'trials': stats['trial_history'],
            }
            all_episodes.append(ep_data)

    return all_episodes


def plot_trajectories(episodes, output_path, max_paths_per_episode=10):
    # Determine figure layout 1 row, N columns (up to 5)
    N = len(episodes)
    fig, axes = plt.subplots(1, N, figsize=(5 * N, 5), squeeze=False)

    for i, ep_data in enumerate(episodes):
        ax = axes[0][i]
        ax.set_title(f"Episode {ep_data['episode']}\nAcc {ep_data['accuracy']:.2f}")

        # Plot all trial trajectories (up to limit)
        for j, trial in enumerate(ep_data['trials'][:max_paths_per_episode]):
            traj = np.array([pose[:2] for pose in trial['trajectory']])
            if traj.shape[0] == 0:
                continue

            color = '#1f77b4' if trial['choice'] == 'left' else '#d62728'
            alpha = 0.15 if not trial['correct'] else 0.35
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha, linewidth=1.5)
            ax.scatter([traj[0, 0]], [traj[0, 1]], color='green', s=20, marker='o', label='start' if j == 0 else '')
            ax.scatter([traj[-1, 0]], [traj[-1, 1]], color='black', s=20, marker='x', label='end' if j == 0 else '')

        ax.set_xlim(0, 14)
        ax.set_ylim(0, 14)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot trained agent example trajectories')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (not needed)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint.pt')
    parser.add_argument('--episodes', type=int, default=4, help='Number of episodes to record')
    parser.add_argument('--output', type=str, default=None, help='Output directory for results')
    parser.add_argument('--max-per-ep', type=int, default=10, help='Max trials per episode to plot')
    args = parser.parse_args()

    cfg = SSRConfig()
    cfg.use_last_choice = False
    cfg.use_start_flag = True
    cfg.use_stem_sector = True
    cfg.sr_loss_coef = 0.1
    cfg.sr_warmup_steps = 20_000
    cfg.entropy_coef = 0.005
    cfg.lr = 3e-4
    cfg.grad_clip = 0.5
    cfg.rollout_length = 256
    cfg.gamma = 0.97
    cfg.num_train_steps = 2_000_000
    cfg.output_dir = 'results/ssr_rnn'

    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = cfg.checkpoint_path

    out_dir = Path(args.output or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes_data = eval_and_record(cfg, checkpoint_path, episodes=args.episodes)

    # Save summary JSON
    json_path = out_dir / 'trajectory_examples.json'
    with open(json_path, 'w') as f:
        json.dump(episodes_data, f, indent=2)

    # Plot
    image_path = out_dir / 'trajectory_examples.png'
    plot_trajectories(episodes_data, image_path, max_paths_per_episode=args.max_per_ep)

    # Print brief summary
    print(f"Saved trajectory summary: {json_path}")
    print(f"Saved trajectory plot: {image_path}")
    for ep in episodes_data:
        print(f"Episode {ep['episode']}: accuracy={ep['accuracy']:.3f}, correct={ep['correct_trials']}, incorrect={ep['incorrect_trials']}")
