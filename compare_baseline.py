"""
Compare untrained baseline vs SSR trained agent behavior in Figure-8 maze.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from figure8_maze_env import Figure8TMazeEnv
from ssr_config import SSRConfig
from ssr_model import SSRRecurrentActorCritic
from test import obs_to_tensor


def evaluate_policy(env, model, config, episodes=20, random_policy=False, device=None):
    device = device or torch.device('cpu')
    model.eval()

    # Avoid expensive visual rendering during eval (fast path)
    env.disable_frame_render = True

    stats = []
    with torch.no_grad():
        for ep in range(episodes):
            obs, _ = env.reset()
            hidden = model.init_hidden(device=device)
            total_reward = 0.0
            done = False
            steps = 0

            while not done:
                if random_policy:
                    action = env.action_space.sample()
                else:
                    obs_t = obs_to_tensor(obs, config, device=device)
                    logits, _, _, hidden, _ = model(obs_t, hidden)
                    action = int(torch.argmax(logits, dim=-1).item())

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

            trial_stats = env.get_trial_statistics()
            stats.append({
                'episode': ep + 1,
                'total_reward': total_reward,
                'steps': steps,
                'accuracy': trial_stats['accuracy'],
                'correct': trial_stats['correct'],
                'incorrect': trial_stats['incorrect'],
            })

    return stats


def summary_stats(stats):
    rewards = [s['total_reward'] for s in stats]
    accuracies = [s['accuracy'] for s in stats]
    return {
        'episodes': len(stats),
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
    }


def plot_comparison(baseline_stats, trained_stats, output_path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot([s['episode'] for s in baseline_stats], [s['total_reward'] for s in baseline_stats], label='Baseline', alpha=0.7)
    ax[0].plot([s['episode'] for s in trained_stats], [s['total_reward'] for s in trained_stats], label='SR agent', alpha=0.7)
    ax[0].set_title('Total Reward per Episode')
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Total Reward')
    ax[0].legend()

    ax[1].plot([s['episode'] for s in baseline_stats], [s['accuracy'] for s in baseline_stats], label='Baseline', alpha=0.7)
    ax[1].plot([s['episode'] for s in trained_stats], [s['accuracy'] for s in trained_stats], label='SR agent', alpha=0.7)
    ax[1].set_title('Alternation Accuracy per Episode')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    fig.suptitle('Baseline vs SR Agent Comparison')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare baseline vs SSR agent')
    parser.add_argument('--episodes', type=int, default=20, help='Num episodes per comparison')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to trained checkpoint')
    parser.add_argument('--output', type=str, default='results/ssr_rnn', help='Output folder')
    args = parser.parse_args()

    cfg = SSRConfig()
    cfg.use_last_choice = False
    cfg.use_start_flag = True
    cfg.use_stem_sector = True

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline policy: no network, random moves
    env_baseline = Figure8TMazeEnv(
        max_trials_per_episode=cfg.max_trials_per_episode,
        step_cost=cfg.step_cost,
        turn_cost=cfg.turn_cost,
    )

    dummy_model = SSRRecurrentActorCritic(
        obs_dim=cfg.obs_dim,
        feature_dim=cfg.feature_dim,
        hidden_size=cfg.hidden_size,
        num_actions=cfg.num_actions,
    )

    baseline_stats = evaluate_policy(env_baseline, dummy_model, cfg, episodes=args.episodes, random_policy=True)
    trained_stats = []

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = cfg.checkpoint_path

    try:
        env_trained = Figure8TMazeEnv(
            max_trials_per_episode=cfg.max_trials_per_episode,
            step_cost=cfg.step_cost,
            turn_cost=cfg.turn_cost,
        )
        trained_model = SSRRecurrentActorCritic(
            obs_dim=cfg.obs_dim,
            feature_dim=cfg.feature_dim,
            hidden_size=cfg.hidden_size,
            num_actions=cfg.num_actions,
        )
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        trained_model.load_state_dict(state_dict, strict=False)
        trained_stats = evaluate_policy(env_trained, trained_model, cfg, episodes=args.episodes, random_policy=False)
    except Exception as e:
        print('Failed to run trained policy:', e)

    baseline_summary = summary_stats(baseline_stats)
    trained_summary = summary_stats(trained_stats) if trained_stats else None

    output_json = out_dir / 'baseline_comparison.json'
    with open(output_json, 'w') as f:
        json.dump({
            'baseline': baseline_summary,
            'trained': trained_summary,
            'baseline_episodes': baseline_stats,
            'trained_episodes': trained_stats,
        }, f, indent=2)

    output_plot = out_dir / 'baseline_comparison.png'
    if trained_stats:
        plot_comparison(baseline_stats, trained_stats, output_plot)

    print('Saved comparison JSON:', output_json)
    if trained_stats:
        print('Saved comparison plot:', output_plot)

    print('\nBaseline summary:', baseline_summary)
    if trained_summary:
        print('Trained summary:', trained_summary)
