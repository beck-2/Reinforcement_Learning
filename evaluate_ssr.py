"""
Evaluate a trained SR-RNN agent and compare to a random baseline.
"""

import argparse
import json
import os
import numpy as np
import torch
from torch.distributions import Categorical

from figure8_maze_env import Figure8TMazeEnv
from ssr_config import SSRConfig
from ssr_model import SSRRecurrentActorCritic
from train_ssr import obs_to_tensor


def run_policy(model, config, num_episodes=20, greedy=True):
    model.eval()
    device = next(model.parameters()).device

    env = Figure8TMazeEnv(
        max_trials_per_episode=config.max_trials_per_episode,
        step_cost=0.0,
        turn_cost=0.0,
    )

    accuracies = []
    ep_returns = []
    ep_lengths = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        hidden = model.init_hidden(device=device)
        done = False
        ep_return = 0.0
        steps = 0

        while not done:
            obs_t = obs_to_tensor(obs, config, device=device)
            with torch.no_grad():
                logits, _, _, hidden, _ = model(obs_t, hidden)

            if greedy:
                action = logits.argmax(dim=-1).item()
            else:
                action = Categorical(logits=logits).sample().item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            steps += 1

        accuracies.append(info.get("accuracy", 0.0))
        ep_returns.append(ep_return)
        ep_lengths.append(steps)

    results = {
        "accuracy": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "return": float(np.mean(ep_returns)),
        "return_std": float(np.std(ep_returns)),
        "length": float(np.mean(ep_lengths)),
        "episodes": int(num_episodes),
        "policy": "greedy" if greedy else "stochastic",
    }

    model.train()
    return results


def run_random(config, num_episodes=20):
    env = Figure8TMazeEnv(
        max_trials_per_episode=config.max_trials_per_episode,
        step_cost=0.0,
        turn_cost=0.0,
    )

    accuracies = []
    ep_returns = []
    ep_lengths = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        steps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            steps += 1

        accuracies.append(info.get("accuracy", 0.0))
        ep_returns.append(ep_return)
        ep_lengths.append(steps)

    return {
        "accuracy": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "return": float(np.mean(ep_returns)),
        "return_std": float(np.std(ep_returns)),
        "length": float(np.mean(ep_lengths)),
        "episodes": int(num_episodes),
        "policy": "random",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    config = SSRConfig()
    if args.episodes is not None:
        config.eval_episodes = args.episodes

    model = SSRRecurrentActorCritic(
        obs_dim=config.obs_dim,
        feature_dim=config.feature_dim,
        hidden_size=config.hidden_size,
        num_actions=config.num_actions,
    )

    checkpoint = args.checkpoint or config.checkpoint_path
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("No checkpoint found — evaluating untrained model (near-random baseline).")

    trained = run_policy(model, config, num_episodes=config.eval_episodes, greedy=True)
    baseline = run_random(config, num_episodes=config.eval_episodes)

    os.makedirs(config.output_dir, exist_ok=True)
    with open(config.eval_summary_path, "w") as f:
        json.dump(trained, f, indent=2)
    with open(config.baseline_summary_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print("\n=== Evaluation ===")
    print(f"  Trained accuracy : {trained['accuracy']:.1%} ± {trained['accuracy_std']:.1%}")
    print(f"  Random accuracy  : {baseline['accuracy']:.1%} ± {baseline['accuracy_std']:.1%}")


if __name__ == "__main__":
    main()
