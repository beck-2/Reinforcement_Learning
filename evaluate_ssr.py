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
from train_ssr import obs_to_tensor, _valid_action_mask
# Add to eval_ssr.py:
_DIR_TO_VEC = {0: (1,0), 1: (0,1), 2: (-1,0), 3: (0,-1)}

def _forward_open(env, direction):
    x, y = env.agent_pos
    dx, dy = _DIR_TO_VEC[direction]
    cell = env.grid.get(x+dx, y+dy)
    return cell is None or cell.can_overlap()

def _valid_action_mask_eval(env, device):
    mask, _ = _valid_action_mask(env, device)
    return mask

def run_policy(model, config, num_episodes=20, greedy=True, reset_hidden_each_step=False):
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
            if reset_hidden_each_step:
                hidden = model.init_hidden(device=device)
            with torch.no_grad():
                logits, _, _, hidden, _ = model(obs_t, hidden)

            # Apply mask — same as training
            mask = _valid_action_mask_eval(env, logits.device)
            masked_logits = logits.masked_fill(~mask.unsqueeze(0), -1e9)

            if greedy:
                action = masked_logits.argmax(dim=-1).item()
            else:
                action = Categorical(logits=masked_logits).sample().item()

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

    checkpoint = args.checkpoint or config.checkpoint_path
    if os.path.exists(checkpoint):
        cfg_path = os.path.join(os.path.dirname(checkpoint), "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                saved = json.load(f)
            for k, v in saved.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        model = SSRRecurrentActorCritic(
            obs_dim=config.obs_dim,
            feature_dim=config.feature_dim,
            hidden_size=config.hidden_size,
            num_actions=config.num_actions,
        )
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("No checkpoint found — evaluating untrained model (near-random baseline).")
        model = SSRRecurrentActorCritic(
            obs_dim=config.obs_dim,
            feature_dim=config.feature_dim,
            hidden_size=config.hidden_size,
            num_actions=config.num_actions,
        )

    trained = run_policy(model, config, num_episodes=config.eval_episodes, greedy=False, reset_hidden_each_step=False)
    no_memory = run_policy(model, config, num_episodes=config.eval_episodes, greedy=False, reset_hidden_each_step=True)
    baseline = run_random(config, num_episodes=config.eval_episodes)

    os.makedirs(config.output_dir, exist_ok=True)
    with open(config.eval_summary_path, "w") as f:
        json.dump({"memory": trained, "no_memory": no_memory}, f, indent=2)
    with open(config.baseline_summary_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print("\n=== Evaluation ===")
    print(f"  Trained (memory) accuracy : {trained['accuracy']:.1%} ± {trained['accuracy_std']:.1%}")
    print(f"  No-memory accuracy        : {no_memory['accuracy']:.1%} ± {no_memory['accuracy_std']:.1%}")
    print(f"  Random accuracy           : {baseline['accuracy']:.1%} ± {baseline['accuracy_std']:.1%}")


if __name__ == "__main__":
    main()
