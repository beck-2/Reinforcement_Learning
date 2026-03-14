"""
Evaluation script: measure alternation accuracy over full episodes.

Success criterion (per spec): stable high alternation rate across long episodes.
Random baseline is ~50% — a trained agent should significantly exceed this.
"""

import os
import numpy as np
import torch
from torch.distributions import Categorical

from figure8_maze_env import Figure8TMazeEnv
from model import RecurrentActorCritic
from config import Config
from train import obs_to_tensor


def evaluate(
    model: RecurrentActorCritic,
    config: Config,
    num_episodes: int = 20,
    greedy: bool = True,
) -> dict:
    """
    Run evaluation episodes and return accuracy statistics.

    Args:
        model:        trained RecurrentActorCritic
        config:       Config used for training
        num_episodes: number of full episodes to evaluate
        greedy:       True → argmax policy; False → sample from policy

    Returns:
        dict with keys: accuracy, accuracy_std, return, return_std
    """
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

    for ep in range(num_episodes):
        obs, _ = env.reset()
        hidden = model.init_hidden(device=device)
        done = False
        ep_return = 0.0
        steps = 0

        while not done:
            obs_t = obs_to_tensor(obs, device=device)
            with torch.no_grad():
                logits, _, hidden = model(obs_t, hidden)

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
        "accuracy":     float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "return":       float(np.mean(ep_returns)),
        "return_std":   float(np.std(ep_returns)),
        "length":       float(np.mean(ep_lengths)),
    }

    print(f"\n=== Evaluation ({num_episodes} episodes, {'greedy' if greedy else 'stochastic'}) ===")
    print(f"  Alternation accuracy : {results['accuracy']:.1%} ± {results['accuracy_std']:.1%}")
    print(f"  Episode return       : {results['return']:.2f} ± {results['return_std']:.2f}")
    print(f"  Episode length       : {results['length']:.0f} steps")
    print(f"  Random baseline      : ~50%")

    model.train()
    return results


if __name__ == "__main__":
    config = Config()
    model = RecurrentActorCritic(
        obs_dim=config.obs_dim,
        hidden_size=config.hidden_size,
        num_actions=config.num_actions,
    )

    if os.path.exists(config.checkpoint_path):
        model.load_state_dict(
            torch.load(config.checkpoint_path, weights_only=True)
        )
        print(f"Loaded checkpoint: {config.checkpoint_path}")
    else:
        print("No checkpoint found — evaluating untrained model (random baseline).")

    evaluate(model, config, num_episodes=20, greedy=True)
