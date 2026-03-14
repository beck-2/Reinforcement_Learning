"""
Diagnostic tool for a saved RecurrentActorCritic checkpoint.

Checks:
  - Choice distribution (left vs right)
  - Whether alternation ever occurs (consecutive opposite choices)
  - Action distribution at T-junction (x=7, y=4)
  - Value estimates across key locations
  - Policy entropy across the maze

Usage:
    .venv/bin/python3 diagnose_checkpoint.py
    .venv/bin/python3 diagnose_checkpoint.py --checkpoint checkpoint.pt --episodes 20
"""

import argparse
import os
import numpy as np
import torch
from torch.distributions import Categorical
from collections import Counter

from figure8_maze_env import Figure8TMazeEnv
from model import RecurrentActorCritic
from config import Config
from train import obs_to_tensor
from constants import MAZE_SIZE


def run_diagnostic_episode(model, config, seed):
    env = Figure8TMazeEnv(
        max_trials_per_episode=config.max_trials_per_episode,
        step_cost=0.0,
        turn_cost=0.0,
    )
    obs, _ = env.reset(seed=seed)
    model.eval()
    device = next(model.parameters()).device
    hidden = model.init_hidden(device=device)

    done = False
    while not done:
        obs_t = obs_to_tensor(obs, device=device)
        with torch.no_grad():
            logits, _, hidden = model(obs_t, hidden)
        action = logits.argmax(dim=-1).item()
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return env.trial_history, info.get("accuracy", 0.0)


def analyze_alternation(all_histories):
    """Check if the agent ever alternates vs always chooses same side."""
    choices_flat = []
    alternation_counts = []
    perseveration_counts = []

    for history in all_histories:
        choices = [t["choice"] for t in history if "choice" in t]
        choices_flat.extend(choices)
        alts = sum(1 for i in range(1, len(choices)) if choices[i] != choices[i-1])
        pers = sum(1 for i in range(1, len(choices)) if choices[i] == choices[i-1])
        alternation_counts.append(alts)
        perseveration_counts.append(pers)

    choice_counter = Counter(choices_flat)
    total = len(choices_flat)
    print(f"\n--- Choice Distribution ({total} total choices) ---")
    for side, count in sorted(choice_counter.items()):
        print(f"  {side:>5}: {count:4d}  ({count/total:.1%})")

    total_alt = sum(alternation_counts)
    total_per = sum(perseveration_counts)
    total_transitions = total_alt + total_per
    print(f"\n--- Alternation vs Perseveration (consecutive trial pairs) ---")
    print(f"  Alternations:    {total_alt:4d}  ({total_alt/max(total_transitions,1):.1%})")
    print(f"  Perseverations:  {total_per:4d}  ({total_per/max(total_transitions,1):.1%})")
    print(f"  (50% alternation = random; >50% = memory; <50% = side bias)")


def analyze_junction_policy(model, config):
    """Sample policy at the T-junction (x=7, y=4) for all directions."""
    device = next(model.parameters()).device
    h0 = model.init_hidden(device=device)
    model.eval()

    action_names = ["turn_left", "turn_right", "forward"]
    dir_names = ["East", "South", "West", "North"]

    print(f"\n--- Policy at T-junction (x=7, y=4) with zeroed hidden state ---")
    for direction in range(4):
        pos = np.array([7, 4], dtype=np.float32) / MAZE_SIZE
        dir_oh = np.zeros(4, dtype=np.float32)
        dir_oh[direction] = 1.0
        obs_vec = torch.tensor(np.concatenate([pos, dir_oh]), dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, value, _ = model(obs_vec, h0)
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        dist = Categorical(logits=logits)
        entropy = dist.entropy().item()

        print(f"  dir={dir_names[direction]:5s}: "
              + "  ".join(f"{action_names[i]}={probs[i]:.2f}" for i in range(3))
              + f"  V={value.item():.3f}  H={entropy:.3f}")


def analyze_value_at_key_locations(model, config):
    """V(s) at important maze locations."""
    device = next(model.parameters()).device
    h0 = model.init_hidden(device=device)
    model.eval()

    key_locs = [
        ("start (stem top)", 7, 2),
        ("T-junction", 7, 4),
        ("left well", 4, 4),
        ("right well", 10, 4),
        ("left return", 5, 6),
        ("right return", 9, 6),
        ("stem bottom", 7, 8),
    ]

    print(f"\n--- Value estimates at key locations (h=0, averaged over directions) ---")
    for name, x, y in key_locs:
        vals = []
        for direction in range(4):
            pos = np.array([x, y], dtype=np.float32) / MAZE_SIZE
            dir_oh = np.zeros(4, dtype=np.float32)
            dir_oh[direction] = 1.0
            obs_vec = torch.tensor(np.concatenate([pos, dir_oh]), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                _, value, _ = model(obs_vec, h0)
            vals.append(value.item())
        print(f"  {name:25s} ({x:2d},{y:2d}): V={np.mean(vals):+.4f}  [min={min(vals):+.4f}, max={max(vals):+.4f}]")


def analyze_entropy_distribution(model, config):
    """Mean policy entropy across all walkable cells."""
    from minigrid.core.world_object import Wall
    from figure8_maze_env import MazeWall

    device = next(model.parameters()).device
    h0 = model.init_hidden(device=device)
    model.eval()

    env = Figure8TMazeEnv(max_trials_per_episode=1, step_cost=0.0, turn_cost=0.0)
    env.reset()

    entropies = []
    for x in range(MAZE_SIZE):
        for y in range(MAZE_SIZE):
            cell = env.grid.get(x, y)
            if isinstance(cell, (Wall, MazeWall)):
                continue
            for direction in range(4):
                pos = np.array([x, y], dtype=np.float32) / MAZE_SIZE
                dir_oh = np.zeros(4, dtype=np.float32)
                dir_oh[direction] = 1.0
                obs_vec = torch.tensor(np.concatenate([pos, dir_oh]), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _, _ = model(obs_vec, h0)
                dist = Categorical(logits=logits)
                entropies.append(dist.entropy().item())

    env.close()
    max_entropy = np.log(3)  # uniform over 3 actions
    mean_h = np.mean(entropies)
    print(f"\n--- Policy Entropy (h=0, all walkable cells) ---")
    print(f"  Mean entropy:  {mean_h:.4f}  ({mean_h/max_entropy:.1%} of max={max_entropy:.4f})")
    print(f"  Min entropy:   {min(entropies):.4f}")
    print(f"  Max entropy:   {max(entropies):.4f}")
    if mean_h / max_entropy > 0.95:
        print("  *** Entropy near maximum — policy is nearly uniform (no learning) ***")
    elif mean_h / max_entropy < 0.5:
        print("  *** Entropy well below max — policy has learned distinct preferences ***")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    config = Config()

    model = RecurrentActorCritic(
        obs_dim=config.obs_dim,
        hidden_size=config.hidden_size,
        num_actions=config.num_actions,
    )
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"WARNING: Checkpoint not found ({args.checkpoint}) — using untrained model.")

    print(f"\nRunning {args.episodes} diagnostic episodes...")
    all_histories = []
    accuracies = []
    for seed in range(args.episodes):
        history, acc = run_diagnostic_episode(model, config, seed=seed)
        all_histories.append(history)
        accuracies.append(acc)

    print(f"\n--- Episode Accuracy ---")
    print(f"  Mean: {np.mean(accuracies):.1%}  Std: {np.std(accuracies):.1%}")
    print(f"  Min:  {min(accuracies):.1%}  Max: {max(accuracies):.1%}")

    analyze_alternation(all_histories)
    analyze_junction_policy(model, config)
    analyze_value_at_key_locations(model, config)
    analyze_entropy_distribution(model, config)


if __name__ == "__main__":
    main()
