"""
Train a successor representation (SR) recurrent agent on the Figure-8 maze.

Key ideas:
- The RNN provides working memory (no explicit last_choice input).
- SR head predicts discounted future feature occupancy.
- Actor-critic loss + SR TD loss.
"""

import argparse
import csv
import json
import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from figure8_maze_env import Figure8TMazeEnv
from constants import MAZE_SIZE
from ssr_config import SSRConfig
from ssr_model import SSRRecurrentActorCritic


# ── Observation encoding ───────────────────────────────────────────────────────

def obs_to_tensor(obs: dict, config: SSRConfig, device=None) -> torch.Tensor:
    """
    Encode env obs dict → flat tensor.

    Encodes: normalized (x, y) position + direction one-hot
    Optionally includes last_choice if config.use_last_choice is True.
    Shape: (1, obs_dim)
    """
    pos = obs["position_vector"] / MAZE_SIZE          # (2,) in [0, 1]
    d = int(obs["direction"])
    dir_oh = np.zeros(4, dtype=np.float32)
    dir_oh[d] = 1.0

    parts = [pos.astype(np.float32), dir_oh]
    if config.use_last_choice:
        last_choice = float(obs["last_choice"]) / 2.0  # normalize 0..2 → 0..1
        parts.append(np.array([last_choice], dtype=np.float32))

    vec = np.concatenate(parts)
    t = torch.tensor(vec, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t.unsqueeze(0)


# ── Return computation ─────────────────────────────────────────────────────────

def compute_returns(rewards, dones, last_value, gamma):
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    R = last_value
    for t in reversed(range(T)):
        if dones[t]:
            R = 0.0
        R = rewards[t] + gamma * R
        returns[t] = R
    return returns


# ── Gradient diagnostics ───────────────────────────────────────────────────────

def gradient_diagnostics(model: SSRRecurrentActorCritic) -> dict:
    total_sq = 0.0
    has_nan = False
    has_inf = False
    norms = []

    for p in model.parameters():
        if p.grad is not None:
            n = p.grad.data.norm(2).item()
            norms.append(n)
            total_sq += n * n
            if torch.isnan(p.grad).any():
                has_nan = True
            if torch.isinf(p.grad).any():
                has_inf = True

    return {
        "grad_norm": total_sq ** 0.5,
        "grad_max": max(norms) if norms else 0.0,
        "grad_min": min(norms) if norms else 0.0,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }


# ── Training loop ──────────────────────────────────────────────────────────────

def train(config: SSRConfig) -> SSRRecurrentActorCritic:
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Running environment state
    obs, _ = env.reset()
    hidden = model.init_hidden(device=device)
    current_ep_reward = 0.0

    # Logging buffers
    ep_rewards = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)

    rollout_idx = 0
    total_steps = 0
    current_ep_steps = 0

    # CSV logging
    log_fields = [
        "steps", "rollout",
        "mean_return", "mean_episode_length",
        "policy_loss", "value_loss", "entropy",
        "sr_loss", "grad_norm",
    ]
    log_file = open(config.training_log_path, "w", newline="")
    csv_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    csv_writer.writeheader()
    log_file.flush()

    print(f"Training for {config.num_train_steps:,} steps")
    print(f"  hidden_size={config.hidden_size} feature_dim={config.feature_dim} rollout={config.rollout_length}")
    print(f"  lr={config.lr} gamma={config.gamma} sr_coef={config.sr_loss_coef}")
    print(f"  metrics → {config.training_log_path}")

    while total_steps < config.num_train_steps:
        hidden = hidden.detach()

        rewards_list = []
        dones_list = []
        values_list = []
        log_probs_list = []
        entropies_list = []
        sr_list = []
        phi_list = []

        for _ in range(config.rollout_length):
            obs_t = obs_to_tensor(obs, config, device=device)
            logits, value, sr_pred, new_hidden, phi = model(obs_t, hidden)

            dist = Categorical(logits=logits)
            action = dist.sample()

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            rewards_list.append(float(reward))
            dones_list.append(done)
            values_list.append(value.squeeze(0))
            log_probs_list.append(dist.log_prob(action))
            entropies_list.append(dist.entropy())
            sr_list.append(sr_pred.squeeze(0))
            phi_list.append(phi.squeeze(0))

            current_ep_reward += reward
            current_ep_steps += 1
            hidden = new_hidden
            total_steps += 1

            if done:
                ep_rewards.append(current_ep_reward)
                ep_lengths.append(current_ep_steps)
                current_ep_reward = 0.0
                current_ep_steps = 0
                obs, _ = env.reset()
                hidden = model.init_hidden(device=device)
            else:
                obs = next_obs

        # Bootstrap value and SR
        with torch.no_grad():
            obs_boot = obs_to_tensor(obs, config, device=device)
            _, last_val, sr_boot, _, _ = model(obs_boot, hidden)
            last_val_scalar = last_val.item()
            sr_boot = sr_boot.squeeze(0)

        # Returns and advantages
        returns_np = compute_returns(rewards_list, dones_list, last_val_scalar, config.gamma)
        returns = torch.tensor(returns_np, dtype=torch.float32, device=device)

        values_t = torch.stack(values_list).squeeze(-1)         # (T,)
        log_probs_t = torch.stack(log_probs_list)               # (T,)
        entropies_t = torch.stack(entropies_list)               # (T,)

        advantages = returns - values_t.detach()

        # SR TD targets
        sr_preds = torch.stack(sr_list)                         # (T, feature_dim)
        phi_t = torch.stack(phi_list)                           # (T, feature_dim)
        sr_next = torch.vstack([sr_preds[1:], sr_boot.unsqueeze(0)])
        done_mask = torch.tensor(dones_list, dtype=torch.float32, device=device).unsqueeze(-1)
        sr_target = phi_t + (1.0 - done_mask) * config.gamma * sr_next.detach()

        sr_loss = F.mse_loss(sr_preds, sr_target)

        # Losses
        policy_loss = -(log_probs_t * advantages).mean()
        value_loss = F.mse_loss(values_t, returns)
        entropy_loss = -entropies_t.mean()

        loss = (
            policy_loss
            + config.value_loss_coef * value_loss
            + config.entropy_coef * entropy_loss
            + config.sr_loss_coef * sr_loss
        )

        if torch.isnan(loss):
            print(f"WARNING: NaN loss at step {total_steps}. Resetting episode state.")
            obs, _ = env.reset()
            hidden = model.init_hidden(device=device)
            current_ep_reward = 0.0
            continue

        optimizer.zero_grad()
        loss.backward()

        grad_info = gradient_diagnostics(model)
        if grad_info["has_nan"] or grad_info["has_inf"]:
            print(f"WARNING: NaN/Inf gradients at step {total_steps}!")

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        rollout_idx += 1

        mean_ret = np.mean(ep_rewards) if ep_rewards else 0.0
        mean_len = np.mean(ep_lengths) if ep_lengths else 0.0

        csv_writer.writerow({
            "steps":               total_steps,
            "rollout":             rollout_idx,
            "mean_return":         round(mean_ret, 4),
            "mean_episode_length": round(mean_len, 1),
            "policy_loss":         round(policy_loss.item(), 6),
            "value_loss":          round(value_loss.item(), 6),
            "entropy":             round(entropies_t.mean().item(), 6),
            "sr_loss":             round(sr_loss.item(), 6),
            "grad_norm":           round(grad_info["grad_norm"], 6),
        })
        log_file.flush()

        if rollout_idx % config.log_interval == 0:
            print(
                f"steps={total_steps:8d}  "
                f"return={mean_ret:6.2f}  "
                f"ploss={policy_loss.item():7.4f}  "
                f"vloss={value_loss.item():6.4f}  "
                f"sr={sr_loss.item():6.4f}  "
                f"ent={entropies_t.mean().item():.3f}  "
                f"gnorm={grad_info['grad_norm']:.3f}"
            )

        if rollout_idx % config.eval_interval == 0:
            torch.save(model.state_dict(), config.checkpoint_path)
            print(f"  [checkpoint saved at step {total_steps}]")

    torch.save(model.state_dict(), config.checkpoint_path)
    log_file.close()
    print(f"\nTraining complete. Checkpoint: {config.checkpoint_path} | Log: {config.training_log_path}")

    # Save config snapshot
    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=2)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="results output directory")
    parser.add_argument("--steps", type=int, default=None, help="override num_train_steps")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = SSRConfig()
    if args.output is not None:
        cfg.output_dir = args.output
    if args.steps is not None:
        cfg.num_train_steps = args.steps
    if args.seed is not None:
        cfg.seed = args.seed

    train(cfg)
