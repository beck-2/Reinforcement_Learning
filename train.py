"""
A2C training loop with truncated BPTT for the Figure-8 alternation task.

Key design decisions (per spec):
- Vanilla RNN (not LSTM) — simplicity first; upgrade if memory fails
- rollout_length=64 steps per update
- Hidden state carried across rollout chunks, detached at chunk boundaries
- Hidden state reset to zeros at episode boundaries
- Sparse reward: +1 correct alternation, 0 otherwise
- Gradient diagnostics logged every step
"""

import csv
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

from figure8_maze_env import Figure8TMazeEnv
from model import RecurrentActorCritic
from config import Config
from constants import MAZE_SIZE


# ── Curriculum stage definitions ───────────────────────────────────────────────
# Stage rewards are applied directly to env attributes at runtime.
# See methods.md for the full rationale behind each stage.

STAGE_CONFIGS = {
    1: dict(
        step_cost=0.0,
        turn_cost=0.0,
        correct_reward=1.0,
        incorrect_reward=0.0,
        foraging_reward=0.5,
        loop_bonus=0.2,
        use_stage1_barriers=True,
        force_alternation_barriers=True,
        wall_bump_penalty=-0.1,
    ),
    2: dict(
        step_cost=-0.001,
        turn_cost=0.0,
        correct_reward=1.0,
        incorrect_reward=0.0,
        foraging_reward=0.1,
        loop_bonus=0.0,
        use_stage1_barriers=True,
        force_alternation_barriers=False,
        wall_bump_penalty=-0.1,
    ),
    3: dict(
        step_cost=-0.001,
        turn_cost=0.0,
        correct_reward=1.0,
        incorrect_reward=0.0,
        foraging_reward=0.0,
        loop_bonus=0.0,
        use_stage1_barriers=False,
        force_alternation_barriers=False,
        wall_bump_penalty=-0.1,
    ),
}


def get_stage(total_steps: int, config: Config) -> int:
    if total_steps >= config.stage3_start:
        return 3
    if total_steps >= config.stage2_start:
        return 2
    return 1


def apply_stage(env: Figure8TMazeEnv, stage: int) -> None:
    """Update env reward attributes in-place for the given curriculum stage."""
    cfg = STAGE_CONFIGS[stage]
    env.step_cost = cfg["step_cost"]
    env.turn_cost = cfg["turn_cost"]
    env.correct_reward = cfg["correct_reward"]
    env.incorrect_reward = cfg["incorrect_reward"]
    env.foraging_reward = cfg["foraging_reward"]
    env.loop_bonus = cfg["loop_bonus"]
    env.use_stage1_barriers = cfg["use_stage1_barriers"]
    env.force_alternation_barriers = cfg["force_alternation_barriers"]
    env.wall_bump_penalty = cfg["wall_bump_penalty"]


# ── Observation encoding ───────────────────────────────────────────────────────

def obs_to_tensor(obs: dict, device=None) -> torch.Tensor:
    """
    Encode env obs dict → flat tensor.

    Encodes: normalized (x, y) position  +  direction one-hot
    Shape: (1, obs_dim) = (1, 6)
    """
    pos = obs["position_vector"] / MAZE_SIZE          # (2,) in [0, 1]
    d = int(obs["direction"])
    dir_oh = np.zeros(4, dtype=np.float32)
    dir_oh[d] = 1.0
    vec = np.concatenate([pos, dir_oh])               # (6,)
    t = torch.tensor(vec, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t.unsqueeze(0)                             # (1, 6)


# ── Return computation ─────────────────────────────────────────────────────────

def compute_returns(
    rewards: list,
    dones: list,
    last_value: float,
    gamma: float,
) -> np.ndarray:
    """
    Discounted returns with bootstrap, respecting episode boundaries.

    When done[t] is True the episode ended at step t; we do not bootstrap
    across that boundary (set R=0 before accumulating).
    """
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

def gradient_diagnostics(model: RecurrentActorCritic) -> dict:
    """
    Compute gradient norm and flag NaN / Inf values.

    Spec requirement: check for exploding gradients, vanishing gradients,
    and NaNs before every optimizer step.
    """
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


def params_have_nan(model: RecurrentActorCritic) -> bool:
    return any(torch.isnan(p).any() for p in model.parameters())


# ── Training loop ──────────────────────────────────────────────────────────────

def train(config: Config = None) -> RecurrentActorCritic:
    if config is None:
        config = Config()

    device = torch.device("cpu")

    env = Figure8TMazeEnv(
        max_trials_per_episode=config.max_trials_per_episode,
    )
    # Start with Stage 1 reward shaping
    current_stage = 1
    apply_stage(env, current_stage)

    model = RecurrentActorCritic(
        obs_dim=config.obs_dim,
        hidden_size=config.hidden_size,
        num_actions=config.num_actions,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Running environment state
    obs, _ = env.reset()
    hidden = model.init_hidden(device=device)
    current_ep_reward = 0.0

    # Logging buffers (last 100 episodes)
    ep_rewards = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)

    rollout_idx = 0
    total_steps = 0
    current_ep_steps = 0

    # ── CSV logging setup ──────────────────────────────────────────────────
    log_path = "training_log.csv"
    log_fields = [
        "steps", "rollout", "stage",
        "mean_return", "mean_episode_length",
        "policy_loss", "value_loss", "entropy",
        "grad_norm",
    ]
    log_file = open(log_path, "w", newline="")
    csv_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    csv_writer.writeheader()
    log_file.flush()

    print(f"Training for {config.num_train_steps:,} steps")
    print(f"  hidden_size={config.hidden_size}  rollout={config.rollout_length}  "
          f"lr={config.lr}  gamma={config.gamma}")
    print(f"  metrics → {log_path}")

    while total_steps < config.num_train_steps:

        # ── Curriculum stage check ─────────────────────────────────────────
        new_stage = get_stage(total_steps, config)
        if new_stage != current_stage:
            current_stage = new_stage
            apply_stage(env, current_stage)
            print(f"  [curriculum] → Stage {current_stage} at step {total_steps:,}")

        # ── Collect rollout ────────────────────────────────────────────────
        # Detach hidden at each rollout boundary → truncated BPTT
        hidden = (hidden[0].detach(), hidden[1].detach())

        obs_tensors = []
        actions_list = []
        rewards_list = []
        dones_list = []
        values_list = []
        log_probs_list = []
        entropies_list = []

        for _ in range(config.rollout_length):
            obs_t = obs_to_tensor(obs, device=device)
            logits, value, new_hidden = model(obs_t, hidden)

            dist = Categorical(logits=logits)
            action = dist.sample()

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            obs_tensors.append(obs_t)
            actions_list.append(action)
            rewards_list.append(float(reward))
            dones_list.append(done)
            values_list.append(value)
            log_probs_list.append(dist.log_prob(action))
            entropies_list.append(dist.entropy())

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
                # Reset hidden at episode boundary — no gradient flows across episodes
                hidden = model.init_hidden(device=device)
            else:
                obs = next_obs

        # ── Bootstrap value ────────────────────────────────────────────────
        with torch.no_grad():
            _, last_val, _ = model(obs_to_tensor(obs, device=device), hidden)
            last_val_scalar = last_val.item()

        # ── Returns and advantages ─────────────────────────────────────────
        returns_np = compute_returns(
            rewards_list, dones_list, last_val_scalar, config.gamma
        )
        returns = torch.tensor(returns_np, dtype=torch.float32, device=device)

        values_t = torch.stack(values_list).squeeze(-1).squeeze(-1)   # (T,)
        log_probs_t = torch.stack(log_probs_list)                      # (T,)
        entropies_t = torch.stack(entropies_list)                      # (T,)

        advantages = returns - values_t.detach()
        advantages = advantages - advantages.mean()
        if advantages.std() > 1e-7:
            advantages = advantages / (advantages.std() + 1e-8)

        # ── Losses ────────────────────────────────────────────────────────
        policy_loss = -(log_probs_t * advantages).mean()
        value_loss = F.mse_loss(values_t, returns)
        entropy_loss = -entropies_t.mean()

        loss = (
            policy_loss
            + config.value_loss_coef * value_loss
            + config.entropy_coef * entropy_loss
        )

        if torch.isnan(loss):
            print(f"WARNING: NaN loss at step {total_steps}. Resetting episode state.")
            obs, _ = env.reset()
            hidden = model.init_hidden(device=device)
            current_ep_reward = 0.0
            continue

        # ── Optimization step ──────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()

        grad_info = gradient_diagnostics(model)
        if grad_info["has_nan"] or grad_info["has_inf"]:
            print(f"WARNING: NaN/Inf gradients at step {total_steps}!")

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if params_have_nan(model):
            print(f"ERROR: NaN in model parameters at step {total_steps}. Stopping.")
            break

        rollout_idx += 1

        # ── Logging (CSV every rollout, console every log_interval) ────────
        mean_ret = np.mean(ep_rewards) if ep_rewards else 0.0
        mean_len = np.mean(ep_lengths) if ep_lengths else 0.0

        csv_writer.writerow({
            "steps":               total_steps,
            "rollout":             rollout_idx,
            "stage":               current_stage,
            "mean_return":         round(mean_ret, 4),
            "mean_episode_length": round(mean_len, 1),
            "policy_loss":         round(policy_loss.item(), 6),
            "value_loss":          round(value_loss.item(), 6),
            "entropy":             round(entropies_t.mean().item(), 6),
            "grad_norm":           round(grad_info["grad_norm"], 6),
        })
        log_file.flush()

        if rollout_idx % config.log_interval == 0:
            print(
                f"steps={total_steps:8d}  stage={current_stage}  "
                f"return={mean_ret:6.2f}  "
                f"ploss={policy_loss.item():7.4f}  "
                f"vloss={value_loss.item():6.4f}  "
                f"ent={entropies_t.mean().item():.3f}  "
                f"gnorm={grad_info['grad_norm']:.3f}"
            )

        # ── Checkpoint ────────────────────────────────────────────────────
        if rollout_idx % config.eval_interval == 0:
            torch.save(model.state_dict(), config.checkpoint_path)
            print(f"  [checkpoint saved at step {total_steps}]")

    torch.save(model.state_dict(), config.checkpoint_path)
    log_file.close()
    print(f"\nTraining complete. Checkpoint: {config.checkpoint_path} | Log: {log_path}")
    return model


if __name__ == "__main__":
    train()
