"""
ppo.py — Recurrent PPO-Clip with GAE for the Figure-8 T-Maze task.

Design decisions documented in DESIGN_DECISIONS.md.

Usage:
    python ppo.py            # full 2M-step curriculum run
    python ppo.py --steps 500000 --stage 1   # stage 1 only
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from config import PPOConfig
from figure8_maze_env import Figure8TMazeEnv
from model import RecurrentActorCritic


# ---------------------------------------------------------------------------
# Curriculum stage configurations
# ---------------------------------------------------------------------------

STAGE_CONFIGS: dict[int, dict] = {
    1: dict(
        step_cost=0.0,
        turn_cost=0.0,
        correct_reward=1.0,
        incorrect_reward=0.0,
        foraging_reward=0.5,
        loop_bonus=0.2,
        use_stage1_barriers=True,
        force_alternation_barriers=True,
        wall_bump_penalty=-0.005,
        potential_shaping_coef=0.05,
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
        wall_bump_penalty=0.0,
        potential_shaping_coef=0.0,
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
        wall_bump_penalty=0.0,
        potential_shaping_coef=0.0,
    ),
}


def make_env(cfg: PPOConfig, stage: int) -> Figure8TMazeEnv:
    """Create a new env for the given stage (used in tests and initial setup)."""
    sc = STAGE_CONFIGS[stage]
    return Figure8TMazeEnv(
        max_trials_per_episode=cfg.max_trials_per_episode,
        **sc,
    )


def apply_stage(env: Figure8TMazeEnv, stage: int) -> None:
    """Update env reward/barrier attributes in-place for the given curriculum stage."""
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
    env.potential_shaping_coef = cfg["potential_shaping_coef"]


# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------

def encode_obs(obs_dict: dict) -> torch.Tensor:
    """
    Convert env obs dict → flat (6,) float tensor:
      [x/14, y/14, dir_0, dir_1, dir_2, dir_3]
    """
    pos = obs_dict["position_vector"].astype(np.float32) / 14.0
    d = int(obs_dict["direction"])
    dir_oh = np.zeros(4, dtype=np.float32)
    dir_oh[d] = 1.0
    return torch.tensor(np.concatenate([pos, dir_oh]), dtype=torch.float32)


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,       # (T,)
    values: torch.Tensor,        # (T,) — V(s_t)
    next_value: torch.Tensor,    # scalar — V(s_{T+1})
    dones: torch.Tensor,         # (T,) bool — 1 if episode ended after step t
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE advantages and discounted returns.

    Episode boundaries (dones=1) zero out the bootstrap value, so the
    recurrent hidden state reset at those boundaries is consistent.

    Returns:
        advantages : (T,) advantage estimates
        returns    : (T,) = advantages + values  (used as critic targets)
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T, dtype=torch.float32)
    gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value.item()
            mask = 1.0 - dones[t].float().item()
        else:
            next_val = values[t + 1].item()
            mask = 1.0 - dones[t].float().item()

        delta = rewards[t].item() + gamma * next_val * mask - values[t].item()
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer(NamedTuple):
    obs: torch.Tensor           # (T, obs_dim)
    actions: torch.Tensor       # (T,) int64
    log_probs: torch.Tensor     # (T,) old log probs
    values: torch.Tensor        # (T,)
    advantages: torch.Tensor    # (T,) normalized
    returns: torch.Tensor       # (T,) critic targets
    dones: torch.Tensor         # (T,) bool — episode boundaries
    # Hidden state at the START of each episode segment (for recurrent replay)
    episode_starts: list        # list of (step_idx, actor_h, critic_h)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(
    env: Figure8TMazeEnv,
    model: RecurrentActorCritic,
    hidden: tuple,
    obs: torch.Tensor,
    cfg: PPOConfig,
    device: torch.device,
) -> tuple[RolloutBuffer, tuple, torch.Tensor, dict]:
    """
    Collect `cfg.rollout_length` steps from the environment.

    Does NOT reset the env at the start — the caller owns the env state and
    passes in the current obs.  Episodes span rollout boundaries naturally.
    Resets hidden state at episode boundaries.  Records the hidden state at
    the beginning of each new episode so PPO update can replay sequentially.

    Returns: (buffer, hidden, next_obs, stats)
    """
    T = cfg.rollout_length

    obs_buf = torch.zeros(T, cfg.obs_dim)
    act_buf = torch.zeros(T, dtype=torch.long)
    lp_buf = torch.zeros(T)
    val_buf = torch.zeros(T)
    rew_buf = torch.zeros(T)
    done_buf = torch.zeros(T, dtype=torch.bool)

    episode_starts: list = []  # (t, actor_h_snapshot, critic_h_snapshot)

    # Track stats
    ep_rewards: list[float] = []
    ep_lengths: list[int] = []
    ep_correct: list[int] = []
    ep_trials: list[int] = []
    cur_ep_reward = 0.0
    cur_ep_len = 0

    # Record hidden state at the start of this rollout segment
    episode_starts.append((0, RecurrentActorCritic.detach_hidden(hidden)))

    model.eval()
    with torch.no_grad():
        for t in range(T):
            obs_buf[t] = obs

            logits, value, new_hidden = model(obs.unsqueeze(0), hidden)
            dist = Categorical(logits=logits[0])
            action = dist.sample()
            log_prob = dist.log_prob(action)

            act_buf[t] = action
            lp_buf[t] = log_prob
            val_buf[t] = value[0]

            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            rew_buf[t] = reward
            done_buf[t] = done

            cur_ep_reward += reward
            cur_ep_len += 1

            if done:
                ep_rewards.append(cur_ep_reward)
                ep_lengths.append(cur_ep_len)
                ep_correct.append(info.get("correct_trials", 0))
                ep_trials.append(info.get("trial_count", 1))
                cur_ep_reward = 0.0
                cur_ep_len = 0

                obs_dict, _ = env.reset()
                obs = encode_obs(obs_dict).to(device)
                hidden = model.init_hidden(device=device)
                if t + 1 < T:
                    episode_starts.append((t + 1, RecurrentActorCritic.detach_hidden(hidden)))
            else:
                hidden = RecurrentActorCritic.detach_hidden(new_hidden)
                obs = encode_obs(obs_dict).to(device)

        # Bootstrap value for last step (obs is now the next obs after the rollout)
        _, next_val, _ = model(obs.unsqueeze(0), hidden)
        next_value = next_val[0].detach()

    # Compute GAE
    advantages, returns = compute_gae(
        rew_buf, val_buf, next_value, done_buf, cfg.gamma, cfg.gae_lambda
    )

    # Normalize advantages
    adv_mean = advantages.mean()
    adv_std = advantages.std()
    if adv_std > 1e-7:
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    stats = {
        "mean_ep_reward": np.mean(ep_rewards) if ep_rewards else 0.0,
        "mean_ep_length": np.mean(ep_lengths) if ep_lengths else 0.0,
        "mean_alt_accuracy": (
            np.sum(ep_correct) / max(np.sum(ep_trials), 1)
        ),
        "n_episodes": len(ep_rewards),
    }

    buf = RolloutBuffer(
        obs=obs_buf,
        actions=act_buf,
        log_probs=lp_buf,
        values=val_buf,
        advantages=advantages,
        returns=returns,
        dones=done_buf,
        episode_starts=episode_starts,
    )
    return buf, hidden, obs, stats


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    model: RecurrentActorCritic,
    buf: RolloutBuffer,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    cfg: PPOConfig,
    device: torch.device,
) -> dict:
    """
    Run `cfg.ppo_epochs` passes over the rollout, processing episode segments
    sequentially (required for correct LSTM gradient flow).

    Returns a dict of mean loss scalars for logging.
    """
    model.train()

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_clip_frac = 0.0
    total_approx_kl = 0.0
    n_updates = 0

    obs = buf.obs.to(device)
    actions = buf.actions.to(device)
    old_lp = buf.log_probs.to(device)
    advantages = buf.advantages.to(device)
    returns = buf.returns.to(device)
    dones = buf.dones.to(device)

    # Build episode segments from episode_starts
    # Each segment: [start, end) where end = next start (or T)
    T = obs.shape[0]
    seg_starts = [ep[0] for ep in buf.episode_starts]
    seg_ends = seg_starts[1:] + [T]
    segments = list(zip(seg_starts, seg_ends, buf.episode_starts))

    for _epoch in range(cfg.ppo_epochs):
        for (start, end, ep_record) in segments:
            seg_len = end - start
            if seg_len == 0:
                continue

            # Restore hidden state from rollout collection
            _, saved_hidden = ep_record
            (h_a, c_a), (h_c, c_c) = saved_hidden
            hidden = (
                (h_a.to(device), c_a.to(device)),
                (h_c.to(device), c_c.to(device)),
            )

            seg_obs = obs[start:end]           # (seg_len, obs_dim)
            seg_act = actions[start:end]       # (seg_len,)
            seg_old_lp = old_lp[start:end]     # (seg_len,)
            seg_adv = advantages[start:end]    # (seg_len,)
            seg_ret = returns[start:end]       # (seg_len,)
            seg_val = buf.values[start:end].to(device)

            # Forward pass — full sequence through LSTM
            logits, values, _ = model(seg_obs, hidden)

            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(seg_act)    # (seg_len,)
            entropy = dist.entropy().mean()

            # PPO-Clip policy loss
            ratio = torch.exp(new_lp - seg_old_lp)
            surr1 = ratio * seg_adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * seg_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Clipped value loss (Schulman et al. 2017 implementation detail)
            val_clipped = seg_val + torch.clamp(values - seg_val, -cfg.clip_eps, cfg.clip_eps)
            value_loss = torch.max(
                (values - seg_ret).pow(2),
                (val_clipped - seg_ret).pow(2),
            ).mean()

            actor_loss = policy_loss - cfg.entropy_coef * entropy
            critic_loss = cfg.value_loss_coef * value_loss

            actor_opt.zero_grad()
            actor_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.actor.parameters(), cfg.grad_clip)
            actor_opt.step()

            critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(model.critic.parameters(), cfg.grad_clip)
            critic_opt.step()

            with torch.no_grad():
                clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()
                approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_clip_frac += clip_frac
            total_approx_kl += approx_kl
            n_updates += 1

    denom = max(n_updates, 1)
    return {
        "policy_loss": total_policy_loss / denom,
        "value_loss": total_value_loss / denom,
        "entropy": total_entropy / denom,
        "clip_frac": total_clip_frac / denom,
        "approx_kl": total_approx_kl / denom,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: PPOConfig | None = None, start_stage: int = 1, max_steps: int | None = None):
    if cfg is None:
        cfg = PPOConfig()

    device = torch.device("cpu")  # MPS not supported for LSTM sequences reliably
    model = RecurrentActorCritic(cfg.obs_dim, cfg.hidden_size, cfg.num_actions).to(device)

    actor_opt = torch.optim.Adam(model.actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(model.critic.parameters(), lr=cfg.critic_lr)

    total_steps = max_steps or cfg.num_train_steps

    # Determine initial stage and build env
    current_stage = start_stage
    env = make_env(cfg, current_stage)

    hidden = model.init_hidden(device=device)
    obs_dict, _ = env.reset()
    obs = encode_obs(obs_dict).to(device)

    # CSV logging
    csv_path = "training_log_ppo.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=[
        "step", "rollout", "stage",
        "mean_ep_reward", "mean_ep_length", "mean_alt_accuracy", "n_episodes",
        "policy_loss", "value_loss", "entropy", "clip_frac", "approx_kl",
        "elapsed_s",
    ])
    writer.writeheader()
    csv_file.flush()

    rollout_idx = 0
    steps_done = 0
    t0 = time.time()

    print(f"PPO training: {total_steps:,} steps | hidden={cfg.hidden_size} | "
          f"rollout={cfg.rollout_length} | device={device}")

    while steps_done < total_steps:
        # Stage transition
        new_stage = current_stage
        if steps_done >= cfg.stage3_start:
            new_stage = 3
        elif steps_done >= cfg.stage2_start:
            new_stage = 2

        if new_stage != current_stage:
            print(f"\n[{steps_done:,}] Transitioning to stage {new_stage}")
            current_stage = new_stage
            apply_stage(env, current_stage)
            hidden = model.init_hidden(device=device)
            obs_dict, _ = env.reset()
            obs = encode_obs(obs_dict).to(device)

        # Collect rollout
        buf, hidden, obs, collect_stats = collect_rollout(env, model, hidden, obs, cfg, device)
        steps_done += cfg.rollout_length

        # PPO update
        update_stats = ppo_update(model, buf, actor_opt, critic_opt, cfg, device)

        rollout_idx += 1
        elapsed = time.time() - t0

        # Logging
        if rollout_idx % cfg.log_interval == 0:
            row = {
                "step": steps_done,
                "rollout": rollout_idx,
                "stage": current_stage,
                "elapsed_s": f"{elapsed:.1f}",
                **{k: f"{v:.4f}" for k, v in collect_stats.items()},
                **{k: f"{v:.6f}" for k, v in update_stats.items()},
            }
            writer.writerow(row)
            csv_file.flush()

            print(
                f"[{steps_done:>8,}] stage={current_stage} "
                f"reward={collect_stats['mean_ep_reward']:.3f} "
                f"alt={collect_stats['mean_alt_accuracy']:.2%} "
                f"pi={update_stats['policy_loss']:.5f} "
                f"v={update_stats['value_loss']:.4f} "
                f"ent={update_stats['entropy']:.3f} "
                f"kl={update_stats['approx_kl']:.5f} "
                f"clip={update_stats['clip_frac']:.3f} "
                f"eps={collect_stats['n_episodes']}"
            )

        # Checkpoint
        if rollout_idx % cfg.eval_interval == 0:
            torch.save(
                {"model_state": model.state_dict(), "step": steps_done, "stage": current_stage},
                cfg.checkpoint_path,
            )

    # Final checkpoint
    torch.save(
        {"model_state": model.state_dict(), "step": steps_done, "stage": current_stage},
        cfg.checkpoint_path,
    )
    csv_file.close()
    env.close()
    print(f"\nDone. {steps_done:,} steps in {time.time() - t0:.1f}s → {cfg.checkpoint_path}")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=None,
                        help="Override total training steps")
    parser.add_argument("--stage", type=int, default=1,
                        help="Starting curriculum stage (1/2/3)")
    args = parser.parse_args()

    train(start_stage=args.stage, max_steps=args.steps)
