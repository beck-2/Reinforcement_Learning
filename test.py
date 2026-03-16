"""
Train a successor representation (SR) recurrent agent on the Figure-8 maze.
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
from constants import (
    MAZE_SIZE,
    START_POS,
    CHOICE_Y,
    STEM_X,
    LEFT_RETURN_X,
    RIGHT_RETURN_X,
    SECTOR_1_RANGE,
    SECTOR_2_RANGE,
    SECTOR_3_RANGE,
    SECTOR_4_RANGE,
)
from ssr_config import SSRConfig
from ssr_model import SSRRecurrentActorCritic

_DIR_TO_VEC = {0: (1,0), 1: (0,1), 2: (-1,0), 3: (0,-1)}

def _forward_open(env, direction):
    x, y = env.agent_pos
    dx, dy = _DIR_TO_VEC[direction]
    cell = env.grid.get(x+dx, y+dy)
    return cell is None or cell.can_overlap()

def _valid_action_mask(env, device):
    d = env.agent_dir
    if _forward_open(env, d):
        return torch.tensor([0, 0, 1], dtype=torch.bool, device=device), False
    left_open  = _forward_open(env, (d-1) % 4)
    right_open = _forward_open(env, (d+1) % 4)
    decision = left_open and right_open
    return torch.tensor([left_open, right_open, 0], dtype=torch.bool, device=device), decision

def obs_to_tensor(obs: dict, config: SSRConfig, device=None) -> torch.Tensor:
    pos = obs["position_vector"] / MAZE_SIZE
    d = int(obs["direction"])
    dir_oh = np.zeros(4, dtype=np.float32)
    dir_oh[d] = 1.0
    parts = [pos.astype(np.float32), dir_oh]
    if config.use_last_choice:
        last_choice = float(obs["last_choice"]) / 2.0
        parts.append(np.array([last_choice], dtype=np.float32))
    if config.use_start_flag:
        at_start = 1.0 if (int(obs["position_vector"][0]) == START_POS[0] and
                           int(obs["position_vector"][1]) == START_POS[1]) else 0.0
        parts.append(np.array([at_start], dtype=np.float32))
    if config.use_stem_sector:
        x, y = obs["position_vector"]
        if int(x) == STEM_X:
            if   SECTOR_1_RANGE[0] <= y <= SECTOR_1_RANGE[1]: sector = 1
            elif SECTOR_2_RANGE[0] <= y <= SECTOR_2_RANGE[1]: sector = 2
            elif SECTOR_3_RANGE[0] <= y <= SECTOR_3_RANGE[1]: sector = 3
            elif SECTOR_4_RANGE[0] <= y <= SECTOR_4_RANGE[1]: sector = 4
            else: sector = 0
        else:
            sector = 0
        parts.append(np.array([sector / 4.0], dtype=np.float32))
    vec = np.concatenate(parts)
    t = torch.tensor(vec, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t.unsqueeze(0)

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

def gradient_diagnostics(model):
    total_sq = 0.0
    has_nan = False
    has_inf = False
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            n = p.grad.data.norm(2).item()
            norms.append(n)
            total_sq += n * n
            if torch.isnan(p.grad).any(): has_nan = True
            if torch.isinf(p.grad).any(): has_inf = True
    return {
        "grad_norm": total_sq ** 0.5,
        "grad_max": max(norms) if norms else 0.0,
        "grad_min": min(norms) if norms else 0.0,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }

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

    obs, _ = env.reset()
    hidden = model.init_hidden(device=device)
    current_ep_reward = 0.0
    ep_rewards = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)
    rollout_idx = 0
    total_steps = 0
    current_ep_steps = 0

    log_fields = [
        "steps", "rollout", "mean_return", "mean_episode_length",
        "policy_loss", "value_loss", "entropy",
        "sr_loss", "reward_pred_loss", "memory_loss", "grad_norm",
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
        if isinstance(hidden, tuple):
            hidden = (hidden[0].detach(), hidden[1].detach())
        else:
            hidden = hidden.detach()

        rewards_list        = []
        dones_list          = []
        values_list         = []
        log_probs_list      = []
        entropies_list      = []
        sr_list             = []
        phi_list            = []
        decision_log_probs_list = []
        decision_indices        = []
        memory_hiddens      = []
        memory_labels       = []

        prev_trial_count = env.trial_count

        shaping_count = 0
        for _ in range(config.rollout_length):
            obs_t = obs_to_tensor(obs, config, device=device)
            logits, value, sr_pred, new_hidden, phi = model(obs_t, hidden)

            mask, decision = _valid_action_mask(env, device)
            masked_logits = logits.masked_fill(~mask.unsqueeze(0), -1e9)
            dist_masked = Categorical(logits=masked_logits)
            dist_raw    = Categorical(logits=logits)
            action = dist_masked.sample()

            # Track decision steps with raw log probs
            if decision:
                decision_log_probs_list.append(dist_raw.log_prob(action))
                decision_indices.append(len(rewards_list))

            log_probs_list.append(dist_masked.log_prob(action))
            entropies_list.append(dist_raw.entropy())

            # Memory supervision: at start of stem, hidden must encode last_choice
            x, y = int(obs["position_vector"][0]), int(obs["position_vector"][1])
            if x == START_POS[0] and y == START_POS[1]:
                lc = int(obs["last_choice"])
                if lc in (1, 2):
                    if isinstance(new_hidden, tuple):
                        memory_hiddens.append(new_hidden[0].squeeze())
                    else:
                        memory_hiddens.append(new_hidden.squeeze())
                    memory_labels.append(lc - 1)  # 0=left, 1=right

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            # Reward shaping: guide agent toward correct arm at choice row
            lc = int(obs["last_choice"])
            x  = int(obs["position_vector"][0])
            nx = int(next_obs["position_vector"][0])
            ny = int(next_obs["position_vector"][1])

            shaping_magnitude = 0.5 if total_steps < 500_000 else 0.0
            if ny == CHOICE_Y and lc in (1, 2) and nx != LEFT_RETURN_X and nx != RIGHT_RETURN_X:
                shaping_count += 1
                if lc == 1:    # last was left, correct next is right
                    if nx > x: reward += shaping_magnitude   # moving right toward correct arm
                    if nx < x: reward -= shaping_magnitude   # moving left toward wrong arm
                elif lc == 2:  # last was right, correct next is left
                    if nx < x: reward += shaping_magnitude   # moving left toward correct arm
                    if nx > x: reward -= shaping_magnitude   # moving right toward wrong arm

            rewards_list.append(float(reward))
            dones_list.append(done)
            values_list.append(value.squeeze(0))
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

        if rollout_idx % config.log_interval == 0:
            print(f"shaping fired {shaping_count} times last rollout")

        # Bootstrap
        with torch.no_grad():
            obs_boot = obs_to_tensor(obs, config, device=device)
            _, last_val, sr_boot, _, _ = model(obs_boot, hidden)
            last_val_scalar = last_val.item()
            sr_boot = sr_boot.squeeze(0)

        # Returns and advantages
        returns_np = compute_returns(rewards_list, dones_list, last_val_scalar, config.gamma)
        returns    = torch.tensor(returns_np, dtype=torch.float32, device=device)

        values_t    = torch.stack(values_list).squeeze(-1)
        log_probs_t = torch.stack(log_probs_list)
        entropies_t = torch.stack(entropies_list)

        advantages = returns - values_t.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # SR TD loss
        sr_preds  = torch.stack(sr_list)
        phi_t     = torch.stack(phi_list)
        sr_next   = torch.vstack([sr_preds[1:], sr_boot.unsqueeze(0)])
        done_mask = torch.tensor(dones_list, dtype=torch.float32, device=device).unsqueeze(-1)
        sr_target = phi_t + (1.0 - done_mask) * config.gamma * sr_next.detach()
        sr_loss   = F.mse_loss(sr_preds, sr_target)

        # Reward prediction loss — trains w
        reward_target    = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        reward_pred      = (phi_t * model.w).sum(-1)
        reward_pred_loss = F.mse_loss(reward_pred, reward_target)

        # Policy loss — only at decision points, using raw log probs
        if len(decision_log_probs_list) > 0:
            dec_log_probs  = torch.stack(decision_log_probs_list)
            dec_advantages = advantages[decision_indices]
            policy_loss    = -(dec_log_probs * dec_advantages.detach()).mean()
        else:
            policy_loss = torch.tensor(0.0, device=device)

        value_loss   = F.mse_loss(values_t, returns)
        entropy_loss = -entropies_t.mean()

        # Memory supervision loss — forces LSTM to encode last_choice in hidden state
        if len(memory_hiddens) > 0:
            mem_h      = torch.stack(memory_hiddens)
            mem_labels = torch.tensor(memory_labels, dtype=torch.long, device=device)
            memory_loss = F.cross_entropy(model.memory_head(mem_h), mem_labels)
        else:
            memory_loss = torch.tensor(0.0, device=device)

        sr_scale = min(1.0, total_steps / max(1, config.sr_warmup_steps))
        loss = (
            policy_loss
            + config.value_loss_coef * value_loss
            + config.entropy_coef * entropy_loss
            + (config.sr_loss_coef * sr_scale) * sr_loss
            + 0.5 * reward_pred_loss
            + 0.3 * memory_loss
        )

        if torch.isnan(loss):
            print(f"WARNING: NaN loss at step {total_steps}. Resetting.")
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
            "steps":            total_steps,
            "rollout":          rollout_idx,
            "mean_return":      round(mean_ret, 4),
            "mean_episode_length": round(mean_len, 1),
            "policy_loss":      round(policy_loss.item(), 6),
            "value_loss":       round(value_loss.item(), 6),
            "entropy":          round(entropies_t.mean().item(), 6),
            "sr_loss":          round(sr_loss.item(), 6),
            "reward_pred_loss": round(reward_pred_loss.item(), 6),
            "memory_loss":      round(memory_loss.item(), 6),
            "grad_norm":        round(grad_info["grad_norm"], 6),
        })
        log_file.flush()

        if rollout_idx % config.log_interval == 0:
            print(
                f"steps={total_steps:8d}  "
                f"return={mean_ret:6.2f}  "
                f"ploss={policy_loss.item():7.4f}  "
                f"vloss={value_loss.item():6.4f}  "
                f"sr={sr_loss.item():6.4f}  "
                f"rpred={reward_pred_loss.item():.4f}  "
                f"mem={memory_loss.item():.4f}  "
                f"ent={entropies_t.mean().item():.3f}  "
                f"gnorm={grad_info['grad_norm']:.3f}"
            )

        if rollout_idx % config.eval_interval == 0:
            torch.save(model.state_dict(), config.checkpoint_path)
            print(f"  [checkpoint saved at step {total_steps}]")

    torch.save(model.state_dict(), config.checkpoint_path)
    log_file.close()
    print(f"\nTraining complete. Checkpoint: {config.checkpoint_path} | Log: {config.training_log_path}")

    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=2)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = SSRConfig()
    cfg.use_last_choice  = False
    cfg.use_start_flag   = True
    cfg.use_stem_sector  = True
    cfg.sr_loss_coef     = 0.1
    cfg.sr_warmup_steps  = 20_000
    cfg.entropy_coef     = 0.005
    cfg.lr               = 3e-4
    cfg.grad_clip        = 0.5
    cfg.rollout_length   = 256
    cfg.gamma            = 0.97
    cfg.num_train_steps  = 2_000_000
    cfg.output_dir       = "results/no_memory"

    if args.output is not None: cfg.output_dir = args.output
    if args.steps  is not None: cfg.num_train_steps = args.steps
    if args.seed   is not None: cfg.seed = args.seed

    train(cfg)