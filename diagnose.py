"""
diagnose.py
Run a short training loop and print detailed diagnostics every rollout.
Checks: logit variance, advantage distribution, gradient flow per layer,
obs distribution, and reward density.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from collections import deque

from figure8_maze_env import Figure8TMazeEnv
from model import RecurrentActorCritic
from train import apply_stage, obs_to_tensor, compute_returns
from config import Config

config = Config()
config.num_train_steps = 20_480  # 20 rollouts
config.rollout_length = 1024
device = torch.device("cpu")

env = Figure8TMazeEnv(max_trials_per_episode=config.max_trials_per_episode)
apply_stage(env, 1)

model = RecurrentActorCritic(config.obs_dim, config.hidden_size, config.num_actions)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

obs, _ = env.reset()
hidden = model.init_hidden(device=device)
current_ep_reward = 0.0

for rollout_idx in range(20):
    hidden = (
        (hidden[0][0].detach(), hidden[0][1].detach()),
        (hidden[1][0].detach(), hidden[1][1].detach()),
    )

    obs_list, actions_list, rewards_list, dones_list = [], [], [], []
    values_list, log_probs_list, entropies_list = [], [], []
    all_logits = []

    for _ in range(config.rollout_length):
        obs_t = obs_to_tensor(obs, device=device)
        logits, value, new_hidden = model(obs_t, hidden)
        all_logits.append(logits.detach())

        dist = Categorical(logits=logits)
        action = dist.sample()

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        obs_list.append(obs_t)
        actions_list.append(action)
        rewards_list.append(float(reward))
        dones_list.append(done)
        values_list.append(value)
        log_probs_list.append(dist.log_prob(action))
        entropies_list.append(dist.entropy())

        hidden = new_hidden
        if done:
            obs, _ = env.reset()
            hidden = model.init_hidden(device=device)
        else:
            obs = next_obs

    with torch.no_grad():
        _, last_val, _ = model(obs_to_tensor(obs, device=device), hidden)

    returns_np = compute_returns(rewards_list, dones_list, last_val.item(), config.gamma)
    returns = torch.tensor(returns_np, dtype=torch.float32)

    values_t = torch.stack(values_list).squeeze(-1).squeeze(-1)
    log_probs_t = torch.stack(log_probs_list)
    entropies_t = torch.stack(entropies_list)

    advantages = returns - values_t.detach()
    adv_mean, adv_std = advantages.mean().item(), advantages.std().item()
    advantages_norm = advantages - advantages.mean()
    if advantages.std() > 1e-7:
        advantages_norm = advantages_norm / (advantages.std() + 1e-8)

    policy_loss = -(log_probs_t * advantages_norm).mean()
    value_loss = F.mse_loss(values_t, returns)
    entropy_loss = -entropies_t.mean()
    loss = policy_loss + config.value_loss_coef * value_loss + config.entropy_coef * entropy_loss

    optimizer.zero_grad()
    loss.backward()

    # --- Diagnostics ---
    logits_stack = torch.cat(all_logits, dim=0)  # (T, 3)
    reward_arr = np.array(rewards_list)
    nonzero_rewards = (reward_arr != 0).sum()

    print(f"\n=== Rollout {rollout_idx+1} ===")
    print(f"  Rewards      : nonzero={nonzero_rewards}/{config.rollout_length}  "
          f"mean={reward_arr.mean():.4f}  min={reward_arr.min():.4f}  max={reward_arr.max():.4f}")
    print(f"  Returns      : mean={returns.mean():.4f}  std={returns.std():.4f}")
    print(f"  Advantages   : mean={adv_mean:.4f}  std={adv_std:.4f}")
    print(f"  Adv (normed) : mean={advantages_norm.mean():.4f}  std={advantages_norm.std():.4f}")
    print(f"  Logits       : mean={logits_stack.mean():.4f}  std={logits_stack.std():.4f}  "
          f"per-action means={logits_stack.mean(0).tolist()}")
    print(f"  Log probs    : mean={log_probs_t.mean():.4f}  std={log_probs_t.std():.4f}")
    print(f"  Entropy      : {entropies_t.mean():.6f}  (max={np.log(3):.6f})")
    print(f"  Policy loss  : {policy_loss.item():.8f}")
    print(f"  Value loss   : {value_loss.item():.6f}")

    # Per-layer gradient norms
    print("  Grad norms   :")
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(f"    {name:40s} {p.grad.norm():.6f}")
        else:
            print(f"    {name:40s} NO GRAD")

    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
