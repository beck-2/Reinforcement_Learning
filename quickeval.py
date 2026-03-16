import torch
import numpy as np
from torch.distributions import Categorical
from figure8_maze_env import Figure8TMazeEnv
from ssr_config import SSRConfig
from ssr_model import SSRRecurrentActorCritic
from train_ssr import obs_to_tensor, _valid_action_mask

cfg = SSRConfig()
cfg.use_last_choice  = True
cfg.use_start_flag   = True
cfg.use_stem_sector  = True
cfg.sr_loss_coef     = 0.0
cfg.entropy_coef     = 0.005
cfg.lr               = 3e-4
cfg.grad_clip        = 0.5
cfg.rollout_length   = 256
cfg.gamma            = 0.97
cfg.num_train_steps  = 200_000
cfg.output_dir       = "results/ceiling"

model = SSRRecurrentActorCritic(
    obs_dim=cfg.obs_dim,
    feature_dim=cfg.feature_dim,
    hidden_size=cfg.hidden_size,
    num_actions=cfg.num_actions,
)
model.load_state_dict(torch.load("results/ssr_rnn/checkpoint.pt", weights_only=True))
model.eval()

print(f"obs_dim={cfg.obs_dim}")
print(f"actor std={model.actor.weight.std().item():.4f}")

env = Figure8TMazeEnv(max_trials_per_episode=50, step_cost=0.0, turn_cost=0.0)
accuracies = []

for ep in range(20):
    obs, _ = env.reset()
    hidden = model.init_hidden()
    done = False

    while not done:
        obs_t = obs_to_tensor(obs, cfg)
        with torch.no_grad():
            logits, _, _, hidden, _ = model(obs_t, hidden)

        mask, _ = _valid_action_mask(env, logits.device)
        masked_logits = logits.masked_fill(~mask.unsqueeze(0), -1e9)
        action = masked_logits.argmax(dim=-1).item()

        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    accuracies.append(info["accuracy"])
    print(f"ep{ep+1:2d} accuracy={info['accuracy']:.2f} trials={info['trial_count']}")

print(f"\nMean accuracy: {np.mean(accuracies):.1%} ± {np.std(accuracies):.1%}")
