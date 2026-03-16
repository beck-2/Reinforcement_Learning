"""
config.py — PPO hyperparameters for Figure-8 T-Maze training.

Design decisions are documented in DESIGN_DECISIONS.md.
"""
from dataclasses import dataclass, field


@dataclass
class PPOConfig:
    # --- Network ---
    hidden_size: int = 128        # LSTM + encoder hidden dim (2× A2C baseline)
    obs_dim: int = 6              # position (2,) + one-hot direction (4,) = 6
    num_actions: int = 3          # forward, turn-left, turn-right

    # --- PPO Clip ---
    clip_eps: float = 0.2         # standard PPO-Clip epsilon
    ppo_epochs: int = 4           # number of passes over each rollout
    gae_lambda: float = 0.95      # GAE λ — low variance, slight bias
    gamma: float = 0.97           # discount factor (same as A2C baseline)
    value_loss_coef: float = 0.5  # critic loss weight
    entropy_coef: float = 0.01    # small entropy bonus to prevent premature collapse
    grad_clip: float = 0.5        # global gradient norm clip

    # --- Optimizers (separate for actor and critic) ---
    actor_lr: float = 3e-4
    critic_lr: float = 3e-5       # 10× slower — prevents critic from dominating

    # --- Rollout & Training ---
    rollout_length: int = 2048    # steps per rollout (longer = lower variance GAE)
    num_train_steps: int = 2_000_000
    max_trials_per_episode: int = 50

    # --- Curriculum stage boundaries ---
    stage2_start: int = 500_000
    stage3_start: int = 1_000_000

    # --- Reward shaping (Stage 1 only) ---
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    foraging_reward: float = 0.5
    loop_bonus: float = 0.2
    wall_bump_penalty: float = -0.005
    potential_shaping_coef: float = 0.05
    step_cost: float = 0.0

    # --- Logging ---
    log_interval: int = 10        # log every N rollouts
    eval_interval: int = 1000     # evaluate every N rollouts
    eval_episodes: int = 10
    checkpoint_path: str = "checkpoint_ppo.pt"
