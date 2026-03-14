from dataclasses import dataclass


@dataclass
class Config:
    # Model
    hidden_size: int = 64
    obs_dim: int = 6        # position (2, normalized) + direction one-hot (4)
    num_actions: int = 3    # turn_left, turn_right, forward

    # Training
    rollout_length: int = 256
    gamma: float = 0.99
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    grad_clip: float = 0.5
    num_train_steps: int = 2_000_000

    # Environment — sparse reward per spec
    max_trials_per_episode: int = 50
    step_cost: float = 0.0
    turn_cost: float = 0.0

    # Logging
    log_interval: int = 100    # rollouts between console logs
    eval_interval: int = 1000  # rollouts between checkpoints/evals
    eval_episodes: int = 10
    checkpoint_path: str = "checkpoint.pt"
