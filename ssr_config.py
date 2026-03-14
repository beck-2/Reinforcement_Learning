from dataclasses import dataclass
from pathlib import Path


@dataclass
class SSRConfig:
    # Model
    hidden_size: int = 128
    feature_dim: int = 64
    obs_dim_base: int = 6   # position (2, normalized) + direction one-hot (4)
    num_actions: int = 3    # turn_left, turn_right, forward

    # Training
    rollout_length: int = 256
    gamma: float = 0.97
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    sr_loss_coef: float = 1.0
    grad_clip: float = 0.5
    num_train_steps: int = 2_000_000
    seed: int = 0

    # Environment — sparse reward per spec
    max_trials_per_episode: int = 50
    step_cost: float = -0.01
    turn_cost: float = -0.01

    # Observation selection
    use_last_choice: bool = False  # Keep False to force RNN memory

    # Logging / outputs
    log_interval: int = 100    # rollouts between console logs
    eval_interval: int = 1000  # rollouts between checkpoints/evals
    eval_episodes: int = 20
    output_dir: str = "results/ssr_rnn"

    @property
    def obs_dim(self) -> int:
        return self.obs_dim_base + (1 if self.use_last_choice else 0)

    @property
    def checkpoint_path(self) -> str:
        return str(Path(self.output_dir) / "checkpoint.pt")

    @property
    def training_log_path(self) -> str:
        return str(Path(self.output_dir) / "training_log.csv")

    @property
    def eval_summary_path(self) -> str:
        return str(Path(self.output_dir) / "eval_summary.json")

    @property
    def baseline_summary_path(self) -> str:
        return str(Path(self.output_dir) / "baseline_summary.json")

    @property
    def trajectory_path(self) -> str:
        return str(Path(self.output_dir) / "trajectory_examples.json")

    @property
    def training_curves_path(self) -> str:
        return str(Path(self.output_dir) / "training_curves.png")

    @property
    def baseline_plot_path(self) -> str:
        return str(Path(self.output_dir) / "baseline_comparison.png")

    @property
    def trajectory_plot_path(self) -> str:
        return str(Path(self.output_dir) / "trajectory_comparison.png")
