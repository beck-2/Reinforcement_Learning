"""
model.py — Split LSTM actor-critic for recurrent PPO.

Architecture:
  ActorNet  : encoder → LSTM → head (logits over 3 actions)
  CriticNet : encoder → LSTM → head (scalar value)
  RecurrentActorCritic: wraps both, exposes joint forward + init_hidden

Design decisions documented in DESIGN_DECISIONS.md.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> nn.Module:
    """Apply orthogonal initialization to all Linear layers."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            nn.init.zeros_(m.bias)
    return module


class ActorNet(nn.Module):
    """
    Actor network: obs → logits.

    Uses a separate encoder+LSTM so the critic cannot corrupt the policy
    representation.  Forward-bias initialization breaks the uniform-policy
    saddle point that killed A2C.
    """

    def __init__(self, obs_dim: int, hidden_size: int, num_actions: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
        )
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=False)
        self.head = nn.Linear(hidden_size, num_actions)

        _orthogonal_init(self.encoder, gain=1.0)
        _orthogonal_init(self.head, gain=0.5)   # small gain → near-uniform init
        # Forward-bias: make "forward" slightly more likely at init.
        # Breaks symmetry without hard-coding behaviour.
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([-0.5, -0.5, 0.5]))

    def forward(
        self,
        obs: torch.Tensor,                     # (T, obs_dim) or (1, obs_dim)
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            logits  : (T, num_actions)
            new_h   : (h, c) each (1, 1, hidden_size)
        """
        x = self.encoder(obs)                  # (T, hidden_size)
        x = x.unsqueeze(1)                     # (T, 1, hidden_size) — batch=1
        out, new_h = self.rnn(x, hidden)       # out: (T, 1, hidden_size)
        logits = self.head(out.squeeze(1))     # (T, num_actions)
        return logits, new_h

    def init_hidden(
        self, batch_size: int = 1, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (z, z.clone())


class CriticNet(nn.Module):
    """
    Critic network: obs → scalar value.

    Separate from actor so critic's large value gradient does not dominate
    the shared representation.
    """

    def __init__(self, obs_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
        )
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=False)
        self.head = nn.Linear(hidden_size, 1)

        _orthogonal_init(self.encoder, gain=1.0)
        _orthogonal_init(self.head, gain=1.0)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            values  : (T,) scalar values
            new_h   : (h, c) each (1, 1, hidden_size)
        """
        x = self.encoder(obs)
        x = x.unsqueeze(1)
        out, new_h = self.rnn(x, hidden)
        values = self.head(out.squeeze(1)).squeeze(-1)  # (T,)
        return values, new_h

    def init_hidden(
        self, batch_size: int = 1, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (z, z.clone())


class RecurrentActorCritic(nn.Module):
    """
    Joint actor-critic wrapper for recurrent PPO.

    Hidden state shape: ((h_a, c_a), (h_c, c_c))
    where each tensor is (1, batch_size, hidden_size).
    """

    def __init__(self, obs_dim: int, hidden_size: int, num_actions: int):
        super().__init__()
        self.actor = ActorNet(obs_dim, hidden_size, num_actions)
        self.critic = CriticNet(obs_dim, hidden_size)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: tuple,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        Args:
            obs    : (T, obs_dim)
            hidden : ((h_a, c_a), (h_c, c_c))
        Returns:
            logits : (T, num_actions)
            values : (T,)
            new_hidden : same structure as hidden
        """
        actor_h, critic_h = hidden
        logits, new_actor_h = self.actor(obs, actor_h)
        values, new_critic_h = self.critic(obs, critic_h)
        return logits, values, (new_actor_h, new_critic_h)

    def init_hidden(
        self, batch_size: int = 1, device: torch.device | None = None
    ) -> tuple:
        return (
            self.actor.init_hidden(batch_size, device),
            self.critic.init_hidden(batch_size, device),
        )

    @staticmethod
    def detach_hidden(hidden: tuple) -> tuple:
        """Detach all tensors in nested hidden state (for BPTT truncation)."""
        (h_a, c_a), (h_c, c_c) = hidden
        return (
            (h_a.detach(), c_a.detach()),
            (h_c.detach(), c_c.detach()),
        )

    def encode_obs(self, obs_dict: dict) -> torch.Tensor:
        """
        Convert env observation dict to flat tensor.

        obs_dict keys: 'position_vector' (2,), 'direction' (int)
        Output: (6,) = [x/14, y/14, sin(dir*π/2), cos(dir*π/2), ...one-hot dir...]

        We use a 6-dim encoding:
          - x, y normalized by (MAZE_SIZE - 1) = 14
          - one-hot direction (4 values)
        """
        import numpy as np
        pos = obs_dict["position_vector"].astype(np.float32) / 14.0   # (2,)
        d = int(obs_dict["direction"])
        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[d] = 1.0
        return torch.tensor(np.concatenate([pos, dir_onehot]), dtype=torch.float32)
