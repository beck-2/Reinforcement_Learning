import torch
import torch.nn as nn


class SSRRecurrentActorCritic(nn.Module):
    """
    Recurrent actor-critic with a successor representation (SR) head.

    Observations are encoded into a feature vector phi(s). A GRU integrates
    memory over time, and the SR head predicts expected discounted future
    occupancy of those features: M(s) = E[sum_t gamma^t phi(s_t)].
    """

    def __init__(self, obs_dim: int, feature_dim: int, hidden_size: int, num_actions: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, feature_dim),
            nn.Tanh(),
        )
        self.rnn = nn.GRU(feature_dim, hidden_size, batch_first=False)
        self.sr_head = nn.Linear(hidden_size, feature_dim)
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                gain = 0.01 if module is self.actor else 1.0
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor):
        """
        Single-step forward pass.

        Args:
            obs: (B, obs_dim)
            hidden: (1, B, hidden_size)

        Returns:
            logits: (B, num_actions)
            value: (B, 1)
            sr_pred: (B, feature_dim)
            new_hidden: (1, B, hidden_size)
            phi: (B, feature_dim)
        """
        phi = self.encoder(obs)                 # (B, feature_dim)
        rnn_in = phi.unsqueeze(0)               # (1, B, feature_dim)
        rnn_out, new_hidden = self.rnn(rnn_in, hidden)
        h = rnn_out.squeeze(0)                  # (B, hidden_size)
        sr_pred = self.sr_head(h)
        return self.actor(h), self.critic(h), sr_pred, new_hidden, phi

    def init_hidden(self, batch_size: int = 1, device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
