import torch
import torch.nn as nn


class RecurrentActorCritic(nn.Module):
    """
    Vanilla RNN actor-critic for the continuous alternation task.

    Architecture:
        obs → Linear(obs_dim, hidden) → Tanh
            → RNN(hidden, hidden)
            → actor head: Linear(hidden, num_actions)
            → critic head: Linear(hidden, 1)

    The recurrent state is the only source of memory — the observation
    contains no explicit previous-choice signal, so the RNN must learn
    to remember trial history in order to alternate correctly.
    """

    def __init__(self, obs_dim: int, hidden_size: int, num_actions: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
        )
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=False)
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                gain = 0.01 if module is self.actor else 1.0
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        obs: torch.Tensor,       # (B, obs_dim)
        hidden: torch.Tensor,    # (1, B, hidden_size)
    ):
        """
        Single-step forward pass.

        Returns:
            logits:     (B, num_actions)
            value:      (B, 1)
            new_hidden: (1, B, hidden_size)
        """
        x = self.encoder(obs.unsqueeze(0))            # (1, B, hidden_size)
        rnn_out, new_hidden = self.rnn(x, hidden)     # (1, B, hidden_size)
        h = rnn_out.squeeze(0)                        # (B, hidden_size)
        return self.actor(h), self.critic(h), new_hidden

    def init_hidden(self, batch_size: int = 1, device=None) -> torch.Tensor:
        """Return a zeroed hidden state for the start of an episode."""
        device = device or next(self.parameters()).device
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
