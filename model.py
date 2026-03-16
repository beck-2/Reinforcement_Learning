import torch
import torch.nn as nn


class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int, num_actions: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.Tanh())
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=False)
        self.head = nn.Linear(hidden_size, num_actions)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = 1.0 if m is not self.head else 0.5
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)
        # Bias the actor toward "forward" (action 2) at initialization.
        # Breaks the uniform-policy symmetry so the agent starts moving
        # forward through the maze rather than spinning in place.
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([-0.5, -0.5, 0.5]))

    def forward(self, obs, hidden):
        x = self.encoder(obs.unsqueeze(0))
        out, new_hidden = self.rnn(x, hidden)
        return self.head(out.squeeze(0)), new_hidden

    def init_hidden(self, batch_size=1, device=None):
        device = device or next(self.parameters()).device
        z = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (z, z.clone())


class CriticNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.Tanh())
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=False)
        self.head = nn.Linear(hidden_size, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, obs, hidden):
        x = self.encoder(obs.unsqueeze(0))
        out, new_hidden = self.rnn(x, hidden)
        return self.head(out.squeeze(0)), new_hidden

    def init_hidden(self, batch_size=1, device=None):
        device = device or next(self.parameters()).device
        z = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (z, z.clone())


class RecurrentActorCritic(nn.Module):
    """
    Split actor-critic: separate encoder + LSTM + head for each.
    Prevents the critic from hijacking the shared representation.

    hidden = (actor_hidden, critic_hidden)
    each hidden is an LSTM state: ((h, c)) tuple of (1, B, H) tensors
    """

    def __init__(self, obs_dim: int, hidden_size: int, num_actions: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.actor = ActorNet(obs_dim, hidden_size, num_actions)
        self.critic = CriticNet(obs_dim, hidden_size)

    def forward(self, obs, hidden):
        actor_h, critic_h = hidden
        logits, new_actor_h = self.actor(obs, actor_h)
        value, new_critic_h = self.critic(obs, critic_h)
        return logits, value, (new_actor_h, new_critic_h)

    def init_hidden(self, batch_size: int = 1, device=None):
        return (
            self.actor.init_hidden(batch_size, device),
            self.critic.init_hidden(batch_size, device),
        )
