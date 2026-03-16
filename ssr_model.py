import torch
import torch.nn as nn

class SSRRecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim, feature_dim, hidden_size, num_actions):
        super().__init__()
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, feature_dim), nn.Tanh()
        )
        self.rnn = nn.LSTM(feature_dim, hidden_size, batch_first=False)
        self.sr_head = nn.Linear(hidden_size, feature_dim)
        self.w = nn.Parameter(torch.zeros(feature_dim))
        self.actor = nn.Linear(hidden_size, num_actions)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.3)
        nn.init.constant_(self.w, 0.0)

    def forward(self, obs, hidden):
        phi = self.encoder(obs)
        rnn_out, new_hidden = self.rnn(phi.unsqueeze(0), hidden)
        h = rnn_out.squeeze(0)
        sr_pred = self.sr_head(h)
        value = (sr_pred * self.w).sum(-1, keepdim=True)
        return self.actor(h), value, sr_pred, new_hidden, phi

    def init_hidden(self, batch_size=1, device=None):
        device = device or next(self.parameters()).device
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (h0, c0)