from model import RecurrentActorCritic
m = RecurrentActorCritic(obs_dim=6, hidden_size=64, num_actions=3)
total = sum(p.numel() for p in m.parameters())
for name, p in m.named_parameters():
    print(f"  {name:40s} {p.numel():>8,}")
print(f"\nTotal: {total:,} parameters")
