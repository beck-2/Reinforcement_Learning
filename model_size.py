from model import RecurrentActorCritic
m = RecurrentActorCritic(obs_dim=6, hidden_size=64, num_actions=3)
total = sum(p.numel() for p in m.parameters())
actor_total = sum(p.numel() for p in m.actor.parameters())
critic_total = sum(p.numel() for p in m.critic.parameters())
for name, p in m.named_parameters():
    print(f"  {name:50s} {p.numel():>8,}")
print(f"\nActor:  {actor_total:,}")
print(f"Critic: {critic_total:,}")
print(f"Total:  {total:,}")
