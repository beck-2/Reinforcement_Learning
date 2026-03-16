# PPO Design Decisions — Figure-8 T-Maze

This document records architectural and algorithmic choices made for the recurrent PPO implementation, including the reasoning behind each decision. It is intended for review before any changes to `ppo.py` or `model.py`.

---

## Why PPO instead of A2C?

A2C failed to converge on this task across all tuning attempts. The root cause: under a near-uniform policy, actions and advantages are statistically independent, so the policy gradient is zero in expectation. The critic converges quickly (value loss → 0), but this makes advantages ≈ 0 permanently — a self-reinforcing trap.

**PPO fixes this in two ways:**
1. Multiple epochs per rollout (4 epochs × long rollout = more gradient signal per environment step).
2. Clipped surrogate objective prevents large policy updates that could destroy good samples before they can be reused.

---

## Architecture: Split Actor / Critic LSTMs

**Decision:** ActorNet and CriticNet are fully separate encoder → LSTM → head stacks with no shared weights.

**Why:** In the shared-network A2C run, the critic's value loss was orders of magnitude larger than the policy loss. The shared LSTM gradient was dominated by the critic, preventing the actor from learning its own representation. Separate networks and separate optimizers eliminate this coupling.

**Downside:** 2× parameters. Acceptable because the maze is small and training is CPU-only.

---

## Hidden size: 128

**Decision:** `hidden_size=128` (2× the A2C baseline of 64).

**Why:** The alternation task requires remembering the last choice over tens of steps. 64 hidden units may be insufficient to reliably store this information alongside positional encoding. 128 gives more capacity with modest extra compute.

---

## LSTM over GRU

**Decision:** `nn.LSTM` (not GRU or vanilla RNN).

**Why:** LSTM's separate cell state and input/forget gates give it stronger gradient flow over long sequences — important here because the agent must remember which well it visited last and hold that information across the full stem traversal (O(10) steps minimum). GRU is similar but LSTM has better-understood training dynamics for memory tasks in the literature.

---

## Forward-bias initialization: `head.bias = [-0.5, -0.5, +0.5]`

**Decision:** Initialize the actor head's bias to slightly favor the "forward" action (action index 2).

**Why:** A uniform-logit policy spends most of its time turning in place. The forward bias ensures the agent explores the maze at all, giving the policy gradient non-zero signal from the start. The magnitude (-0.5/+0.5) was chosen empirically: -2/+2 caused the agent to get permanently stuck at walls.

**Scope:** Applied only to the actor head. Critic head stays zero-biased.

---

## Separate learning rates: actor 3e-4, critic 3e-5

**Decision:** Critic learns 10× slower than actor.

**Why:** If the critic converges too quickly, all advantages collapse to zero (the core A2C failure mode). A slower critic keeps advantage signal alive longer, giving the actor time to find good trajectories.

---

## Rollout length: 2048

**Decision:** `rollout_length=2048`.

**Why:** GAE variance decreases with longer rollouts. The alternation task has sparse rewards (one reward every ~20–30 steps), so a 64-step rollout often contains zero reward signals. 2048 steps guarantees multiple rewards per rollout even in early training.

**Tradeoff:** Longer rollouts delay updates. With PPO's multiple epochs (4), we partially compensate by reusing each rollout more.

---

## GAE: λ=0.95, γ=0.97

**Decision:** GAE `lambda=0.95`, discount `gamma=0.97`.

**Why:** λ=0.95 is the standard PPO value from Schulman et al. 2017 — low variance, slight bias. γ=0.97 was inherited from the A2C baseline and matches the task's ~20–30 step episode segments (0.97^30 ≈ 0.40, so distant rewards still matter).

---

## Sequential recurrent replay (no random minibatches)

**Decision:** During each PPO epoch, process episode segments sequentially in order, restoring the LSTM hidden state from the rollout's episode-start snapshots.

**Why:** Standard PPO shuffles random minibatches, but this breaks LSTM causality — a segment processed out of order would receive an incorrect hidden state from a different part of the episode. Sequential replay maintains correct hidden state while still allowing multiple gradient steps per rollout.

**Implementation:** `episode_starts` is a list of `(step_index, saved_hidden)` tuples. Each epoch replays all segments in sequence.

---

## Clipped value loss

**Decision:** `value_loss = max((v - ret)², (v_clipped - ret)²)`.

**Why:** Prevents the critic from making large updates that could destabilize training. Clipping value changes to `[-ε, +ε]` matches the policy clipping magnitude, keeping both networks in sync.

---

## Entropy coefficient: 0.01

**Decision:** `entropy_coef=0.01` (small but nonzero).

**Why:** Zero entropy collapsed the A2C policy prematurely. A small positive coefficient prevents mode-collapse without overpowering the forward bias. In PPO, the clip constraint already limits destructive updates, so entropy tuning is less critical.

---

## Curriculum: 3 stages

| Stage | Steps      | Barriers | Force alternation | Shaping |
|-------|-----------|----------|------------------|---------|
| 1     | 0–500k    | Yes      | Yes              | Yes     |
| 2     | 500k–1M   | Yes      | No               | No      |
| 3     | 1M–2M     | No       | No               | No      |

**Stage 1** scaffolds exploration: physical barriers enforce the correct circuit, foraging reward (+0.5 for any well visit) encourages visiting wells, and potential shaping (+0.05/step toward nearest well) provides dense gradient signal.

**Stage 2** removes the force-alternation top barriers, so the agent must start learning to choose correctly on its own — but flanking barriers still constrain the path.

**Stage 3** removes all barriers: pure alternation task matching the Wood et al. (2000) experimental paradigm.

---

## Potential shaping: Stage 1 only

**Decision:** `potential_shaping_coef=0.05` in Stage 1, zero in Stages 2 and 3.

**Why:** Distance-to-nearest-well shaping is consistent with Ng et al. (1999) potential-based shaping theory (does not change optimal policy). It is removed in later stages to avoid biasing the agent toward the nearest well rather than the correct well.

---

## Wall bump penalty: -0.005 (Stage 1 only)

**Decision:** Small penalty for running into walls or barriers in Stage 1.

**Why:** Without any penalty, the agent wastes time bumping against walls. -0.005 is small enough not to collapse returns (we saw -0.1 collapse them to -128) but large enough to discourage repeated wall contact. Removed in Stages 2/3 where barriers are fewer and step cost (-0.001) is already active.

**Biological justification:** Rats physically feel the maze walls — a tiny aversive signal is appropriate.

---

## No step/turn cost in Stage 1

**Decision:** `step_cost=0.0`, `turn_cost=0.0` in Stage 1; `step_cost=-0.001` from Stage 2 onward.

**Why:** In Stage 1 the agent needs to explore freely to discover the wells. A step cost would incentivize staying still. Stages 2–3 add a tiny step cost (-0.001) to encourage efficiency once the task structure is known.

---

## Observation encoding: position + one-hot direction (6-dim)

**Decision:** `[x/14, y/14, dir_0, dir_1, dir_2, dir_3]` — no explicit memory features.

**Why:** Replicates the information available to a rat: proprioceptive position and heading. The LSTM must learn to maintain working memory of the last choice in its hidden state — which is exactly what we want to study (analogue of hippocampal activity).

---

## No `last_choice` in observation

**Decision:** `last_choice` was explicitly removed from the observation space.

**Why:** Providing it would make the task trivially solvable by a memoryless policy. The agent must internalize this information in its recurrent hidden state — the point of the Wood et al. (2000) replication.
