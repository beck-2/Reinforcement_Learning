# Tech Spec: Recurrent Actor-Critic for Continuous Alternation (MiniGrid)

## Goal
Train a simple recurrent actor-critic agent to solve a **continuous alternation maze** using MiniGrid.  
The agent repeatedly traverses a shared central stem and must **alternate left/right choices** at a T-junction across trials.

Key task property:
- the **same stem is traversed on both trial types**
- correct action depends on **memory of the previous trial**

Primary success criterion:
- **high sustained alternation accuracy**

Representation analysis is out of scope for v1 except for minimal logging hooks.

---

# Environment

## Library
Use **MiniGrid** with a custom environment.

## Maze Structure
Environment must contain:
- central stem
- T-junction
- left and right arms
- return path(s) to the stem start
- continuous traversal across many trials per episode

This replicates the continuous alternation paradigm where the animal repeatedly returns to the stem.

## Observations (Option B)
Use a **simple local navigation representation**, not raw coordinates.

Include:
- MiniGrid observation
- agent direction
- optional compact maze-region identifier if needed

Do **not include**:
- previous turn
- previous reward
- any explicit memory variable

Reason: preserve the task’s internal memory demand.

## Actions (Option B)
Discrete navigation actions everywhere:

- `forward`
- `turn_left`
- `turn_right`

## Episode Structure (Option B)
Each episode contains **multiple consecutive trials/laps**.

Configuration:
- configurable max episode length
- long enough for many alternation attempts

Episodes should **not terminate after each decision**.

## Reward (Option A)
Sparse reward only:

- `+1` for correct alternation
- `0` otherwise

No shaping in v1.

## Error Handling
Incorrect alternation **does not terminate the episode**.  
Agent continues through the maze.

---

# Agent

## Architecture
Simple **recurrent actor-critic**.

Structure:
obs → encoder → RNN → shared representation
↙ ↘
actor critic

### Initial Model
- lightweight MiniGrid-compatible encoder
- **1 recurrent layer**
- **vanilla RNN**
- hidden size: **64**
- actor head: discrete policy over 3 actions
- critic head: scalar value

### Why RNN (not LSTM initially)
Project prioritizes **simplicity over robustness**.

Memory horizon is short, so a vanilla RNN may suffice.

Risk:
- vanishing gradients
- unstable memory retention

Mitigation:
- implement gradient diagnostics
- switch to **GRU or LSTM only if needed**

Upgrade rule:
- if navigation works but alternation memory fails, or gradients collapse/explode → switch to GRU/LSTM.

---

# Training

## Algorithm
Use **A2C (synchronous actor-critic)**.

Reasons:
- simpler than PPO
- adequate for small environments
- easier debugging

## Rollouts / BPTT
Use **truncated backpropagation through time**.

Initial setting:
- rollout length: **64 steps**

Behavior:
- hidden state carried across rollout chunks
- gradients truncated at chunk boundary
- hidden state reset only at episode boundaries

## Hyperparameters
Expose for tuning:

- learning rate
- discount factor
- entropy coefficient
- value loss coefficient
- gradient clipping threshold
- rollout length
- hidden size
- max episode length

Recommended defaults:
hidden_size = 64
rollout_length = 64
gamma = 0.99
gradient_clip = 0.5–1.0
optimizer = Adam
entropy_coef = small nonzero

---

# Logging & Evaluation

## Training Metrics
Log:

- episodic return
- alternation accuracy
- policy loss
- value loss
- entropy
- gradient norm
- episode length

## Behavioral Success Metric
Primary evaluation:

**alternation accuracy across long evaluation episodes**

Success = stable high alternation rate, not just reward spikes.

## Recurrent Diagnostics
Add checks for:

- exploding gradients
- vanishing gradients
- NaNs in loss/parameters
- hidden-state shape consistency
- hidden-state reset on episode termination

---

# Tests

## Environment Tests
Verify:

- maze resets correctly
- correct reward for valid alternation
- no reward for repeating same arm
- episode continues after errors
- repeated traversal loop works

## Model Tests
Verify:

- forward pass tensor shapes
- hidden state update
- hidden state reset between episodes
- actor output size matches action space
- critic outputs scalar value

## Training Health Tests
Smoke tests should confirm:

- gradients remain finite
- losses remain finite
- parameters update during training
- performance improves above random baseline

---

# Deliverables

Claude Code should implement:

1. custom MiniGrid continuous alternation environment
2. recurrent actor-critic model (vanilla RNN)
3. A2C training loop with truncated BPTT
4. config system for hyperparameters
5. logging of training and gradient metrics
6. environment/model unit tests
7. evaluation script measuring alternation accuracy

---

# Non-Goals (v1)

Do **not** include yet:

- GRU/LSTM unless RNN fails
- representation probing
- trial-type decoding
- hippocampal unit analysis
- ablation infrastructure
- successor representation models
- continuous control

---

# Future Plans

## Ablations
Add later:

- remove recurrence
- recurrence + explicit previous-turn observation
- GRU instead of RNN
- LSTM instead of RNN

## Representation Analysis
Later experiments should:

- log hidden states on the stem
- test whether hidden state predicts trial type
- compare to hippocampal trial-type differentiation reported in the paper

## Training Upgrades
Potential improvements:

- switch A2C → PPO for stability
- upgrade RNN → GRU/LSTM if memory weak
- larger seed sweeps and evaluation protocols

---

# Final Implementation Summary

Build the **simplest viable system**:

- MiniGrid custom maze
- discrete navigation actions
- multi-trial continuous episodes
- sparse alternation reward
- **vanilla RNN actor-critic**
- **A2C training**
- truncated BPTT (64 steps)
- strong gradient diagnostics

This is intentionally minimal and should be sufficient for an easy task while leaving room for future neuroscience-focused analysis.

Visualizations:
At minimum, include:
-Learning curve (reward vs episodes or timesteps)
-Example trajectories of trained agent behavior
-Comparison to baseline (random agent or untrained agent)
-Optional (if relevant):
-Value function visualization
-Policy visualization
-State representation visualization (PCA/t-SNE if using neural network representations)