# Successor Representation RNN Agent

## Paper-grounded task summary
The environment mirrors the figure-8 continuous alternation task described in the paper:
- Rats traverse the central stem on every trial, reach a T-junction, and alternate left/right turns.
- Correct alternations are rewarded at water wells at the ends of the choice arms.
- After reward, rats return along the connecting arms and begin the next trial.
- The central stem is divided into four sectors for analyzing trial-type selectivity.

Our agent follows this structure by operating over long episodes (recording sessions) with many trials and learning to alternate using memory.

## Why SR + RNN
Successor representation (SR) provides a predictive map of future state features, while the RNN carries memory of recent choices. This matches the paper’s emphasis on hippocampal representations that distinguish trial types while the animal is in the same location on the stem.

## Action space
Discrete actions (same as environment):
- `0`: turn left
- `1`: turn right
- `2`: move forward

## State space and representation
Environment observation includes:
- `position_vector`: `(x, y)` grid coordinates
- `direction`: facing direction (0=E, 1=S, 2=W, 3=N)
- `image`: full 15x15 RGB rendering
- `last_choice`: last trial’s choice (0/1/2)
- `trial_number`: current trial index

**Agent input (by design):**
- `position_vector` and `direction` only.
- `last_choice` and `trial_number` are deliberately excluded so the RNN must store memory internally.

Input encoding (default):
- Position normalized by `MAZE_SIZE` → 2 floats in `[0,1]`
- Direction as one-hot → 4 floats
- Total `obs_dim = 6`

## SR formulation
Let `phi(s)` be the feature encoding of the current observation. The SR head predicts:

`M(s) = E[ sum_{t>=0} gamma^t * phi(s_t) ]`

TD target for SR learning:

`M_target = phi(s_t) + gamma * M(s_{t+1})` (no bootstrap across episode boundaries)

The SR loss is mean squared error between `M(s_t)` and `M_target`.

## Training objective
Total loss combines:
- Policy loss (A2C)
- Value loss (A2C)
- Entropy bonus (exploration)
- SR TD loss (predictive map)

## Default hyperparameters
Defined in `ssr_config.py`:
- `hidden_size = 128`
- `feature_dim = 64`
- `rollout_length = 256`
- `gamma = 0.97`
- `lr = 3e-4`
- `entropy_coef = 0.01`
- `value_loss_coef = 0.5`
- `sr_loss_coef = 1.0`
- `grad_clip = 0.5`
- `num_train_steps = 2_000_000`
- `max_trials_per_episode = 50`
- `step_cost = -0.01`
- `turn_cost = -0.01`

## How to run
Train:
```
python3 train_ssr.py
```

Evaluate and compare to random baseline:
```
python3 evaluate_ssr.py
```

Make plots and trajectory comparisons:
```
python3 plot_ssr_results.py
python3 collect_trajectories.py
```

Outputs are written to `results/ssr_rnn/` by default.
