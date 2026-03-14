# Reinforcement_Learning
Recreating the figure-8 maze experiment with an RL agent.

## Successor Representation RNN Agent
This repo now includes an SR-RNN agent that learns continuous alternation using an RNN for memory and a successor representation head for predictive state features.

Key files:
- `ssr_model.py` — SR-RNN model definition
- `ssr_config.py` — hyperparameters
- `train_ssr.py` — training loop
- `evaluate_ssr.py` — evaluation + random baseline
- `plot_ssr_results.py` — learning curves + baseline comparison
- `collect_trajectories.py` — example trajectories
- `SSR_AGENT.md` — full agent documentation
- `RESULTS.md` — results section template

## Quickstart
Train:
```
python3 train_ssr.py
```

Evaluate:
```
python3 evaluate_ssr.py
```

Plot:
```
python3 plot_ssr_results.py
python3 collect_trajectories.py
```

Results are written to `results/ssr_rnn/`.
