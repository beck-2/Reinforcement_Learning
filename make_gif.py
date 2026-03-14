"""
Generate a GIF of the trained agent navigating the Figure-8 maze.

Usage:
    .venv/bin/python3 make_gif.py
    .venv/bin/python3 make_gif.py --checkpoint checkpoint.pt --out agent.gif --fps 8
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image

from figure8_maze_env import Figure8TMazeEnv
from model import RecurrentActorCritic
from config import Config
from train import obs_to_tensor


def run_and_capture(model_or_none, config: Config, seed: int = 42, max_steps: int = 2000):
    env = Figure8TMazeEnv(
        max_trials_per_episode=config.max_trials_per_episode,
        step_cost=0.0,
        turn_cost=0.0,
        render_mode="rgb_array",
    )
    obs, _ = env.reset(seed=seed)

    if model_or_none is not None:
        model_or_none.eval()
        device = next(model_or_none.parameters()).device
        hidden = model_or_none.init_hidden(device=device)

    frames = []
    done = False
    steps = 0

    while not done and steps < max_steps:
        frame = env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame))

        if model_or_none is None:
            action = env.action_space.sample()
        else:
            obs_t = obs_to_tensor(obs, device=device)
            with torch.no_grad():
                logits, _, hidden = model_or_none(obs_t, hidden)
            from torch.distributions import Categorical
            action = Categorical(logits=logits).sample().item()

        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    # Capture final frame
    frame = env.render()
    if frame is not None:
        frames.append(Image.fromarray(frame))

    env.close()
    return frames, info.get("accuracy", 0.0), steps


def save_gif(frames, out_path: str, fps: int = 8):
    if not frames:
        print("No frames captured.")
        return
    duration_ms = int(1000 / fps)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )
    print(f"Saved GIF: {out_path}  ({len(frames)} frames, {fps} fps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--out", default="agent.gif")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()

    config = Config()

    model = RecurrentActorCritic(
        obs_dim=config.obs_dim,
        hidden_size=config.hidden_size,
        num_actions=config.num_actions,
    )
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"Checkpoint not found ({args.checkpoint}) — using untrained model.")

    print(f"Capturing episode (seed={args.seed}, max_steps={args.max_steps})...")
    frames, accuracy, steps = run_and_capture(model, config, seed=args.seed, max_steps=args.max_steps)
    print(f"Episode: {steps} steps, accuracy={accuracy:.1%}")

    save_gif(frames, args.out, fps=args.fps)


if __name__ == "__main__":
    main()
