"""
stage_agent_demo.py
Run the trained checkpoint in a given stage and save a GIF.
Usage: .venv/bin/python3 stage_agent_demo.py <stage> <output_path>
"""

import sys
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.distributions import Categorical

from figure8_maze_env import Figure8TMazeEnv
from model import RecurrentActorCritic
from train import apply_stage, obs_to_tensor
from config import Config
from constants import MAZE_SIZE

stage = int(sys.argv[1])
out_path = sys.argv[2]

TILE = int(480 / MAZE_SIZE)
MAX_STEPS = 3000

config = Config()
device = torch.device("cpu")

model = RecurrentActorCritic(config.obs_dim, config.hidden_size, config.num_actions)
model.load_state_dict(torch.load("checkpoint.pt", map_location=device))
model.eval()


def annotate(raw_frame, env, step, last_reward):
    img = Image.fromarray(raw_frame)
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    # Draw placed barriers (stage 1/2)
    for (bx, by) in env._stage1_placed_barriers:
        r = [bx*TILE, by*TILE, (bx+1)*TILE, (by+1)*TILE]
        draw.rectangle(r, fill=(210, 30, 30, 200))
        draw.rectangle(r, outline=(255,255,255,255), width=2)
        draw.text(((bx+0.5)*TILE-4, (by+0.5)*TILE-6), "✕", fill=(255,255,255,255))

    t = env._trailing_barrier_cell
    if t is not None:
        r = [t[0]*TILE, t[1]*TILE, (t[0]+1)*TILE, (t[1]+1)*TILE]
        draw.rectangle(r, fill=(255, 140, 0, 200))
        draw.rectangle(r, outline=(255,255,200,255), width=2)

    lines = [
        f"Stage {stage}  step {step}  trial {env.trial_count}  last: {env.last_choice}  reward: {last_reward:+.2f}",
        f"correct: {env.correct_trials}  incorrect: {env.incorrect_trials}  "
        f"acc: {env.correct_trials/max(env.trial_count,1):.0%}",
    ]
    bar_h = 14 * len(lines) + 6
    draw.rectangle([0, H-bar_h, W, H], fill=(0,0,0,210))
    for i, line in enumerate(lines):
        draw.text((4, H-bar_h+3+i*14), line, fill=(220,220,220,255))
    return np.array(img)


env = Figure8TMazeEnv(render_mode="rgb_array", max_trials_per_episode=50)
apply_stage(env, stage)
obs, _ = env.reset()
hidden = model.init_hidden(device=device)

frames = []
last_reward = 0.0
last_pos = None
stuck_count = 0

for step in range(MAX_STEPS):
    frames.append(annotate(env.render(), env, step, last_reward))
    with torch.no_grad():
        obs_t = obs_to_tensor(obs, device=device)
        logits, _, hidden = model(obs_t, hidden)
        action = Categorical(logits=logits).sample().item()

    # If stuck in the same position for 5 steps, force a random turn
    if env.agent_pos == last_pos:
        stuck_count += 1
        if stuck_count >= 5:
            action = np.random.choice([0, 1])  # random turn
            stuck_count = 0
    else:
        stuck_count = 0
    last_pos = env.agent_pos

    obs, last_reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
        hidden = model.init_hidden(device=device)
        last_pos = None
        stuck_count = 0

env.close()

pil_frames = [Image.fromarray(f) for f in frames]
pil_frames[0].save(out_path, save_all=True, append_images=pil_frames[1:], loop=0, duration=80)
print(f"Saved {out_path}  ({len(frames)} frames, {env.trial_count} trials, "
      f"{env.correct_trials}/{env.trial_count} correct)")
