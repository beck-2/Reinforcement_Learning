"""
agent_demo.py
Run the trained agent (checkpoint.pt) through Stage 3 and save a GIF.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
from figure8_maze_env import Figure8TMazeEnv
from model import RecurrentActorCritic
from train import apply_stage, obs_to_tensor
from config import Config
from constants import MAZE_SIZE

TILE = int(480 / MAZE_SIZE)
MAX_STEPS = 2000

config = Config()
device = torch.device("cpu")

model = RecurrentActorCritic(config.obs_dim, config.hidden_size, config.num_actions)
model.load_state_dict(torch.load("checkpoint.pt", map_location=device))
model.eval()


def annotate(raw_frame, env, step, last_reward):
    img = Image.fromarray(raw_frame)
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    lines = [
        f"step {step}   trial {env.trial_count}   last choice: {env.last_choice}   reward: {last_reward:+.2f}",
        f"correct: {env.correct_trials}   incorrect: {env.incorrect_trials}",
    ]
    bar_h = 14 * len(lines) + 6
    draw.rectangle([0, H - bar_h, W, H], fill=(0, 0, 0, 210))
    for i, line in enumerate(lines):
        draw.text((4, H - bar_h + 3 + i * 14), line, fill=(220, 220, 220, 255))
    return np.array(img)


env = Figure8TMazeEnv(render_mode="rgb_array", max_trials_per_episode=50)
apply_stage(env, 3)
obs, _ = env.reset()
hidden = model.init_hidden(device=device)

frames = []
last_reward = 0.0

for step in range(MAX_STEPS):
    frames.append(annotate(env.render(), env, step, last_reward))

    with torch.no_grad():
        obs_t = obs_to_tensor(obs, device=device)
        logits, _, hidden = model(obs_t, hidden)
        action = torch.argmax(logits, dim=-1).item()  # greedy

    obs, last_reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        break

env.close()

out = "agent.gif"
pil_frames = [Image.fromarray(f) for f in frames]
pil_frames[0].save(out, save_all=True, append_images=pil_frames[1:], loop=0, duration=80)
print(f"Saved {out}  ({len(frames)} frames, {step+1} steps)")
print(f"Trials: {env.trial_count}  Correct: {env.correct_trials}  Incorrect: {env.incorrect_trials}")
acc = env.correct_trials / env.trial_count if env.trial_count else 0
print(f"Accuracy: {acc:.1%}")
