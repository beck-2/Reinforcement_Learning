"""
random_agent_stage1.py — Generate a GIF of a random agent in Stage 1.

Runs until 8 well visits are recorded (or 3000 steps), then saves the GIF.
Stuck detection: force a random turn if the agent hasn't moved for 6 steps.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from config import PPOConfig
from figure8_maze_env import Figure8TMazeEnv
from ppo import STAGE_CONFIGS

cfg = PPOConfig()
sc = STAGE_CONFIGS[1]
env = Figure8TMazeEnv(
    render_mode="rgb_array",
    max_trials_per_episode=cfg.max_trials_per_episode,
    **sc,
)
obs, _ = env.reset()

frames = []
last_pos = None
stuck_count = 0
well_visits = 0
TARGET_VISITS = 8
MAX_STEPS = 3000

for _ in range(MAX_STEPS):
    pos = tuple(env.agent_pos)
    if pos == last_pos:
        stuck_count += 1
    else:
        stuck_count = 0
    last_pos = pos

    # If stuck, force a turn instead of purely random
    if stuck_count >= 6:
        action = np.random.choice([0, 1])   # turn left or right
        stuck_count = 0
    else:
        action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())

    if info["trial_count"] > well_visits:
        well_visits = info["trial_count"]
        print(f"  Visit {well_visits}: choice={info['last_choice']}  "
              f"acc={info['accuracy']:.0%}")

    if well_visits >= TARGET_VISITS or terminated or truncated:
        break

env.close()
print(f"\nTotal frames: {len(frames)}")

# Save GIF
fig, ax = plt.subplots(figsize=(5, 5))
ax.axis("off")
im = ax.imshow(frames[0])

def update(i):
    im.set_array(frames[i])
    return [im]

anim = FuncAnimation(fig, update, frames=len(frames), interval=80, blit=True)
out = "random_agent_stage1.gif"
anim.save(out, writer=PillowWriter(fps=12))
plt.close()
print(f"Saved → {out}")
