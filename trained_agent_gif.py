"""
trained_agent_gif.py — GIF of trained PPO agent in Stage 1 with barriers visible.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch

from figure8_maze_env import Figure8TMazeEnv
from model import RecurrentActorCritic
from ppo import STAGE_CONFIGS, encode_obs
from config import PPOConfig

TILE = 32

def draw_barriers(frame, placed, trailing, depleted_wells=None):
    out = frame.copy()
    # Mark visited wells with a bright green border (still passable)
    for (wx, wy) in (depleted_wells or []):
        px, py = wx * TILE, wy * TILE
        for thickness in range(4):
            out[py+thickness, px:px+TILE] = [0, 220, 80]
            out[py+TILE-1-thickness, px:px+TILE] = [0, 220, 80]
            out[py:py+TILE, px+thickness] = [0, 220, 80]
            out[py:py+TILE, px+TILE-1-thickness] = [0, 220, 80]
    for (bx, by) in placed:
        px, py = bx * TILE, by * TILE
        out[py:py+TILE, px:px+TILE] = [220, 30, 30]
        out[py:py+2, px:px+TILE] = 255
        out[py+TILE-2:py+TILE, px:px+TILE] = 255
        out[py:py+TILE, px:px+2] = 255
        out[py:py+TILE, px+TILE-2:px+TILE] = 255
    if trailing:
        bx, by = trailing
        px, py = bx * TILE, by * TILE
        out[py:py+TILE, px:px+TILE] = [230, 130, 0]
        out[py:py+2, px:px+TILE] = 255
        out[py+TILE-2:py+TILE, px:px+TILE] = 255
        out[py:py+TILE, px:px+2] = 255
        out[py:py+TILE, px+TILE-2:px+TILE] = 255
    return out

cfg = PPOConfig()
model = RecurrentActorCritic(cfg.obs_dim, cfg.hidden_size, cfg.num_actions)
ck = torch.load("checkpoint_ppo.pt", map_location="cpu")
model.load_state_dict(ck["model_state"])
model.eval()

env = Figure8TMazeEnv(
    render_mode="rgb_array",
    max_trials_per_episode=cfg.max_trials_per_episode,
    **STAGE_CONFIGS[1],
)
obs_dict, _ = env.reset()
obs = encode_obs(obs_dict)
hidden = model.init_hidden()

from constants import LEFT_WELL_LOC, RIGHT_WELL_LOC

frames = []
well_visits = 0
TARGET = 8
MAX_STEPS = 2000
last_pos = None
stuck = 0
depleted = []   # well locations that have been visited this episode

with torch.no_grad():
    for _ in range(MAX_STEPS):
        pos = tuple(env.agent_pos)
        stuck = stuck + 1 if pos == last_pos else 0
        last_pos = pos

        if stuck >= 3:
            action = int(np.random.choice([0, 1]))
            stuck = 0
        else:
            logits, _, hidden = model(obs.unsqueeze(0), hidden)
            dist = torch.distributions.Categorical(logits=logits[0])
            action = int(dist.sample())

        obs_dict, r, term, trunc, info = env.step(action)
        hidden = RecurrentActorCritic.detach_hidden(hidden)
        obs = encode_obs(obs_dict)

        # Track which well was just visited to show as depleted
        if info["trial_count"] > well_visits:
            well_visits = info["trial_count"]
            visited_loc = LEFT_WELL_LOC if info["last_choice"] == "left" else RIGHT_WELL_LOC
            if visited_loc not in depleted:
                depleted.append(visited_loc)
            # Reset depletion after both wells visited (new cycle)
            if len(depleted) >= 2:
                depleted = [visited_loc]
            print(f"  Visit {well_visits}: {info['last_choice']}  "
                  f"acc={info['accuracy']:.0%}  steps={len(frames)}")

        raw = env.render()
        frames.append(draw_barriers(raw, env._stage1_placed_barriers,
                                    env._trailing_barrier_cell,
                                    depleted)[::2, ::2])

        if well_visits >= TARGET or term or trunc:
            break

env.close()
print(f"Frames: {len(frames)}")

fig, ax = plt.subplots(figsize=(5.5, 6))
ax.axis("off")
im = ax.imshow(frames[0])
ax.set_title(f"Stage 1 trained agent ({ck['step']:,} steps)\nReaches well fast, then policy collapsed", fontsize=9, pad=4)
ax.legend(handles=[
    Patch(color=(220/255, 30/255, 30/255), label="Force barrier"),
    Patch(color=(230/255, 130/255, 0),     label="Trailing barrier"),
    Patch(color=(0, 220/255, 80/255),      label="Well visited"),
], loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8)

def update(i):
    im.set_array(frames[i])
    return [im]

anim = FuncAnimation(fig, update, frames=len(frames), interval=80, blit=True)
anim.save("trained_agent_stage1.gif", writer=PillowWriter(fps=12))
plt.close()
print("Saved → trained_agent_stage1.gif")
