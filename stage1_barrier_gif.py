"""
stage1_barrier_gif.py — Random agent in Stage 1 with barriers rendered visibly.

Barriers are overlaid as bright red rectangles on each frame.
Trailing barrier shown in orange. Runs until 8 well visits.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from figure8_maze_env import Figure8TMazeEnv
from ppo import STAGE_CONFIGS
from config import PPOConfig

TILE = 32          # pixels per grid cell (480 / 15)
MAZE_SIZE = 15

def draw_barriers(frame: np.ndarray, placed: set, trailing) -> np.ndarray:
    """Draw barrier overlays onto a copy of the frame."""
    out = frame.copy()
    for (bx, by) in placed:
        px, py = bx * TILE, by * TILE
        # Bright red fill
        out[py:py+TILE, px:px+TILE] = [220, 30, 30]
        # White border for visibility
        out[py:py+2,    px:px+TILE] = 255
        out[py+TILE-2:py+TILE, px:px+TILE] = 255
        out[py:py+TILE, px:px+2]   = 255
        out[py:py+TILE, px+TILE-2:px+TILE] = 255
    if trailing is not None:
        bx, by = trailing
        px, py = bx * TILE, by * TILE
        # Orange for trailing barrier
        out[py:py+TILE, px:px+TILE] = [230, 130, 0]
        out[py:py+2,    px:px+TILE] = 255
        out[py+TILE-2:py+TILE, px:px+TILE] = 255
        out[py:py+TILE, px:px+2]   = 255
        out[py:py+TILE, px+TILE-2:px+TILE] = 255
    return out

cfg = PPOConfig()
env = Figure8TMazeEnv(
    render_mode="rgb_array",
    max_trials_per_episode=cfg.max_trials_per_episode,
    **STAGE_CONFIGS[1],
)
env.reset()

frames = []
last_pos = None
stuck_count = 0
well_visits = 0
TARGET_VISITS = 6
MAX_STEPS = 1500

for _ in range(MAX_STEPS):
    pos = tuple(env.agent_pos)
    stuck_count = stuck_count + 1 if pos == last_pos else 0
    last_pos = pos

    action = np.random.choice([0, 1]) if stuck_count >= 6 else env.action_space.sample()
    if stuck_count >= 6:
        stuck_count = 0

    obs, reward, terminated, truncated, info = env.step(action)

    raw = env.render()
    frame = draw_barriers(raw, env._stage1_placed_barriers, env._trailing_barrier_cell)
    # Downsample 2x to keep GIF manageable (480→240)
    frames.append(frame[::2, ::2])

    if info["trial_count"] > well_visits:
        well_visits = info["trial_count"]
        print(f"  Visit {well_visits}: {info['last_choice']}  acc={info['accuracy']:.0%}")

    if well_visits >= TARGET_VISITS or terminated or truncated:
        break

env.close()
print(f"Frames: {len(frames)}")

# Build annotated GIF with legend
fig, ax = plt.subplots(figsize=(5.5, 6))
ax.axis("off")
im = ax.imshow(frames[0])

# Static legend patches
from matplotlib.patches import Patch
legend = [
    Patch(color=(220/255, 30/255, 30/255), label="Force barrier"),
    Patch(color=(230/255, 130/255, 0),     label="Trailing barrier"),
]
ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.08),
          ncol=2, fontsize=9, framealpha=0.9)
ax.set_title("Stage 1 — Random agent (barriers visible)", fontsize=10, pad=4)

def update(i):
    im.set_array(frames[i])
    return [im]

anim = FuncAnimation(fig, update, frames=len(frames), interval=80, blit=True)
out = "stage1_with_barriers.gif"
anim.save(out, writer=PillowWriter(fps=12))
plt.close()
print(f"Saved → {out}")
