"""
gen_stage_gifs.py — Random agent GIFs for Stage 1 and Stage 2.

Stage 1: force-alternation barriers visible (red=force, orange=trailing)
Stage 2: barriers present but no force-alternation (barriers shown, step cost active)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch

from figure8_maze_env import Figure8TMazeEnv
from ppo import STAGE_CONFIGS, apply_stage
from config import PPOConfig

TILE = 32


def draw_barriers(frame, placed, trailing):
    out = frame.copy()
    for (bx, by) in placed:
        px, py = bx * TILE, by * TILE
        out[py:py+TILE, px:px+TILE] = [220, 30, 30]
        out[py:py+2, px:px+TILE] = 255
        out[py+TILE-2:py+TILE, px:px+TILE] = 255
        out[py:py+TILE, px:px+2] = 255
        out[py:py+TILE, px+TILE-2:px+TILE] = 255
    if trailing is not None:
        bx, by = trailing
        px, py = bx * TILE, by * TILE
        out[py:py+TILE, px:px+TILE] = [230, 130, 0]
        out[py:py+2, px:px+TILE] = 255
        out[py+TILE-2:py+TILE, px:px+TILE] = 255
        out[py:py+TILE, px:px+2] = 255
        out[py:py+TILE, px+TILE-2:px+TILE] = 255
    return out


def run_stage(stage: int, target_visits: int = 6, max_steps: int = 1500, out_path: str = None):
    cfg = PPOConfig()
    env = Figure8TMazeEnv(
        render_mode="rgb_array",
        max_trials_per_episode=cfg.max_trials_per_episode,
    )
    apply_stage(env, stage)
    env.reset()

    frames = []
    last_pos = None
    stuck_count = 0
    well_visits = 0

    for _ in range(max_steps):
        pos = tuple(env.agent_pos)
        stuck_count = stuck_count + 1 if pos == last_pos else 0
        last_pos = pos

        action = np.random.choice([0, 1]) if stuck_count >= 6 else env.action_space.sample()
        if stuck_count >= 6:
            stuck_count = 0

        obs, reward, terminated, truncated, info = env.step(action)

        raw = env.render()
        placed = getattr(env, "_stage1_placed_barriers", set())
        trailing = getattr(env, "_trailing_barrier_cell", None)
        frames.append(draw_barriers(raw, placed, trailing)[::2, ::2])

        if info["trial_count"] > well_visits:
            well_visits = info["trial_count"]
            print(f"  [Stage {stage}] Visit {well_visits}: {info['last_choice']}  acc={info['accuracy']:.0%}")

        if well_visits >= target_visits or terminated or truncated:
            break

    env.close()
    print(f"  Frames: {len(frames)}")

    sc = STAGE_CONFIGS[stage]
    barriers_label = (
        "Force barriers (red) + trailing (orange)"
        if sc["force_alternation_barriers"]
        else "Barriers present, no force-alternation"
        if sc["use_stage1_barriers"]
        else "No barriers"
    )
    title = f"Stage {stage} — Random agent\n{barriers_label}"

    fig, ax = plt.subplots(figsize=(5.5, 6))
    ax.axis("off")
    im = ax.imshow(frames[0])
    ax.set_title(title, fontsize=9, pad=4)

    legend_handles = [
        Patch(color=(220/255, 30/255, 30/255), label="Force barrier"),
        Patch(color=(230/255, 130/255, 0),     label="Trailing barrier"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, framealpha=0.9)

    def update(i):
        im.set_array(frames[i])
        return [im]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=80, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=12))
    plt.close()
    print(f"  Saved → {out_path}")


print("Generating Stage 1 GIF...")
run_stage(1, target_visits=6, max_steps=1500, out_path="stage1_env_demo.gif")

print("Generating Stage 2 GIF...")
run_stage(2, target_visits=6, max_steps=2000, out_path="stage2_env_demo.gif")

print("Done.")
