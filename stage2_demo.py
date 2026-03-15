"""
stage2_demo.py
Random agent completing one loop (start → well → start) in Stage 2.
Stage 2: bottom flanking barriers only — no force-alternation barriers at top.
"""

import numpy as np
from PIL import Image, ImageDraw
from figure8_maze_env import Figure8TMazeEnv
from train import apply_stage
from constants import MAZE_SIZE

TILE = int(480 / MAZE_SIZE)
MAX_STEPS = 4000

RED_PLACED   = (210, 30, 30, 200)
RED_EDGE     = (255, 255, 255, 255)
ORANGE_TRAIL = (255, 140, 0, 220)
ORANGE_EDGE  = (255, 255, 200, 255)


def cell_rect(gx, gy):
    return [gx * TILE, gy * TILE, (gx + 1) * TILE, (gy + 1) * TILE]


def annotate(raw_frame: np.ndarray, env: Figure8TMazeEnv, step: int) -> np.ndarray:
    img = Image.fromarray(raw_frame)
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    for (bx, by) in env._stage1_placed_barriers:
        r = cell_rect(bx, by)
        draw.rectangle(r, fill=RED_PLACED)
        draw.rectangle(r, outline=RED_EDGE, width=2)
        cx, cy = (bx + 0.5) * TILE, (by + 0.5) * TILE
        draw.text((cx - 4, cy - 6), "✕", fill=(255, 255, 255, 255))

    t = env._trailing_barrier_cell
    if t is not None:
        r = cell_rect(*t)
        draw.rectangle(r, fill=ORANGE_TRAIL)
        draw.rectangle(r, outline=ORANGE_EDGE, width=2)
        cx, cy = (t[0] + 0.5) * TILE, (t[1] + 0.5) * TILE
        draw.text((cx - 4, cy - 6), "←", fill=(255, 255, 255, 255))

    phase_names = {0: "stem traversal", 1: "choice", 2: "left return", 3: "right return"}
    phase = phase_names.get(env._barrier_state, "?")
    placed_str = str(sorted(env._stage1_placed_barriers)) if env._stage1_placed_barriers else "none"
    trail_str  = str(env._trailing_barrier_cell) if env._trailing_barrier_cell else "none"

    lines = [
        f"STAGE 2   step {step}   trial {env.trial_count}   phase: {phase}   last choice: {env.last_choice}",
        f"placed barriers: {placed_str}",
        f"trailing: {trail_str}",
    ]
    bar_h = 14 * len(lines) + 6
    draw.rectangle([0, H - bar_h, W, H], fill=(0, 0, 0, 210))
    for i, line in enumerate(lines):
        draw.text((4, H - bar_h + 3 + i * 14), line, fill=(220, 220, 220, 255))

    return np.array(img)


if __name__ == "__main__":
    env = Figure8TMazeEnv(render_mode="rgb_array", max_trials_per_episode=50)
    apply_stage(env, 2)
    obs, _ = env.reset()

    frames = []
    step = 0
    loops_done = 0

    while step < MAX_STEPS:
        frames.append(annotate(env.render(), env, step))

        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        step += 1

        # One loop = agent visited a well and returned to start (_barrier_state back to 0)
        if env._barrier_state == 0 and env.trial_count > loops_done:
            loops_done = env.trial_count
            # Capture 30 extra frames to see final state
            for _ in range(30):
                frames.append(annotate(env.render(), env, step))
                obs, _, term, trunc, _ = env.step(env.action_space.sample())
                step += 1
            break

        if terminated or truncated:
            break

    env.close()

    out = "stage2_demo.gif"
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        out, save_all=True, append_images=pil_frames[1:],
        loop=0, duration=80,
    )
    print(f"Saved {out}  ({len(frames)} frames, {step} steps, {env.trial_count} trials)")
