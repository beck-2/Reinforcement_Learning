"""
visualize_stages.py
Generate a GIF showing the trained agent's behavior at each curriculum stage,
with reward values and the loop-gate barrier status annotated on each frame.
"""

import numpy as np
import torch
from torch.distributions import Categorical
from PIL import Image, ImageDraw, ImageFont

from figure8_maze_env import Figure8TMazeEnv
from model import RecurrentActorCritic
from config import Config
from train import STAGE_CONFIGS, apply_stage
from constants import MAZE_SIZE


# ── Obs encoding (mirrors train.py) ────────────────────────────────────────────

def obs_to_tensor(obs, device=None):
    pos = obs["position_vector"] / MAZE_SIZE
    d = int(obs["direction"])
    dir_oh = np.zeros(4, dtype=np.float32)
    dir_oh[d] = 1.0
    vec = np.concatenate([pos, dir_oh])
    t = torch.tensor(vec, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t.unsqueeze(0)


# ── Per-frame annotation ────────────────────────────────────────────────────────

GATE_LABELS = {
    0: "OPEN  (well reward active)",
    1: "LOCKED — reach arm corner",
    2: "LOCKED — reach stem base",
    3: "LOCKED — reach T-junction",
}

STAGE_COLORS = {1: (60, 180, 75), 2: (255, 165, 0), 3: (220, 50, 50)}

def annotate_frame(raw_frame: np.ndarray, env: Figure8TMazeEnv, stage_num: int) -> np.ndarray:
    """Overlay reward table + barrier status onto the rendered frame."""
    cfg = STAGE_CONFIGS[stage_num]
    img = Image.fromarray(raw_frame)
    draw = ImageDraw.Draw(img)

    W, H = img.size

    # ── Reward table (top-left) ──────────────────────────────────────────────
    stage_color = STAGE_COLORS[stage_num]
    box_x, box_y, box_w, box_h = 4, 4, 210, 104
    draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h],
                   fill=(0, 0, 0, 180), outline=stage_color, width=2)

    rows = [
        (f"── Stage {stage_num} ──",        stage_color),
        (f"  correct:    +{cfg['correct_reward']:.2f}", (100, 255, 100)),
        (f"  incorrect:  {cfg['incorrect_reward']:+.2f}", (255, 120, 120)),
        (f"  foraging:   +{cfg['foraging_reward']:.2f}", (100, 200, 255)),
        (f"  loop bonus: +{cfg['loop_bonus']:.2f}",       (200, 200, 100)),
        (f"  step cost:  {cfg['step_cost']:+.4f}",        (180, 180, 180)),
    ]

    ty = box_y + 6
    for text, color in rows:
        draw.text((box_x + 6, ty), text, fill=color)
        ty += 16

    # ── Loop-gate barrier (bottom-left) ─────────────────────────────────────
    phase = getattr(env, "_loop_phase", 0)
    gate_text = f"Gate: {GATE_LABELS[phase]}"
    gate_color = (100, 255, 100) if phase == 0 else (255, 80, 80)

    gbox_y = H - 28
    draw.rectangle([4, gbox_y, W - 4, H - 4],
                   fill=(0, 0, 0, 180), outline=gate_color, width=2)
    draw.text((10, gbox_y + 6), gate_text, fill=gate_color)

    # ── Trial counter (top-right) ─────────────────────────────────────────────
    trial_text = f"trial {env.trial_count}"
    draw.text((W - 70, 8), trial_text, fill=(220, 220, 220))

    return np.array(img)


# ── Run one stage ───────────────────────────────────────────────────────────────

def run_stage(env, model, stage_num, num_frames=90, device="cpu"):
    apply_stage(env, stage_num)
    obs, _ = env.reset()
    hidden = model.init_hidden(device=torch.device(device))

    frames = []

    # Title card (15 frames ~1.5s at 10 fps)
    title_img = Image.new("RGB", (env.render().shape[1], env.render().shape[0]), (15, 15, 15))
    draw = ImageDraw.Draw(title_img)
    cfg = STAGE_CONFIGS[stage_num]
    color = STAGE_COLORS[stage_num]
    draw.text((title_img.width // 2 - 60, title_img.height // 2 - 20),
              f"Stage {stage_num}", fill=color)
    stage_desc = {1: "Foraging / Navigation Bootstrap",
                  2: "Guided Alternation",
                  3: "Pure Alternation (Paper Spec)"}
    draw.text((title_img.width // 2 - 100, title_img.height // 2 + 10),
              stage_desc[stage_num], fill=(180, 180, 180))
    title_arr = np.array(title_img)
    for _ in range(15):
        frames.append(title_arr)

    for _ in range(num_frames):
        raw = env.render()
        frames.append(annotate_frame(raw, env, stage_num))

        obs_t = obs_to_tensor(obs, device=torch.device(device))
        with torch.no_grad():
            logits, _, hidden = model(obs_t, hidden)
        dist = Categorical(logits=logits)
        action = dist.sample()

        obs, _, terminated, truncated, _ = env.step(action.item())
        if terminated or truncated:
            obs, _ = env.reset()
            hidden = model.init_hidden(device=torch.device(device))

    return frames


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    device = "cpu"
    cfg = Config()

    env = Figure8TMazeEnv(render_mode="rgb_array", max_trials_per_episode=50)

    model = RecurrentActorCritic(
        obs_dim=cfg.obs_dim,
        hidden_size=cfg.hidden_size,
        num_actions=cfg.num_actions,
    )

    if os.path.exists(cfg.checkpoint_path):
        model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device))
        print(f"Loaded checkpoint: {cfg.checkpoint_path}")
    else:
        print("No checkpoint found — using untrained (random) model")

    model.eval()

    all_frames = []
    for stage in [1, 2, 3]:
        print(f"Rendering Stage {stage}...")
        all_frames += run_stage(env, model, stage, num_frames=120)

    env.close()

    # Save GIF
    out_path = "stages_behavior.gif"
    pil_frames = [Image.fromarray(f) for f in all_frames]
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=100,   # 10 fps
    )
    print(f"Saved: {out_path}  ({len(all_frames)} frames)")
