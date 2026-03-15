"""
stage_diagrams.py
Three static side-by-side diagrams showing reward locations and barriers
for each curriculum stage.

Legend:
  Yellow circle  = active reward location (fires this stage)
  Blue circle    = reward location exists but inactive this stage
  Red diamond    = loop-gate barrier checkpoint
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from figure8_maze_env import Figure8TMazeEnv
from train import STAGE_CONFIGS
from constants import (
    STEM_X, STEM_TOP, STEM_BOTTOM,
    LEFT_RETURN_X, RIGHT_RETURN_X,
    LEFT_WELL_LOC, RIGHT_WELL_LOC,
    MAZE_SIZE,
)

TILE = 480 / MAZE_SIZE   # pixels per grid cell (= 32.0)


def cell_center(gx, gy):
    """Grid cell → pixel center (x, y) in image coords."""
    return (gx + 0.5) * TILE, (gy + 0.5) * TILE


# ── Reward locations and barrier checkpoints ────────────────────────────────

WELL_LOCATIONS = [
    (LEFT_WELL_LOC,  "left well"),
    (RIGHT_WELL_LOC, "right well"),
]

T_JUNCTION = (STEM_X, STEM_TOP)

# Barriers per stage — list of ((gx,gy), label)
BARRIERS_BY_STAGE = {
    1: [
        ((STEM_X - 1, STEM_BOTTOM),        "initial"),   # (6,12)
        ((STEM_X + 1, STEM_BOTTOM),        "initial"),   # (8,12)
        ((STEM_X,     STEM_TOP + 1),       "@ T-jxn"),   # (7,5)
        ((LEFT_RETURN_X + 1, STEM_TOP),    "after L"),   # (5,4)
        ((RIGHT_RETURN_X - 1, STEM_TOP),   "after R"),   # (9,4)
        ((STEM_X - 1, STEM_TOP),           "force R"),   # (6,4)
        ((STEM_X + 1, STEM_TOP),           "force L"),   # (8,4)
    ],
    2: [
        ((LEFT_RETURN_X,  STEM_BOTTOM), "arm corner L"),
        ((RIGHT_RETURN_X, STEM_BOTTOM), "arm corner R"),
        ((STEM_X,         STEM_BOTTOM), "stem base"),
        (T_JUNCTION,                    "T-junction"),
    ],
    3: [
        ((LEFT_RETURN_X,  STEM_BOTTOM), "arm corner L"),
        ((RIGHT_RETURN_X, STEM_BOTTOM), "arm corner R"),
        ((STEM_X,         STEM_BOTTOM), "stem base"),
        (T_JUNCTION,                    "T-junction"),
    ],
}


def draw_stage(ax, env, stage_num):
    cfg = STAGE_CONFIGS[stage_num]

    # ── Maze background ──────────────────────────────────────────────────────
    env.reset()
    frame = env.render()
    ax.imshow(frame, origin="upper")
    ax.set_axis_off()

    stage_titles = {
        1: "Stage 1 — Foraging Bootstrap",
        2: "Stage 2 — Guided Alternation",
        3: "Stage 3 — Pure Alternation",
    }
    ax.set_title(stage_titles[stage_num], fontsize=11, fontweight="bold", pad=6)

    r = TILE * 0.38   # circle radius in pixels

    # ── Wells ────────────────────────────────────────────────────────────────
    for (gx, gy), name in WELL_LOCATIONS:
        px, py = cell_center(gx, gy)

        # Correct alternation reward is always active (+1.0 in every stage)
        # Foraging bonus adds on top in stages 1/2
        foraging = cfg["foraging_reward"]
        is_foraging_active = foraging > 0

        # Main circle — yellow (always active for alternation)
        circle = plt.Circle((px, py), r, color="gold", alpha=0.85, zorder=3)
        ax.add_patch(circle)
        ax.add_patch(plt.Circle((px, py), r, color="none",
                                edgecolor="darkorange", linewidth=1.5, zorder=4))

        # Reward labels
        label_lines = [f"+{cfg['correct_reward']:.1f} alt"]
        if is_foraging_active:
            label_lines.append(f"+{foraging:.2f} any")
        ax.text(px, py, "\n".join(label_lines),
                ha="center", va="center", fontsize=5.5,
                fontweight="bold", color="black", zorder=5)

    # ── Loop-bonus marker at T-junction (yellow if active, blue if not) ──────
    loop_bonus = cfg["loop_bonus"]
    tjx, tjy = cell_center(*T_JUNCTION)

    if loop_bonus > 0:
        bonus_color, text_color, edge_color = "gold", "black", "darkorange"
        bonus_label = f"+{loop_bonus:.2f}\nloop"
    else:
        bonus_color, text_color, edge_color = "steelblue", "white", "navy"
        bonus_label = "loop\nn/a"

    bonus_circle = plt.Circle((tjx, tjy), r * 0.75,
                               color=bonus_color, alpha=0.75, zorder=3)
    ax.add_patch(bonus_circle)
    ax.add_patch(plt.Circle((tjx, tjy), r * 0.75, color="none",
                             edgecolor=edge_color, linewidth=1.5, zorder=4))
    ax.text(tjx, tjy, bonus_label,
            ha="center", va="center", fontsize=5, color=text_color,
            fontweight="bold", zorder=5)

    # ── Barrier checkpoints (red diamonds) ──────────────────────────────────
    for (gx, gy), label in BARRIERS_BY_STAGE[stage_num]:
        px, py = cell_center(gx, gy)
        ax.plot(px, py, marker="D", markersize=18,
                color="red", markeredgecolor="white",
                markeredgewidth=1.5, zorder=6, alpha=0.9)
        ax.text(px, py, "✕", ha="center", va="center",
                fontsize=7, color="white", fontweight="bold", zorder=7)
        ax.text(px, py + r * 1.25, label,
                ha="center", va="bottom", fontsize=5,
                color="red", fontweight="bold", zorder=7)

    # ── Reward table (bottom annotation) ─────────────────────────────────────
    table_lines = [
        f"correct: +{cfg['correct_reward']:.1f}",
        f"incorrect: {cfg['incorrect_reward']:+.1f}",
        f"foraging: +{cfg['foraging_reward']:.2f}",
        f"loop bonus: +{cfg['loop_bonus']:.2f}",
        f"step cost: {cfg['step_cost']:+.4f}",
    ]
    ax.text(0.5, -0.02, "  |  ".join(table_lines),
            transform=ax.transAxes, ha="center", va="top",
            fontsize=6, color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#222", alpha=0.85))


# ── Main ─────────────────────────────────────────────────────────────────────

env = Figure8TMazeEnv(render_mode="rgb_array")

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.patch.set_facecolor("#111")

for i, ax in enumerate(axes, start=1):
    draw_stage(ax, env, i)

env.close()

# Legend
legend_elements = [
    mpatches.Patch(facecolor="gold", edgecolor="darkorange", label="Active reward"),
    mpatches.Patch(facecolor="steelblue", edgecolor="navy", label="Inactive reward"),
    mpatches.Patch(facecolor="red", edgecolor="darkred", alpha=0.6, label="Loop-gate barrier"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           fontsize=9, framealpha=0.8,
           bbox_to_anchor=(0.5, 0.0))

plt.suptitle("Curriculum Reward Shaping — Stage Overview",
             fontsize=13, fontweight="bold", color="white", y=1.01)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("stage_diagrams.png", dpi=150, bbox_inches="tight",
            facecolor="#111")
plt.close()
print("Saved: stage_diagrams.png")
