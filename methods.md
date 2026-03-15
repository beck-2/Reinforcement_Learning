# Reward-Shaping Curriculum for Figure-8 T-Maze Training

Extracted from provided specification images.

---

## Stage 1 — Foraging / Navigation Bootstrap

**Goal:** Teach the agent to move through the maze and reach the reward wells.

**Active during:** Steps 0 – 500,000

**Rewards:**
- Reach any well (left or right): **+0.5**
- Complete a full loop back to start (after visiting a well): **+0.2**
- Correct alternation: **+1.0** (still applied if alternation is correct, but not required)
- Incorrect alternation: **0.0**
- step_cost: **0.0** (no step penalty — exploration encouraged)
- turn_cost: **0.0**

**Purpose:** The agent has no prior knowledge of the maze. Dense well-visit rewards give it a positive signal just for finding the wells, regardless of whether it alternates. Loop-completion bonus trains the full figure-8 circuit behavior before introducing the alternation requirement.

### Stage 1 Dynamic Barrier Sequence

Stage 1 uses physical barriers (impassable cells) that appear and disappear based on the agent's progress, guiding it through the correct figure-8 circuit.

**Phase 0 — Stem traversal (initial state):**
- Barriers at **(6,12)** and **(8,12)** flank the start position (7,12)
- Agent can only move north up the central stem

**Phase 1 — Choice phase (triggered when agent reaches T-junction (7,4)):**
- Barrier added at **(7,5)** — blocks backtracking down the stem
- Both wells are accessible; both rewards are active

**Phase 2a — Left return (triggered when agent receives left reward):**
- Barrier at (7,5) removed
- Barrier added at **(5,4)** — blocks path back to T-junction, forces agent south
- Barrier at (6,12) **lifted** — opens the left return path to start
- Left well deactivated until circuit is complete

**Phase 2b — Right return (triggered when agent receives right reward):**
- Barrier at (7,5) removed
- Barrier added at **(9,4)** — blocks path back to T-junction, forces agent south
- Barrier at (8,12) **lifted** — opens the right return path to start
- Right well deactivated until circuit is complete

**Phase 3 — Reset to stem traversal (triggered when agent returns to start (7,12)):**
- Previously lifted bottom barrier is re-dropped: (6,12) if came from left, (8,12) if came from right

**Phase 1 (second visit to T-junction) — Forced alternation:**
- Old arm barrier ((5,4) or (9,4)) cleared
- If last choice was **left**: barrier added at **(6,4)** — blocks left arm, forces right
- If last choice was **right**: barrier added at **(8,4)** — blocks right arm, forces left
- Barrier (7,5) re-added behind agent

This sequence repeats, enforcing strict figure-8 alternation throughout Stage 1.

---

## Stage 2 — Guided Alternation

**Goal:** Introduce the alternation requirement while keeping a weaker navigation scaffold.

**Active during:** Steps 500,000 – 1,500,000

**Rewards:**
- Correct alternation: **+1.0**
- Incorrect alternation: **0.0**
- Reach any well (regardless of alternation correctness): **+0.1** (reduced scaffold)
- Complete a full loop: **0.0** (loop bonus removed)
- step_cost: **-0.001**
- turn_cost: **0.0**

**Purpose:** The agent already knows how to navigate. Now it must learn the alternation rule. The small negative for incorrect choices discourages side-perseveration. The residual well-visit bonus prevents the agent from "going dark" (refusing to visit wells at all to avoid the -0.5).

---

## Stage 3 — Pure Alternation (Target Behavior)

**Goal:** Match the experimental reward structure from Wood et al. (2000) exactly.

**Active during:** Steps 1,500,000 – end of training

**Rewards:**
- Correct alternation: **+1.0**
- Incorrect alternation: **0.0** (no reward, no penalty — matches the paper)
- Any well visit bonus: **0.0** (removed)
- Loop bonus: **0.0** (removed)
- step_cost: **-0.001**
- turn_cost: **0.0**

**Purpose:** Remove all shaping rewards. The agent must now rely purely on the sparse alternation signal and its RNN hidden state to track which arm it visited last.

---

## Stage Transition Logic

Transitions are **step-count based**, not performance-gated. This avoids getting stuck if the agent is learning slowly.

| Stage | Entry condition |
|-------|----------------|
| 1 → 2 | total_steps >= 500,000 |
| 2 → 3 | total_steps >= 1,500,000 |

Total training budget: **2,000,000 steps** (unchanged from prior config).

---

## Removed Reward Components

The following reward components that existed in the prior codebase are **removed entirely**:

- `step_cost` in Stages 1–2 is 0; only small (-0.001) in Stage 3
- `turn_cost` is 0 throughout all stages
- No explicit penalty for staying still or spinning (entropy bonus in A2C handles this)
