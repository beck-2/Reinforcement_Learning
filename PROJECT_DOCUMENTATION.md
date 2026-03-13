#@title PROJECT_DOCUMENTATION.md - Complete environment specification for poster
  %%writefile PROJECT_DOCUMENTATION.md

  # Figure-8 T-Maze Spatial Alternation Environment
  ## Reinforcement Learning Implementation of Wood et al. (2000)

  ---

  ## 1.2.1 TITLE & AUTHORS INFORMATION

  **Project Title:**
  "Hippocampal-Inspired Reinforcement Learning in a Spatial Alternation Task"

  **Environment Name:**
  Figure8TMaze-v0

  **Environment Type:**
  Modified T-Maze with Figure-8 Return Path (Continuous Spatial Alternation)

  **Agent Type(s):**
  [YOUR TEAMMATES FILL IN - e.g., Q-Learning, SARSA, Successor Representation, Actor-Critic, PPO, etc.]

  **Group Members:**
  - Riley Yupitun - Environment Engineer
  - [Name 2] - Agent Developer
  - [Name 3] - Analysis Lead
  - [Name 4] - Poster Coordinator

  ---

  ## 1.2.2 ENVIRONMENT, TASK, & AGENT DESCRIPTION

  ### **Environment Dynamics**

  #### **State Space (Markov State)**
  The environment state consists of:

  1. **Spatial State:**
     - Agent position: (x, y) ∈ {0...14} × {0...14}
     - Agent direction: d ∈ {0, 1, 2, 3} (East, South, West, North)
     - Full state dimensionality: 15 × 15 × 4 = 900 possible poses

  2. **Working Memory State:**
     - Last choice: {None, 'left', 'right'}
     - Trial number: integer [0, max_trials]

  3. **Complete Markov State:**
     - S = (x, y, d, last_choice, trial_number)
     - This satisfies the Markov property: P(s_{t+1} | s_t, a_t) is independent of history

  #### **Action Space**
  - A = {0, 1, 2}
    - 0: Turn left (rotate counter-clockwise)
    - 1: Turn right (rotate clockwise)
    - 2: Move forward (if valid)

  - Actions are deterministic (no stochasticity in transitions)

  #### **Transition Dynamics**
  - **Turn actions (0, 1):** Change orientation only
    - T(s' | s, turn) = 1 if s' has same (x,y) but rotated direction

  - **Forward action (2):** Attempt to move to cell in front
    - T(s' | s, forward) = 1 if forward cell is valid (floor/well)
    - T(s' | s, forward) = 1 if forward cell is wall (stay in place)

  - **Special transition:** Upon reaching a reward well, agent teleports to start position
    - This creates the "continuous" task structure

  ### **Reward Structure**

  #### **Primary Rewards (Task Objective)**
  - **Correct Alternation:** +1.0
    - Agent reaches LEFT well after previously going RIGHT (or vice versa)
    - Agent reaches either well on first trial (no prior choice)

  - **Incorrect Alternation:** 0.0
    - Agent reaches SAME well as previous trial

  #### **Secondary Costs (Efficiency Incentives)**
  - **Step cost:** -0.01 per timestep
    - Encourages finding shortest path
    - Prevents infinite wandering

  - **Turn cost:** -0.01 per turn action
    - Discourages spinning in place
    - Promotes efficient orientation

  #### **Cumulative Reward Function**
  R(s, a, s') = R_alternation + R_step + R_turn

  where:
    R_alternation = +1.0 if correct choice, 0.0 otherwise
    R_step = -0.01 always
    R_turn = -0.01 if a ∈ {0,1}, 0.0 if a = 2

  ### **Task Description**

  **Objective:**
  Train an agent to maximize cumulative reward by learning to alternate between left and right choice arms
  on a T-maze.

  **Task Requirements:**
  1. Navigate from start position (base of stem) to T-junction
  2. Choose left or right arm
  3. Remember previous choice
  4. On next trial, choose opposite arm
  5. Repeat for 30-50 trials per episode

  **Cognitive Demands:**
  - **Spatial navigation:** Learn maze layout
  - **Working memory:** Remember last choice
  - **Sequential decision-making:** Plan path to correct goal
  - **Episodic memory:** Distinguish left-turn vs right-turn episodes

  **Neuroscience Connection:**
  This task replicates Wood et al. (2000) where rats showed hippocampal neurons that fired differentially
  on left-turn vs right-turn trials, even when the rat was in the same physical location on the central
  stem.

  ### **Task Classification: Episodic or Continuing?**

  **Answer: HYBRID (Continuing task with episodic structure)**

  - **Within-episode:** CONTINUING
    - Agent performs multiple trials (30-50) without episode termination
    - Trials are marked by reaching a well and resetting to start
    - Alternation history persists across trials within an episode
    - No discounting to zero between trials

  - **Between-episodes:** EPISODIC
    - Episodes terminate after max_trials (default: 50)
    - Alternation history resets between episodes
    - Performance is evaluated per episode

  **Practical Implementation:**
  - Episode = One "session" of 30-50 trials (matches experimental paper)
  - Each trial = One traversal from start → choice → return to start
  - Total steps per episode ≈ 50 trials × 40 steps/trial = 2000 steps

  ### **Agent Type**

  [YOUR AGENT TEAM FILLS THIS IN]

  **Options your team might choose:**
  - **Tabular Q-Learning:** Direct state-action value learning
  - **SARSA:** On-policy TD learning
  - **Deep Q-Network (DQN):** Neural network Q-function
  - **Actor-Critic:** Separate policy and value networks
  - **Successor Representation (SR):** Hippocampal-inspired predictive representation
  - **Proximal Policy Optimization (PPO):** Modern policy gradient method

  **Recommended for neuroscience connection:**
  - **Successor Representation** - Directly models hippocampal predictive coding
  - **Actor-Critic with LSTM** - Memory component for alternation

  ---

  ## 2.1 ENVIRONMENT TEAM VERIFICATION CHECKLIST

  ### ✅ **Markov State Space Design**

  **State Components:**
  - [x] Position (x, y)
  - [x] Direction (0-3)
  - [x] Last choice (None/left/right)
  - [x] Trial number

  **State Space Properties:**
  - [x] Fully observable (agent can see entire maze)
  - [x] Markov property satisfied
  - [x] Includes working memory for task requirement

  **Observation Format:**
  ```python
  obs = {
      'image': np.array([15, 15, 3]),      # RGB image
      'position_vector': np.array([x, y]),  # Explicit position
      'direction': int,                     # 0-3
      'last_choice': int,                   # 0=none, 1=left, 2=right
      'trial_number': int                   # Current trial count
  }

  ✅ Transition Dynamics

  Deterministic Transitions:
  - Turn left: d' = (d - 1) mod 4
  - Turn right: d' = (d + 1) mod 4
  - Move forward: (x', y') = (x, y) + direction_vector if valid

  Special Transitions:
  - Reaching well → teleport to start (trial completion)
  - Hitting wall → stay in place (no-op)

  Transition Function Verified:
  P(s'|s,a) = 1.0 for deterministic outcome
  P(s'|s,a) = 0.0 for all other s'

  ✅ Reward Logic Implementation

  Reward Calculation:
  def calculate_reward(last_choice, current_choice):
      if last_choice is None:
          return +1.0  # First trial
      elif current_choice != last_choice:
          return +1.0  # Correct alternation
      else:
          return 0.0   # Incorrect (same as last)

  Additional Costs:
  - Step cost: -0.01 per step
  - Turn cost: -0.01 per turn
  - Cumulative reward tracking

  ✅ Reset Logic

  Episode Reset:
  - Agent position → START_POS
  - Agent direction → START_DIR (facing north)
  - Trial count → 0
  - Last choice → None
  - Trajectory → empty list

  Within-Episode Reset (Trial Boundary):
  - Agent position → START_POS
  - Agent direction → START_DIR
  - Trial count → increment
  - Last choice → updated to current choice
  - Trajectory → append to history

  ✅ Termination Logic

  Episode Terminates When:
  - Trial count reaches max_trials (50)
  - Step count exceeds max_steps (2000)

  Episode Does NOT Terminate When:
  - Agent makes incorrect choice (continues running)
  - Agent completes a single trial (continues to next trial)

  Truncation vs Termination:
  - terminated = True when max_trials reached (task complete)
  - truncated = True when timeout occurs (max_steps exceeded)

  ✅ Environment Behavior Verification

  Test Cases Implemented:
  1. Random agent can navigate maze
  2. Rewards given correctly for alternation
  3. No reward given for repeated choice
  4. Trial counter increments properly
  5. Episode terminates at correct trial count
  6. Reset clears history appropriately

  Verification Scripts:
  - test_and_visualize.py - Random agent testing
  - verify_environment.py - Systematic verification (see below)

  ---
  ENVIRONMENT DIAGRAM

  Figure-8 T-Maze Layout (Top-down view)

           WELL_L ←─────┐   ┐─────→ WELL_R
          (Water)        │   │        (Water)
                         │   │
           ┌─────────────┴───┴─────────────┐
           │   LEFT ARM    T    RIGHT ARM   │  ← CHOICE POINT
           └───────────────┬─────────────── ┘     (Sector 4)
                           │
                           │  ← Sector 3
                           │
                           │  ← Sector 2
                           │
                           │  ← Sector 1
                           S  ← START
                           │
           ┌───────────────┴─────────────── ┐
           │   RETURN      │      RETURN    │
           │   PATH        │      PATH      │
           └───────────────┴─────────────── ┘

  Legend:
  - S = Start position (agent spawns here)
  - T = T-junction / choice point
  - WELL_L/WELL_R = Water reward locations
  - Sectors 1-4 = Central stem divisions (for neural analysis)
  - Return paths = Complete the figure-8 pattern

  Maze Coordinates:
  - Grid size: 15×15
  - Start: (7, 12)
  - Left well: (4, 4)
  - Right well: (10, 4)
  - Stem: Column 7, rows 4-12

  ---
  STATE REPRESENTATION FOR AGENT TEAM

  Input Options

  Option 1: Image-based (CNN)
  obs['image']  # Shape: (15, 15, 3)
  # Use CNN to process visual input

  Option 2: Vector-based (MLP)
  state_vector = [
      obs['position_vector'][0],  # x
      obs['position_vector'][1],  # y
      obs['direction'],           # direction
      obs['last_choice']          # working memory
  ]
  # Shape: (4,)

  Option 3: Hybrid (for Successor Representation)
  # Position encoding
  onehot_position = one_hot(x * 15 + y, 225)  # 15×15 grid
  onehot_direction = one_hot(direction, 4)
  onehot_memory = one_hot(last_choice, 3)

  state = concat([onehot_position, onehot_direction, onehot_memory])
  # Shape: (232,)

  ---
  EXPECTED PERFORMANCE METRICS

  Random Agent:
  - Accuracy: ~50% (chance level)
  - Trials to first reward: ~20 steps
  - Episode reward: ~25 (50 trials × 0.5 accuracy)

  Trained Agent (Expected):
  - Accuracy: >95%
  - Trials to first reward: ~10 steps (optimal path)
  - Episode reward: >45 (50 trials × 0.9+ accuracy)

  ---
  FILES INCLUDED

  1. constants.py - Environment parameters
  2. figure8_maze_env.py - Main environment class
  3. test_and_visualize.py - Testing and GIF generation
  4. verify_environment.py - Systematic verification (below)
  5. PROJECT_DOCUMENTATION.md - This file

  ---
  END OF DOCUMENTATION