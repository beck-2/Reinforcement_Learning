# figure8_maze_env.py
# =============================================================================
# FIGURE-8 T-MAZE ENVIRONMENT - MAIN IMPLEMENTATION
# Replicates Wood et al. (2000) experimental paradigm as RL environment
# =============================================================================
#
# PAPER SUMMARY:
# Wood et al. (2000) recorded from rat hippocampal CA1 neurons while rats
# performed continuous spatial alternation. Key findings:
# - 2/3 of hippocampal cells fired DIFFERENTLY on left-turn vs right-turn trials
# - This differential firing occurred even when the rat was in the SAME LOCATION
#   (on the central stem common to both trial types)
# - Differences could not be explained by speed, heading, or lateral position
# - This suggests hippocampal neurons encode "episodic" information beyond space
#
# OUR GOAL:
# Train an RL agent on this task and analyze whether its internal representations
# (value functions, neural network activations) show similar trial-type selectivity
# =============================================================================

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Minigrid imports - we're using this framework for grid-based RL environments
from minigrid.core.constants import COLORS, COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall, Ball, Floor, Goal
from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.rendering import fill_coords, point_in_rect

# Import our constants defined above
from constants import *

# =============================================================================
# CUSTOM COLORS (for better visualization)
# =============================================================================
# We add custom colors to distinguish different maze components

# Water reward wells - bright blue to stand out
COLORS["water_blue"] = np.array([0, 191, 255])  # RGB for cyan/water color

# Maze walls - gray to be neutral
COLORS["gray"] = np.array([128, 128, 128])      # Standard gray

# Floor tiles - dark gray for contrast
COLORS["dark_floor"] = np.array([64, 64, 64])   # Dark gray floor

# Register these colors in MiniGrid's color system
COLOR_TO_IDX["water_blue"] = 11  # Index 11 in color lookup table
COLOR_TO_IDX["gray"] = 12        # Index 12
COLOR_TO_IDX["dark_floor"] = 13  # Index 13


# =============================================================================
# CUSTOM OBJECTS (extending MiniGrid base classes)
# =============================================================================

class WaterWell(Ball):
    """
    Represents a water reward well at the end of a choice arm.
    
    Paper Connection: "Water rewards... were provided at water ports (small 
    circles) on the end of each choice arm"
    
    Inherits from Ball (a MiniGrid object type) and customizes rendering
    and interaction behavior.
    """

    def __init__(self, color='water_blue', visible=False):
        """
        Initialize a water well.
        
        Args:
            color: Visual color (default: water_blue)
            visible: Whether to show in rendering (default: False for minimal display)
        """
        super().__init__(color)  # Initialize parent Ball class
        self.visible = visible    # Control visibility in rendering
        self.has_reward = True    # Track whether reward is available (always True for us)

    def can_overlap(self):
        """
        Allow agent to move onto this tile.
        
        Returns: True - agent can occupy same cell as well
        
        Paper Connection: Rats physically reach the well location to get reward
        """
        return True

    def render(self, img):
        """
        Custom rendering: only draw if visible and has reward.
        
        Args:
            img: Image array to draw on
            
        This creates a small colored square representing the water well.
        """
        if self.visible and self.has_reward:
            # Draw filled rectangle from 0.2 to 0.8 of cell (centered, smaller than full cell)
            fill_coords(img, point_in_rect(0.2, 0.8, 0.2, 0.8), COLORS[self.color])


class MazeWall(Wall):
    """
    Wall object that is see-through for visualization.
    
    Paper Connection: The paper used a maze with visible spatial layout.
    Transparent walls allow the agent (and us) to see the full maze structure,
    similar to how rats can perceive the entire environment.
    """

    def __init__(self, color='gray'):
        """Initialize wall with gray color."""
        super().__init__(color)

    def see_behind(self):
        """
        Override to make walls transparent.
        
        Returns: True - can see through this wall
        
        This enables full maze visibility (allocentric representation)
        """
        return True


class StemFloor(Floor):
    """
    Special floor tile for the central stem that tracks its sector number.
    
    Paper Connection: Critical for analysis! The paper divided the stem into
    4 sectors and analyzed neural firing separately in each sector. By tracking
    which sector each floor tile belongs to, we can later analyze our agent's
    representations (Q-values, activations) by sector, just like the paper.
    
    Example use: "Does the agent's Q(s,a) for 'turn left' differ between
    left-turn and right-turn trials when in sector 3?"
    """

    def __init__(self, sector=0):
        """
        Initialize stem floor tile.
        
        Args:
            sector: Which sector this tile belongs to (0=not on stem, 1-4=stem sectors)
        """
        super().__init__()
        self.sector = sector  # Store sector ID for later analysis
        self.color = 'dark_floor'  # Visual appearance


# =============================================================================
# REWARDED POSES HELPER FUNCTION
# =============================================================================

def rewarded_poses_for_target(tx: int, ty: int):
    """
    Calculate which agent poses count as "reaching" a target location.
    
    Paper Connection: The paper measured when rats "reached" a reward well.
    In our discrete grid, we define "reaching" as being ADJACENT TO and
    FACING the target. This is more realistic than just occupying the same cell.
    
    Args:
        tx: Target X coordinate
        ty: Target Y coordinate
        
    Returns:
        Set of (x, y, direction) tuples representing valid "reached" poses
        
    MiniGrid Direction Convention:
        0 = East (right)
        1 = South (down)
        2 = West (left)
        3 = North (up)
        
    Example: If target is at (5, 5), rewarded poses are:
        - (5, 4, 1) = North of target, facing South toward it
        - (5, 6, 3) = South of target, facing North toward it
        - (4, 5, 0) = West of target, facing East toward it
        - (6, 5, 2) = East of target, facing West toward it
    """
    return {
        (tx, ty - 1, 1),  # Approach from North, face South (toward target)
        (tx, ty + 1, 3),  # Approach from South, face North (toward target)
        (tx - 1, ty, 0),  # Approach from West, face East (toward target)
        (tx + 1, ty, 2),  # Approach from East, face West (toward target)
    }


# =============================================================================
# MAIN ENVIRONMENT CLASS
# =============================================================================

class Figure8TMazeEnv(MiniGridEnv):
    """
    Continuous spatial alternation task environment.
    
    TASK DESCRIPTION (from paper):
    "Rats performed a continuous alternation task in which they traversed the
    central stem of the apparatus on each trial and then alternated between
    left and right turns at the T junction."
    
    EPISODIC STRUCTURE:
    - Episode = One recording session (30-50 trials in paper)
    - Trial = One traversal: start → stem → choice → return to start
    - Agent must remember last choice to alternate correctly
    
    OBSERVATIONS:
    - Visual: Full maze view (RGB image)
    - Position: (x, y) coordinates
    - Direction: Facing direction (0-3)
    - Working Memory: Last choice made (critical for task!)
    - Trial number: Current trial count
    
    ACTIONS:
    - 0: Turn left (counter-clockwise rotation)
    - 1: Turn right (clockwise rotation)
    - 2: Move forward (if not blocked by wall)
    
    REWARDS:
    - +1.0: Correct alternation (chose opposite arm from last trial)
    - 0.0: Incorrect alternation (repeated same arm)
    - -0.01: Per step (encourages efficiency)
    - -0.01: Per turn (discourages spinning)
    """

    # Metadata for Gymnasium registration
    metadata = {
        "render_modes": ["human", "rgb_array"],  # Supported rendering modes
        "render_fps": RENDER_FPS                  # Animation speed
    }

    def __init__(
        self,
        size=MAZE_SIZE,              # Grid size (15x15)
        max_steps: int | None = None, # Max timesteps per episode
        **kwargs                      # Additional arguments
    ):
        """
        Initialize the Figure-8 T-Maze environment.
        
        Args:
            size: Grid dimension (default: 15 from constants)
            max_steps: Maximum timesteps before episode truncation
            **kwargs: Additional parameters (reward values, trial limits, etc.)
        """

        # --- Extract Task Parameters from kwargs ---
        # These allow teammates to customize the environment behavior

        # How many trials per episode (default: 50, matching paper's upper bound)
        self.max_trials_per_episode = kwargs.pop("max_trials_per_episode", MAX_TRIALS_PER_EPISODE)

        # --- Reward Configuration ---
        # Teammates can modify these to experiment with different reward structures
        self.correct_reward = kwargs.pop("correct_reward", CORRECT_ALTERNATION_REWARD)
        self.incorrect_reward = kwargs.pop("incorrect_reward", INCORRECT_ALTERNATION_REWARD)
        self.step_cost = kwargs.pop("step_cost", STEP_COST)
        self.turn_cost = kwargs.pop("turn_cost", TURN_COST)
        # Curriculum shaping rewards (0 by default = no shaping)
        self.foraging_reward = kwargs.pop("foraging_reward", 0.0)  # bonus for any well visit
        self.loop_bonus = kwargs.pop("loop_bonus", 0.0)            # bonus for completing full circuit
        self.wall_bump_penalty = kwargs.pop("wall_bump_penalty", 0.0)  # penalty for hitting walls/barriers

        # Stage 1 dynamic barrier system
        # When True, physical barriers guide the agent through the figure-8 circuit
        self.use_stage1_barriers = kwargs.pop("use_stage1_barriers", False)
        # When True, also place force-alternation barriers at (6,4)/(8,4) after each well visit
        self.force_alternation_barriers = kwargs.pop("force_alternation_barriers", True)
        # Placed barriers: event-triggered, temporary (none are permanent)
        self._stage1_placed_barriers: set = set()
        # Trailing barrier: always the most recently vacated cell (prevents backtracking)
        self._trailing_barrier_cell: tuple = None
        self._barrier_state: int = 0  # 0=stem, 1=choice, 2=left_return, 3=right_return

        # --- Start Position Configuration ---
        self.agent_start_pos = START_POS  # (7, 12) - base of stem
        self.agent_start_dir = START_DIR  # 3 = facing North

        # --- Trial Tracking Variables ---
        # CRITICAL: These maintain the "alternation history" across trials
        # Paper Connection: The task requires remembering the previous choice

        self.trial_count = 0        # How many trials completed this episode
        self.last_choice = None     # 'left', 'right', or None (first trial)
                                    # This is the WORKING MEMORY component!

        self.correct_trials = 0     # Number of correct alternations
        self.incorrect_trials = 0   # Number of incorrect (repeated) choices

        self.trial_history = []     # List storing all trial outcomes
                                    # Each entry: {'trial': int, 'choice': str,
                                    #              'correct': bool, 'reward': float,
                                    #              'trajectory': list of poses}

        # --- Trajectory Tracking ---
        # For analysis and visualization - stores agent's path through maze
        self.trajectory = []              # All poses across entire episode
        self.current_trial_trajectory = [] # Poses for current trial only

        # --- Episode Tracking ---
        self.episode_count = 0  # Total episodes run (increments on each reset)

        # --- Initialize Parent MiniGridEnv Class ---

        # Define mission (task description shown to agent)
        mission_space = MissionSpace(mission_func=self._gen_mission)

        # Calculate max steps if not provided
        # Default: 50 trials × 200 steps/trial = 10,000 steps
        if max_steps is None:
            max_steps = self.max_trials_per_episode * MAX_STEPS_PER_TRIAL

        # Call parent constructor (this calls _gen_grid internally)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,  # Enable full maze visibility (allocentric view)
            max_steps=max_steps,
            **kwargs  # Pass remaining kwargs to parent
        )

        # --- Define Action Space ---
        # IMPORTANT: We restrict to 3 actions (not MiniGrid's default 7)
        # 0 = turn left, 1 = turn right, 2 = move forward
        # We exclude: pickup, drop, toggle, done (not needed for this task)
        self.action_space = spaces.Discrete(3)

        # --- Define Observation Space ---
        # Simple local navigation representation per spec:
        # position (x, y) + facing direction — no explicit memory variables
        self.observation_space = spaces.Dict({
            # Direction agent is facing (0=E, 1=S, 2=W, 3=N)
            "direction": spaces.Discrete(4),

            # Agent (x, y) position in the grid
            "position_vector": spaces.Box(
                low=0,
                high=size,
                shape=(2,),
                dtype=np.float32
            ),
        })

        # --- Precompute Rewarded Poses ---
        # Only the correct approach direction counts — agent must come from the
        # T-junction arm side, not from the return arm below or any other direction.
        # Left well (4,4): agent must approach from the East → pose (5, 4, West=2)
        # Right well (10,4): agent must approach from the West → pose (9, 4, East=0)
        self.rewarded_poses_left = {(LEFT_WELL_LOC[0] + 1, LEFT_WELL_LOC[1], 2)}   # (5, 4, West)
        self.rewarded_poses_right = {(RIGHT_WELL_LOC[0] - 1, RIGHT_WELL_LOC[1], 0)} # (9, 4, East)

    @property
    def _dynamic_barriers(self) -> set:
        """All currently active barriers: placed + trailing."""
        s = set(self._stage1_placed_barriers)
        if self._trailing_barrier_cell is not None:
            s.add(self._trailing_barrier_cell)
        return s

    # =========================================================================
    # GRID GENERATION - BUILD THE MAZE STRUCTURE
    # =========================================================================

    def _gen_grid(self, width, height):
        """
        Generate the figure-8 T-maze layout.
        
        Called automatically by MiniGridEnv.__init__() and reset().
        
        Paper Connection: Replicates the physical maze structure from the paper:
        - Central stem (divided into 4 sectors for analysis)
        - Left and right choice arms
        - Return paths creating figure-8 pattern
        
        Args:
            width: Grid width (15)
            height: Grid height (15)
        """

        # --- Step 1: Create Empty Grid ---
        self.grid = Grid(width, height)

        # --- Step 2: Fill Grid with Walls ---
        # Start with all walls, then carve out paths
        # This ensures the maze is enclosed
        for i in range(width):
            for j in range(height):
                self.grid.set(i, j, MazeWall())  # Place wall at every position

        # --- Step 3: Build Central Stem ---
        # Paper Connection: "The central portion of the stem" analyzed in the paper
        # We carve out floor tiles and track which sector each belongs to

        for y in range(STEM_TOP, STEM_BOTTOM + 1):  # Y from 4 to 12 (top to bottom)

            # Determine which of the 4 sectors this Y-coordinate belongs to
            # This is CRITICAL for later analysis matching the paper's methodology

            if SECTOR_1_RANGE[0] <= y <= SECTOR_1_RANGE[1]:
                sector = 1  # Bottom sector (near start)
            elif SECTOR_2_RANGE[0] <= y <= SECTOR_2_RANGE[1]:
                sector = 2  # Second sector
            elif SECTOR_3_RANGE[0] <= y <= SECTOR_3_RANGE[1]:
                sector = 3  # Third sector
            elif SECTOR_4_RANGE[0] <= y <= SECTOR_4_RANGE[1]:
                sector = 4  # Top sector (near choice point)
            else:
                sector = 0  # Not on stem (shouldn't happen in this range)

            # Place a StemFloor tile (instead of wall) with sector label
            # This allows us to later query: env.grid.get(x, y).sector
            self.grid.set(STEM_X, y, StemFloor(sector=sector))

        # --- Step 4: Build Left Choice Arm ---
        # Horizontal path extending left from T-junction
        # From column (STEM_X - ARM_LENGTH) to STEM_X at row CHOICE_Y
        for x in range(LEFT_WELL_LOC[0], STEM_X):  # X from 4 to 6 (left to center)
            self.grid.set(x, CHOICE_Y, Floor())  # Place floor tile

        # --- Step 5: Build Right Choice Arm ---
        # Horizontal path extending right from T-junction
        for x in range(STEM_X + 1, RIGHT_WELL_LOC[0] + 1):  # X from 8 to 10 (center to right)
            self.grid.set(x, CHOICE_Y, Floor())

        # --- Step 6: Build Left Return Path ---
        # Vertical path connecting left arm back to start (figure-8 pattern)
        # Paper Connection: "The rat returned to the base of the stem via connecting arms"
        for y in range(CHOICE_Y, STEM_BOTTOM + 1):  # Y from 4 to 12 (top to bottom)
            self.grid.set(LEFT_RETURN_X, y, Floor())  # Column 4, all Y from choice to start

        # --- Step 7: Build Right Return Path ---
        # Mirror of left return path
        for y in range(CHOICE_Y, STEM_BOTTOM + 1):
            self.grid.set(RIGHT_RETURN_X, y, Floor())  # Column 10

        # --- Step 8: Build Bottom Connector ---
        # Horizontal path at bottom connecting left and right return paths
        # This completes the figure-8 circuit
        for x in range(LEFT_RETURN_X, RIGHT_RETURN_X + 1):  # X from 4 to 10 (left to right)
            self.grid.set(x, STEM_BOTTOM, Floor())  # Row 12 (bottom)

        # Now the agent can: start → stem → choice → return → start (continuous loop!)

        # --- Step 9: Place Water Wells ---
        # Paper Connection: "Water rewards... were provided at water ports"

        self.left_well = WaterWell(visible=True)   # Create left well object
        self.right_well = WaterWell(visible=True)  # Create right well object

        # Place wells at end of choice arms
        self.put_obj(self.left_well, *LEFT_WELL_LOC)    # Put at (4, 4)
        self.put_obj(self.right_well, *RIGHT_WELL_LOC)  # Put at (10, 4)

        # --- Step 10: Place Agent at Start ---
        self.agent_pos = self.agent_start_pos  # (7, 12) - base of stem
        self.agent_dir = self.agent_start_dir  # 3 = facing North

        # Set mission string
        self.mission = "Alternate between left and right choice arms"

    @staticmethod
    def _gen_mission():
        """
        Generate mission string shown to agent.
        
        Returns:
            String describing the task objective
        """
        return "Alternate between left and right choice arms"

    # =========================================================================
    # RESET - START NEW EPISODE
    # =========================================================================

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Reset environment for a new episode.
        
        Paper Connection: Each episode represents one "recording session"
        of 30-50 trials from the paper.
        
        IMPORTANT DESIGN DECISION:
        - Alternation history (last_choice) is RESET between episodes
        - But PERSISTS across trials within an episode (this is the working memory!)
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            obs: Initial observation
            info: Initial info dictionary
        """

        # Call parent reset (regenerates grid, resets step count, etc.)
        super().reset(seed=seed)

        # --- Reset Agent Position ---
        self.agent_pos = self.agent_start_pos  # Back to (7, 12)
        self.agent_dir = self.agent_start_dir  # Facing North

        # --- Reset Trial Tracking for New Episode ---
        # Paper Connection: Each recording session starts fresh
        self.trial_count = 0          # No trials completed yet
        self.last_choice = None       # No prior choice (first trial of session)
        self.correct_trials = 0       # No correct trials yet
        self.incorrect_trials = 0     # No incorrect trials yet
        self.trial_history = []       # Empty history
        self._at_well_side = None     # Guard: 'left'/'right'/None — prevents double-counting same well visit
        # 3-stage figure-8 loop gate (phase 0 = reward OK):
        #   1 → must reach correct return arm bottom corner
        #   2 → must reach stem bottom (STEM_X, STEM_BOTTOM)
        #   3 → must reach T-junction (STEM_X, STEM_TOP)
        self._loop_phase = 0
        self._loop_arm_bottom = None  # (x, STEM_BOTTOM) for the arm just visited

        # Reset Stage 1 barrier state
        self._barrier_state = 0
        self._trailing_barrier_cell = None
        if self.use_stage1_barriers:
            self._stage1_placed_barriers = {
                (STEM_X - 1, STEM_BOTTOM),  # (6,12)
                (STEM_X + 1, STEM_BOTTOM),  # (8,12)
            }
        else:
            self._stage1_placed_barriers = set()

        # --- Reset Trajectory Tracking ---
        self.trajectory = []  # No poses recorded yet
        self.current_trial_trajectory = [(*self.agent_pos, self.agent_dir)]  # Start with initial pose

        # --- Increment Episode Counter ---
        self.episode_count += 1

        # Generate and return initial observation
        return self.gen_obs(), self._get_info()

    # =========================================================================
    # STEP - EXECUTE ONE ACTION (CORE RL LOOP)
    # =========================================================================

    def step(self, action: int):
        """
        Execute one action and return next state, reward, termination flags.
        
        This is the core RL interaction: (s, a) → (s', r, done, info)
        
        Paper Connection: Each step represents one movement decision by the rat.
        Multiple steps compose one trial (start to choice to return).
        Multiple trials compose one episode (recording session).
        
        Args:
            action: Integer action to take
                0 = turn left (rotate counter-clockwise)
                1 = turn right (rotate clockwise)
                2 = move forward (if not blocked)
                
        Returns:
            obs: Next observation (s')
            reward: Reward received (r)
            terminated: Whether episode ended naturally (reached max trials)
            truncated: Whether episode timed out (exceeded max steps)
            info: Additional information dictionary
        """

        # --- Step Counter ---
        self.step_count += 1  # Track total steps taken this episode

        # --- Initialize Reward ---
        reward = 0.0  # Start with zero, will add components below
        terminated = False  # Episode not done yet
        truncated = False   # Not timed out yet

        # --- Apply Living Cost ---
        # Small negative reward each step to encourage efficiency
        reward += self.step_cost  # Typically -0.01

        # --- Execute Action ---
        # Modify agent's pose based on action taken

        if action == 0:  # Turn left (counter-clockwise)
            # Rotate direction: 3→2→1→0→3 (North→West→South→East→North)
            self.agent_dir = (self.agent_dir - 1) % 4  # Modulo wraps around
            reward += self.turn_cost  # Add turn cost (typically -0.01)

        elif action == 1:  # Turn right (clockwise)
            # Rotate direction: 3→0→1→2→3 (North→East→South→West→North)
            self.agent_dir = (self.agent_dir + 1) % 4
            reward += self.turn_cost

        elif action == 2:  # Move forward
            # Get position in front of agent based on current direction
            fwd_pos = self.front_pos  # MiniGrid property: calculates forward cell

            # Check what's in the forward cell
            fwd_cell = self.grid.get(*fwd_pos)

            # Only move if not blocked by a wall or a dynamic barrier
            if tuple(fwd_pos) not in self._dynamic_barriers and \
               (fwd_cell is None or fwd_cell.can_overlap()):
                prev_pos = self.agent_pos
                self.agent_pos = tuple(fwd_pos)
                # Update trailing barrier (one step behind the agent)
                if self.use_stage1_barriers:
                    self._trailing_barrier_cell = prev_pos
            else:
                # Agent bumped into a wall or barrier
                reward += self.wall_bump_penalty

        else:
            # Invalid action (should never happen if action_space is used correctly)
            raise ValueError("Invalid action. Use 0=left, 1=right, 2=forward")

        # --- Record Current Pose ---
        # Pose = (x, y, direction) - fully describes agent's configuration
        current_pose = (*self.agent_pos, self.agent_dir)
        self.trajectory.append(current_pose)  # Add to full episode trajectory
        self.current_trial_trajectory.append(current_pose)  # Add to current trial

        # --- Stage 1 barrier update (position-triggered) ---
        if self.use_stage1_barriers:
            self._update_stage1_barriers_on_step()

        # --- Check for Choice/Reward ---
        # Determine if agent has reached a reward well

        choice_made = None     # Will be 'left', 'right', or None
        choice_correct = None  # Will be True, False, or None

        # Clear per-side guard once agent leaves that side's rewarded poses
        if self._at_well_side == 'left' and current_pose not in self.rewarded_poses_left:
            self._at_well_side = None
        elif self._at_well_side == 'right' and current_pose not in self.rewarded_poses_right:
            self._at_well_side = None

        if self.use_stage1_barriers:
            # Stage 1: barriers enforce the circuit; _loop_phase not needed
            if self._at_well_side != 'left' and current_pose in self.rewarded_poses_left:
                choice_made = 'left'
            elif self._at_well_side != 'right' and current_pose in self.rewarded_poses_right:
                choice_made = 'right'
        else:
            # Stages 2/3: use loop-phase gate
            if self._loop_phase == 1 and self.agent_pos == self._loop_arm_bottom:
                self._loop_phase = 2
            elif self._loop_phase == 2 and self.agent_pos == (STEM_X, STEM_BOTTOM):
                self._loop_phase = 3
            elif self._loop_phase == 3 and self.agent_pos == (STEM_X, STEM_TOP):
                self._loop_phase = 0
                reward += self.loop_bonus

            if self._at_well_side != 'left' and self._loop_phase == 0 and current_pose in self.rewarded_poses_left:
                choice_made = 'left'
            elif self._at_well_side != 'right' and self._loop_phase == 0 and current_pose in self.rewarded_poses_right:
                choice_made = 'right'

        # --- Evaluate Alternation Rule ---
        # Paper Connection: "Correct alternations" were rewarded in the paper

        if choice_made is not None:
            # Agent made a choice! Now evaluate if it's correct

            self.trial_count += 1  # Increment trial counter

            # Foraging bonus: reward any well visit (curriculum Stage-1 scaffold)
            reward += self.foraging_reward

            # --- Case 1: First Trial of Episode ---
            if self.last_choice is None:
                # No prior choice to compare to
                # Paper Connection: First trial in a session is always rewarded
                choice_correct = True
                reward += self.correct_reward  # Add +1.0
                self.correct_trials += 1

            # --- Case 2: Correct Alternation ---
            elif choice_made != self.last_choice:
                # Chose opposite arm from last trial - CORRECT!
                choice_correct = True
                reward += self.correct_reward  # Add +1.0
                self.correct_trials += 1

            # --- Case 3: Incorrect (Repeated Choice) ---
            else:
                # Chose same arm as last trial - INCORRECT
                # Paper: "no reward was provided"
                choice_correct = False
                reward += self.incorrect_reward  # Add 0.0 (no reward)
                self.incorrect_trials += 1

            # --- Record Trial Outcome ---
            # Store detailed information about this trial for later analysis
            self.trial_history.append({
                'trial': self.trial_count,                    # Trial number
                'choice': choice_made,                        # 'left' or 'right'
                'correct': choice_correct,                    # True/False
                'reward': reward,                             # Total reward this step
                'trajectory': self.current_trial_trajectory.copy()  # Path taken
            })

            # --- Update Working Memory ---
            # CRITICAL: Remember this choice for next trial
            # This is the "episodic memory" component from the paper
            self.last_choice = choice_made
            self._at_well_side = choice_made  # Prevent re-triggering until agent leaves this well

            if self.use_stage1_barriers:
                # Stage 1: update dynamic barriers to guide agent through circuit
                self._update_stage1_barriers_on_choice(choice_made)
            else:
                # Stages 2/3: activate loop-phase gate
                self._loop_arm_bottom = (LEFT_RETURN_X, STEM_BOTTOM) if choice_made == 'left' else (RIGHT_RETURN_X, STEM_BOTTOM)
                self._loop_phase = 1

            # Reset trial trajectory (agent continues physically from well)
            self.current_trial_trajectory = [current_pose]

            # --- Check if Episode Should End ---
            # Paper Connection: Sessions lasted 30-50 trials
            if self.trial_count >= self.max_trials_per_episode:
                terminated = True  # Episode completed naturally

        # --- Check for Timeout ---
        # If agent takes too many steps, truncate episode
        if self.step_count >= self.max_steps:
            truncated = True  # Episode exceeded time limit

        # --- Return RL Tuple ---
        return self.gen_obs(), reward, terminated, truncated, self._get_info()

    # =========================================================================
    # OBSERVATION GENERATION
    # =========================================================================

    def gen_obs(self):
        """
        Generate observation dictionary for the agent.
        
        Returns:
            Dictionary containing:
            - image: RGB rendering of maze
            - direction: Current facing direction
            - mission: Task description
            - last_choice: Working memory (0=none, 1=left, 2=right)
            - trial_number: Current trial count
            - position_vector: (x, y) coordinates
        """

        obs = {
            "direction": self.agent_dir,
            "position_vector": np.array(self.agent_pos, dtype=np.float32),
        }

        return obs

    def _get_info(self):
        """
        Generate info dictionary with episode statistics.
        
        Returns:
            Dictionary with trial counts, accuracy, etc.
        """
        return {
            'trial_count': self.trial_count,
            'last_choice': self.last_choice,
            'correct_trials': self.correct_trials,
            'incorrect_trials': self.incorrect_trials,
            'accuracy': self.correct_trials / max(1, self.trial_count),  # Avoid division by zero
            'episode': self.episode_count,
        }

    # =========================================================================
    # ANALYSIS HELPERS (for teammates doing poster analysis)
    # =========================================================================

    def get_current_stem_sector(self):
        """
        Get which sector (1-4) the agent is currently in on the central stem.
        
        Paper Connection: Allows analyzing behavior/representations by sector,
        matching the paper's methodology of dividing stem into 4 sectors.
        
        Returns:
            int: Sector number (0=not on stem, 1-4=stem sectors)
            
        Example use by teammates:
            sector = env.get_current_stem_sector()
            if sector > 0:
                # Agent is on stem, record Q-values for this sector
                q_values_by_sector[sector].append(agent.get_q_values(obs))
        """
        x, y = self.agent_pos

        # Check if agent is on central stem column
        if x != STEM_X:
            return 0  # Not on stem

        # Get the cell object at current position
        cell = self.grid.get(x, y)

        # If it's a StemFloor tile, it has a sector attribute
        if isinstance(cell, StemFloor):
            return cell.sector  # Return 1, 2, 3, or 4

        return 0  # Not on stem (shouldn't happen if x == STEM_X)

    def get_trial_statistics(self):
        """
        Get summary statistics for the current episode.
        
        Returns:
            Dictionary with trial counts, accuracy, and full history
            
        Example use by teammates:
            stats = env.get_trial_statistics()
            print(f"Accuracy: {stats['accuracy']:.1%}")
            for trial in stats['trial_history']:
                print(f"Trial {trial['trial']}: {trial['choice']} - {trial['correct']}")
        """
        return {
            'total_trials': self.trial_count,
            'correct': self.correct_trials,
            'incorrect': self.incorrect_trials,
            'accuracy': self.correct_trials / max(1, self.trial_count),
            'trial_history': self.trial_history,  # Full detailed history
        }

    def export_trajectory_data(self):
        """
        Export trajectory data separated by trial type.
        
        Paper Connection: Enables analysis comparing left-turn vs right-turn
        trial trajectories, similar to how the paper analyzed neural firing
        separately for each trial type.
        
        Returns:
            Dictionary with:
            - 'left_trials': List of all left-turn trial data
            - 'right_trials': List of all right-turn trial data
            - 'all_trials': Complete trial history
            
        Example use by teammates (for poster analysis):
            data = env.export_trajectory_data()
            
            # Analyze left-turn trials only
            for trial in data['left_trials']:
                if trial['correct']:
                    # Plot trajectory for correct left turns
                    plot_trajectory(trial['trajectory'])
            
            # Compare left vs right
            left_accuracy = sum(t['correct'] for t in data['left_trials']) / len(data['left_trials'])
            right_accuracy = sum(t['correct'] for t in data['right_trials']) / 
len(data['right_trials'])
        """

        # Filter trials by choice type
        left_trials = [t for t in self.trial_history if t['choice'] == 'left']
        right_trials = [t for t in self.trial_history if t['choice'] == 'right']

        return {
            'left_trials': left_trials,
            'right_trials': right_trials,
            'all_trials': self.trial_history,
        }

    # =========================================================================
    # STAGE 1 DYNAMIC BARRIER STATE MACHINE
    # =========================================================================

    def _update_stage1_barriers_on_step(self):
        """
        Position-triggered placed-barrier updates for Stage 1.
        Called every step after agent_pos is updated.

        The trailing barrier (one step behind) is updated separately in step().
        Placed barriers here are all temporary — each is lifted at some point.

        State transitions:
          0 (stem traversal) → agent reaches T-junction (7,4) → 1 (choice)
          2 (left return)    → agent reaches start (7,12)     → 0
          3 (right return)   → agent reaches start (7,12)     → 0
        """
        pos = self.agent_pos

        if self._barrier_state == 0 and pos == (STEM_X, STEM_TOP):
            self._barrier_state = 1

        elif self._barrier_state == 2 and pos == (STEM_X, STEM_BOTTOM):
            # Agent completed left return — re-drop left flanking barrier
            self._stage1_placed_barriers.add((STEM_X - 1, STEM_BOTTOM))    # (6,12)
            self._barrier_state = 0

        elif self._barrier_state == 3 and pos == (STEM_X, STEM_BOTTOM):
            # Agent completed right return — re-drop right flanking barrier
            self._stage1_placed_barriers.add((STEM_X + 1, STEM_BOTTOM))    # (8,12)
            self._barrier_state = 0

    def _update_stage1_barriers_on_choice(self, choice_made: str):
        """
        Choice-triggered placed-barrier updates for Stage 1.
        Called immediately after a well reward is granted.

        Lifts the flanking barrier on the return side so the agent can
        traverse back to the start. Force-alternation barrier is also lifted.
        """
        if choice_made == 'left':
            if self.force_alternation_barriers:
                self._stage1_placed_barriers.add((STEM_X - 1, STEM_TOP))     # place (6,4) — block left arm
                self._stage1_placed_barriers.discard((STEM_X + 1, STEM_TOP)) # lift (8,4) — right arm now open
            self._stage1_placed_barriers.discard((STEM_X - 1, STEM_BOTTOM))  # lift (6,12) — open left return
            self._barrier_state = 2
        else:
            if self.force_alternation_barriers:
                self._stage1_placed_barriers.add((STEM_X + 1, STEM_TOP))     # place (8,4) — block right arm
                self._stage1_placed_barriers.discard((STEM_X - 1, STEM_TOP)) # lift (6,4) — left arm now open
            self._stage1_placed_barriers.discard((STEM_X + 1, STEM_BOTTOM))  # lift (8,12) — open right return
            self._barrier_state = 3