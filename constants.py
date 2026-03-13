# constants.py
# =============================================================================
# FIGURE-8 T-MAZE ENVIRONMENT CONSTANTS
# Based on Wood et al. (2000): "Hippocampal Neurons Encode Information about 
# Different Types of Memory Episodes Occurring in the Same Location"
# =============================================================================
# 
# PURPOSE: Define all spatial parameters and task specifications that replicate
# the experimental setup from the paper where rats performed continuous spatial
# alternation on a modified T-maze.
#
# PAPER CONNECTION: The paper used a T-maze with:
# - A central stem traversed on every trial
# - Left and right choice arms
# - Connecting return arms forming a figure-8 pattern
# - Water rewards at the end of each choice arm
# - Continuous alternation requirement (left-right-left-right...)
# =============================================================================

# --- Maze Layout Coordinates ---
# These define the spatial structure of the environment

# MAZE_SIZE: Total grid dimension (15x15 cells)
# Paper Connection: The maze needs to be large enough to contain:
# - A central stem divided into 4 sectors (as analyzed in the paper)
# - Choice arms extending left and right
# - Return paths for the figure-8 structure
MAZE_SIZE = 15

# Central stem configuration
# Paper Connection: The paper specifically analyzed neural activity on the
# "central portion of the stem" - the shared path traversed on both trial types
STEM_X = 7  # Center column - the stem runs vertically down column 7
            # This creates a symmetric maze with equal-length arms

STEM_BOTTOM = 12  # Y-coordinate where the stem begins (rat start position)
                # Higher Y values = lower on screen (standard grid convention)

STEM_TOP = 4  # Y-coordinate of the T-junction (choice point)
            # This is where the rat must decide left vs right

# --- Sector Division (CRITICAL for replicating paper's analysis) ---
# Paper Connection: Wood et al. (2000) divided the central stem into 4 sectors
# and analyzed neural firing rates separately for each sector. Quote from paper:
# "For each of these 33 cells... the firing rate was calculated on every trial
# for each of four central sectors of the stem"

STEM_LENGTH = STEM_BOTTOM - STEM_TOP  # Total length of stem = 8 cells
SECTOR_HEIGHT = STEM_LENGTH // 4       # Each sector = 2 cells tall

# Define Y-coordinate ranges for each sector (from bottom to top)
# Sector 1 = Bottom of stem (near start)
# Sector 4 = Top of stem (near choice point)
SECTOR_1_RANGE = (STEM_BOTTOM - SECTOR_HEIGHT, STEM_BOTTOM)
# Range: (10, 12) - Agent just started moving up stem

SECTOR_2_RANGE = (STEM_BOTTOM - 2*SECTOR_HEIGHT, STEM_BOTTOM - SECTOR_HEIGHT)
# Range: (8, 10) - Agent is 1/4 way up stem

SECTOR_3_RANGE = (STEM_BOTTOM - 3*SECTOR_HEIGHT, STEM_BOTTOM - 2*SECTOR_HEIGHT)
# Range: (6, 8) - Agent is 3/4 way up stem

SECTOR_4_RANGE = (STEM_TOP, STEM_BOTTOM - 3*SECTOR_HEIGHT)
# Range: (4, 6) - Agent approaching choice point

# IMPORTANCE: These sectors allow us to later analyze if our RL agent's
# internal representations (e.g., value functions, hidden layer activations)
# differ between left-turn and right-turn trials in the same spatial location,
# just like the hippocampal neurons in the paper

# --- Choice Arm Configuration ---
ARM_LENGTH = 3  # How far the choice arms extend left/right from the stem
                # Paper Connection: Arms must be long enough to clearly
                # distinguish left vs right choices spatially

CHOICE_Y = STEM_TOP  # Y-coordinate where choice arms branch off
                    # This is the T-junction where alternation decisions occur

# --- Reward Well Locations ---
# Paper Connection: "Water rewards for correct alternations were provided at
# water ports on the end of each choice arm"
LEFT_WELL_LOC = (STEM_X - ARM_LENGTH, CHOICE_Y)   # (4, 4) - Left arm endpoint
RIGHT_WELL_LOC = (STEM_X + ARM_LENGTH, CHOICE_Y)  # (10, 4) - Right arm endpoint

# --- Return Path Configuration (creates the "figure-8" pattern) ---
# Paper Connection: "The rat returned to the base of the stem via connecting
# arms, and then traversed the central stem again on the next trial"
LEFT_RETURN_X = STEM_X - ARM_LENGTH   # X-coordinate of left return path (column 4)
RIGHT_RETURN_X = STEM_X + ARM_LENGTH  # X-coordinate of right return path (column 10)

# These vertical paths connect the choice arms back to the start, allowing
# continuous running without manually resetting the rat

# --- Agent Start Configuration ---
START_POS = (STEM_X, STEM_BOTTOM)  # (7, 12) - Base of central stem
                                    # Agent begins each trial here

START_DIR = 3  # Direction: 0=East, 1=South, 2=West, 3=North
                # Starting facing North (upward) toward the choice point
                # Paper Connection: Rats naturally run "up" the stem toward choices

# --- Rendering Constants ---
# These control visualization and agent perception

AGENT_VIEW_SIZE = 15  # Size of agent's visual field (15x15 = full maze)
                    # Paper Connection: Rats can see the entire maze from any
                    # position. This is "allocentric" (world-centered) rather
                    # than "egocentric" (body-centered) representation
                    # Hippocampus is known for allocentric spatial coding

VIEW_TILE_SIZE = 32  # Pixel size of each grid cell when rendering
                    # Larger = more detailed visualization

RENDER_FPS = 10  # Frames per second for animations
                # 10 FPS gives smooth but not overwhelming visualization

# --- Task Parameters ---
# Paper Connection: "Recording sessions consisted of 30-50 trials composed of
# an equal number of left-turn and right-turn trials"
MAX_TRIALS_PER_EPISODE = 50  # One "session" = 50 trials (upper bound from paper)
                            # This means one episode = 50 left/right decisions

MAX_STEPS_PER_TRIAL = 200  # Maximum timesteps allowed per single trial
                            # Prevents agent from wandering infinitely
                            # A well-trained agent should complete trial in 15-20 steps

# --- Reward Structure ---
# These define the reinforcement learning objective

# Primary reward: Correct alternation
# Paper Connection: "On each trial when the rat made a correct (alternating)
# arm choice, a drop of water was delivered to the well in that arm"
CORRECT_ALTERNATION_REWARD = 1.0  # +1 for choosing opposite arm from last trial

# No reward for incorrect alternation
# Paper Connection: "On trials when the animal made the incorrect choice,
# no reward was provided"
INCORRECT_ALTERNATION_REWARD = 0.0  # 0 for repeating the same choice

# Secondary costs (to shape efficient behavior)
# These are NOT in the original paper but are standard RL practices to:
# 1. Encourage finding shortest path
# 2. Prevent spinning in place or random wandering
STEP_COST = -0.01  # Small penalty per timestep (-0.01 per step)
                    # Encourages agent to complete trials quickly

TURN_COST = -0.01  # Small penalty per turn action (-0.01 per turn)
                    # Discourages excessive turning/spinning behavior

# IMPORTANT NOTE FOR TEAMMATES:
# If you want to study "pure" alternation learning without efficiency pressure,
# you can set STEP_COST = 0.0 and TURN_COST = 0.0
# This will make the agent only care about alternation, not speed