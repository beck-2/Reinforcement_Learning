# test_and_visualize.py
# =============================================================================
# TESTING AND VISUALIZATION SCRIPT
# Purpose: Verify environment works correctly and generate visualizations
# =============================================================================
#
# WHAT THIS SCRIPT DOES:
# 1. Creates the Figure-8 T-Maze environment
# 2. Tests it with a random agent
# 3. Generates maze structure image
# 4. Creates animated GIF of agent behavior
# 5. Prints statistics and next steps
#
# USAGE:
#   python test_and_visualize.py
#
# OUTPUTS:
#   - maze_structure.png: Static image of maze layout
#   - figure8_random_agent.gif: Animated GIF of random agent behavior
#   - Console output: Trial statistics and accuracy
#
# FOR TEAMMATES:
#   - Environment team: Use this to verify maze is correct
#   - Agent team: Replace random agent with your trained agent
#   - Analysis team: Modify to generate poster visualizations
#   - Poster team: Use outputs directly in poster
# =============================================================================

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from figure8_maze_env import Figure8TMazeEnv

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_maze_structure(env):
    """
    Display and save static image of the maze layout.
    
    Paper Connection: Shows the physical structure of the maze from the paper
    (central stem, choice arms, return paths).
    
    For Poster: Use this image in "Environment, Task, & Agent Description" section
    
    Args:
        env: Figure8TMazeEnv instance
    """

    # Reset environment to get clean initial state
    env.reset()

    # Render maze as RGB image
    img = env.render()

    # Create matplotlib figure
    plt.figure(figsize=(8, 8))
    plt.imshow(img)  # Display image
    plt.title("Figure-8 T-Maze Layout")
    plt.axis('off')  # Hide axis ticks
    plt.tight_layout()

    # Save to file
    plt.savefig('maze_structure.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Maze structure saved as 'maze_structure.png'")


def test_random_agent(env, num_trials=10):
    """
    Test environment with random action policy.
    
    Purpose:
    - Verify environment dynamics work correctly
    - Establish baseline performance (random should get ~50% accuracy)
    - Generate frames for visualization
    
    Paper Connection: Random agent should perform at chance level (~50% accuracy)
    since it can't remember previous choice. This demonstrates the task requires
    working memory, just like in the paper.
    
    Args:
        env: Figure8TMazeEnv instance
        num_trials: How many trials to run (default: 10)
        
    Returns:
        List of rendered frames for GIF creation
        
    For Teammates:
    - Agent team: Replace env.action_space.sample() with your policy
    - Analysis team: Collect Q-values or activations during this loop
    """

    # Reset environment
    obs, info = env.reset()

    # Print header
    print("\n" + "="*60)
    print("TESTING RANDOM AGENT")
    print("="*60)

    frames = []       # Store rendered frames for GIF
    trial_count = 0   # Track completed trials

    # Run until we complete desired number of trials
    while trial_count < num_trials:

        # --- Random Action Selection ---
        # For random agent: just sample uniformly from {0, 1, 2}
        action = env.action_space.sample()  # Randomly choose turn left/right/forward

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Render Frame ---
        # Capture current state as image for GIF
        frame = env.render()
        frames.append(frame)

        # --- Print Trial Outcomes ---
        # When reward > 0.1, agent completed a trial (reached a well)
        if reward > 0.1:  # Threshold to detect trial completion (reward is 1.0 - costs)
            # Determine if this was correct or incorrect
            trial_type = 'CORRECT' if reward > 0.5 else 'INCORRECT'
            print(f"Trial {info['trial_count']}: {env.last_choice} - "
                f"{trial_type} (reward={reward:.2f})")
            trial_count = info['trial_count']

        # --- Check for Episode End ---
        if terminated or truncated:
            break

    # --- Print Final Statistics ---
    stats = env.get_trial_statistics()
    print("\n" + "-"*60)
    print(f"Total trials: {stats['total_trials']}")
    print(f"Correct: {stats['correct']}")
    print(f"Incorrect: {stats['incorrect']}")
    print(f"Accuracy: {stats['accuracy']:.1%}")
    print("="*60 + "\n")

    # Expected: Random agent should get ~50% accuracy
    # If accuracy is very different, there may be a bug!

    return frames


def create_gif(frames, filename='random_agent_behavior.gif', fps=10):
    """
    Create animated GIF from list of frames.
    
    Purpose: Visualize agent behavior over time for poster/presentation
    
    Args:
        frames: List of numpy arrays (RGB images)
        filename: Output filename (default: 'random_agent_behavior.gif')
        fps: Frames per second for animation (default: 10)
        
    For Poster: Include GIF showing:
    - Random agent (baseline)
    - Trained agent (comparison)
    - Side-by-side if possible
    """

    # Check if we have frames to save
    if len(frames) == 0:
        print("No frames to save!")
        return

    # Create matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')  # Hide axis

    # Initialize with first frame
    im = ax.imshow(frames[0])

    def update(frame_idx):
        """Update function for animation - called for each frame."""
        im.set_array(frames[frame_idx])
        return [im]

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames),  # Total number of frames
        interval=1000//fps,  # Milliseconds between frames
        blit=True            # Optimize rendering
    )

    # Save as GIF
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close()

    print(f"Saved GIF: {filename}")


def plot_learning_curve_placeholder():
    """
    Placeholder reminder for agent team to generate learning curves.
    
    For Agent Team: After training, plot:
    1. Cumulative reward per episode vs episode number
    2. Accuracy (% correct alternations) vs episode number
    3. Average steps per trial vs episode number
    
    Expected curves:
    - Reward should increase from ~25 (random) to ~45 (near-perfect)
    - Accuracy should increase from ~50% to >95%
    - Steps per trial should decrease as agent learns efficient paths
    
    For Poster: Include learning curves in "Results" section
    """
    print("\n" + "="*60)
    print("PLACEHOLDER: Learning curve will be generated after RL training")
    print("Your agent team should plot:")
    print("  - Reward per episode vs episodes")
    print("  - Accuracy (% correct alternations) vs episodes")
    print("  - Steps per trial vs episodes")
    print("="*60 + "\n")


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":
    """
    Main execution: Create environment, test it, generate visualizations.
    
    To run: python test_and_visualize.py
    """

    # --- Create Environment ---
    print("Creating Figure-8 T-Maze environment...")
    env = Figure8TMazeEnv(
        render_mode='rgb_array',      # Render to numpy array (for saving images)
        max_trials_per_episode=10     # Short episode for quick testing
    )

    # --- Print Environment Info ---
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # --- Visualize Maze Structure ---
    # Generates: maze_structure.png
    visualize_maze_structure(env)

    # --- Test with Random Agent ---
    # Runs 10 trials and collects frames
    frames = test_random_agent(env, num_trials=10)

    # --- Create GIF ---
    # Generates: figure8_random_agent.gif
    create_gif(frames, filename='figure8_random_agent.gif', fps=10)

    # --- Placeholder for Learning Curves ---
    plot_learning_curve_placeholder()

    # --- Clean Up ---
    env.close()

    # --- Print Next Steps ---
    print("\nEnvironment testing complete!")
    print("\nNext steps for your team:")
    print("1. Agent team: Implement RL algorithm (Q-learning/SARSA/etc.)")
    print("2. Agent team: Train on this environment")
    print("3. Analysis team: Plot learning curves")
    print("4. Analysis team: Analyze trial-type specific behavior")
    print("5. Poster team: Create visualizations comparing to Wood et al. (2000)")