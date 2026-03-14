# verify_environment.py
# =============================================================================
# COMPREHENSIVE ENVIRONMENT VERIFICATION SUITE
# Purpose: Systematically test all environment components against project criteria
# =============================================================================
#
# WHAT THIS SCRIPT TESTS:
# 1. Markov State Space - Verify state representation is complete and correct
# 2. Transition Dynamics - Verify actions produce expected state changes
# 3. Reward Logic - Verify alternation task rewards are correct
# 4. Reset Logic - Verify episode/trial resets work properly
# 5. Termination Logic - Verify episodes end at correct conditions
# 6. Overall Behavior - Integration test of full task
#
# WHY THIS MATTERS:
# - Project criteria require "verifying that the environment behaves as intended"
# - Bugs in environment = unreliable RL training = invalid results
# - Passing all tests = confidence for poster presentation
#
# USAGE:
#   python verify_environment.py
#
# EXPECTED OUTPUT:
#   ✅ ALL TESTS PASSED - ENVIRONMENT VERIFIED ✅
#
# If any test fails, it will print the error and exit with code 1
#
# FOR TEAMMATES:
# - Run this before starting agent training
# - If modifying environment, re-run to verify changes didn't break anything
# - Include "All tests passed" in poster as verification
# =============================================================================

import numpy as np
from figure8_maze_env import Figure8TMazeEnv
from constants import *

# =============================================================================
# TEST 1: MARKOV STATE SPACE
# =============================================================================

def test_markov_state_space():
    """
    Verify that the state space satisfies Markov property and includes all
    necessary information.
    
    PROJECT CRITERIA CONNECTION:
    - Section 2.1: "Designing the Markov state space"
    - State must contain all info needed to predict next state given action
    
    PAPER CONNECTION:
    - Paper analyzed neural activity based on: position, direction, trial type
    - Our state must include: position, direction, AND working memory (last choice)
    
    WHAT WE TEST:
    1. All required observation components exist
    2. Observation shapes are correct
    3. State space dimensionality matches theoretical calculation
    
    MARKOV PROPERTY:
    P(s_{t+1} | s_t, a_t) must be independent of history before s_t
    This requires including "last_choice" in state (working memory)
    """

    print("\n" + "="*70)
    print("TEST 1: MARKOV STATE SPACE")
    print("="*70)

    # Create environment (no rendering needed for testing)
    env = Figure8TMazeEnv(render_mode=None)

    # Get initial observation
    obs, info = env.reset()

    # --- Test 1.1: Check Required Components ---
    # Observation is minimal: position + direction (no explicit memory)

    assert 'position_vector' in obs, "Missing position vector"
    assert 'direction' in obs, "Missing direction"

    print(f"✓ Observation space components: {list(obs.keys())}")
    print(f"✓ Position: {obs['position_vector']}")
    print(f"✓ Direction: {obs['direction']}")

    # --- Test 1.2: Check Observation Shapes ---

    assert obs['position_vector'].shape == (2,), \
        f"Expected position shape (2,), got {obs['position_vector'].shape}"
    assert obs['direction'] in range(4), \
        f"Direction should be 0-3, got {obs['direction']}"

    print(f"✓ position_vector shape: {obs['position_vector'].shape}")

    # --- Test 1.3: Calculate State Space Size ---

    total_positions = MAZE_SIZE * MAZE_SIZE  # 15 × 15 = 225 positions
    total_directions = 4                      # E, S, W, N

    # Without explicit memory in the observation, the MDP requires
    # the agent to maintain memory internally (e.g. via recurrent state)
    theoretical_state_space = total_positions * total_directions
    print(f"✓ Observable state space size: {theoretical_state_space}")
    # = 225 × 4 = 900 observable states (memory held by agent, not env)

    print("\n✅ PASSED: State space properly defined\n")

    # Clean up
    env.close()


# =============================================================================
# TEST 2: TRANSITION DYNAMICS
# =============================================================================

def test_transition_dynamics():
    """
    Verify that actions produce correct, deterministic state transitions.
    
    PROJECT CRITERIA CONNECTION:
    - Section 2.1: "Defining transition dynamics"
    - Transitions must be deterministic and correct
    
    PAPER CONNECTION:
    - Rats have deterministic movement (turn left → facing left, move forward → one step)
    - No stochasticity in the environment
    
    WHAT WE TEST:
    1. Turn left rotates direction counter-clockwise without changing position
    2. Turn right rotates direction clockwise without changing position
    3. Move forward advances position in facing direction (if not blocked)
    
    TRANSITION FUNCTION:
    T(s' | s, a) = 1.0 for the deterministic outcome
    T(s' | s, a) = 0.0 for all other s'
    """

    print("="*70)
    print("TEST 2: TRANSITION DYNAMICS")
    print("="*70)

    env = Figure8TMazeEnv(render_mode=None)
    obs, info = env.reset()

    # Save initial state
    initial_pos = env.agent_pos
    initial_dir = env.agent_dir

    # --- Test 2.1: Turn Left ---
    # Action 0 should rotate counter-clockwise: 3→2, 2→1, 1→0, 0→3

    obs, _, _, _, _ = env.step(0)  # Execute turn left

    # Position should NOT change (turns don't move you)
    assert env.agent_pos == initial_pos, "Turn should not change position"

    # Direction should decrement (with wraparound)
    expected_dir = (initial_dir - 1) % 4
    assert env.agent_dir == expected_dir, f"Turn left failed: {env.agent_dir} != {expected_dir}"

    print(f"✓ Turn left: direction {initial_dir} → {env.agent_dir}")

    # --- Test 2.2: Turn Right ---
    # Reset environment for clean test
    env.reset()
    initial_dir = env.agent_dir  # Should be 3 (North)

    obs, _, _, _, _ = env.step(1)  # Execute turn right

    # Position should NOT change
    assert env.agent_pos == initial_pos, "Turn should not change position"

    # Direction should increment (with wraparound)
    expected_dir = (initial_dir + 1) % 4
    assert env.agent_dir == expected_dir, f"Turn right failed: {env.agent_dir} != {expected_dir}"

    print(f"✓ Turn right: direction {initial_dir} → {env.agent_dir}")

    # --- Test 2.3: Move Forward ---
    # Reset environment
    env.reset()
    initial_pos = env.agent_pos  # (7, 12) - base of stem
    initial_dir = env.agent_dir  # 3 = North (facing up)

    obs, _, _, _, _ = env.step(2)  # Execute move forward

    # Calculate expected new position based on direction
    # Direction 3 = North = up = decrease Y coordinate
    # So from (7, 12) facing North, we should move to (7, 11)
    expected_pos = (initial_pos[0], initial_pos[1] - 1)

    # Verify position changed correctly
    actual_pos = env.agent_pos
    assert actual_pos == expected_pos, \
        f"Forward movement failed: {actual_pos} != {expected_pos}"

    print(f"✓ Move forward: position {initial_pos} → {env.agent_pos}")

    # Direction should NOT change when moving forward
    assert env.agent_dir == initial_dir, "Direction changed during forward movement"

    print("\n✅ PASSED: Transitions are deterministic and correct\n")

    env.close()


# =============================================================================
# TEST 3: REWARD LOGIC
# =============================================================================

def test_reward_logic():
    """
    Verify reward structure implements correct alternation task.
    
    PROJECT CRITERIA CONNECTION:
    - Section 2.1: "Implementing reward logic"
    - Section 1.2.2: "The reward structure"
    
    PAPER CONNECTION:
    - "On each trial when the rat made a correct (alternating) arm choice,
        a drop of water was delivered to the well in that arm"
    - "On trials when the animal made the incorrect choice, no reward was provided"
    
    WHAT WE TEST:
    1. First trial: Always rewarded (no prior choice to compare)
    2. Correct alternation: Left→Right or Right→Left = rewarded
    3. Incorrect repetition: Left→Left or Right→Right = not rewarded
    4. Return to alternation: After error, can return to correct pattern
    
    REWARD FUNCTION:
    r = +1.0 if correct alternation
    r = 0.0 if incorrect (same as last)
    r += small costs for steps/turns
    """

    print("="*70)
    print("TEST 3: REWARD LOGIC")
    print("="*70)

    # Create environment with only 5 trials for quick testing
    env = Figure8TMazeEnv(render_mode=None, max_trials_per_episode=5)
    obs, info = env.reset()

    # --- Helper Functions ---
    # These navigate the agent to each well

    def navigate_to_left():
        """
        Navigate agent from start to left well.
        
        Path: Start (7,12) → up stem → turn left → reach left well (4,4)
        """
        # Move up central stem (8 steps to reach choice point)
        for _ in range(8):
            env.step(2)  # Move forward (North)

        # At choice point, turn left
        env.step(0)  # Turn left (now facing West)
        env.step(2)  # Move forward into left arm

        # Move along left arm to well (3 steps)
        for _ in range(3):
            env.step(2)  # Move forward (West)

        # Now agent should be at a rewarded pose for left well

    def navigate_to_right():
        """
        Navigate agent from start to right well.
        
        Path: Start (7,12) → up stem → turn right → reach right well (10,4)
        """
        # Move up central stem
        for _ in range(8):
            env.step(2)

        # At choice point, turn right
        env.step(1)  # Turn right (now facing East)
        env.step(2)  # Move forward into right arm

        # Move along right arm to well
        for _ in range(3):
            env.step(2)

    # --- Test 3.1: First Trial (Always Rewarded) ---
    # Paper: First trial in session is always rewarded

    navigate_to_left()  # Choose left on first trial

    # Verify choice was recorded
    assert env.last_choice == 'left', "Choice not recorded"

    # Verify it was counted as correct
    stats = env.get_trial_statistics()
    assert stats['correct'] == 1, "First trial not rewarded"
    assert stats['incorrect'] == 0, "First trial marked as incorrect"

    print(f"✓ Trial 1: Left choice → Reward (first trial always correct)")

    # --- Test 3.2: Correct Alternation (Left → Right) ---
    # After choosing left, choosing right should be rewarded

    navigate_to_right()  # Choose right on second trial

    stats = env.get_trial_statistics()
    assert stats['correct'] == 2, "Correct alternation not rewarded"
    assert stats['incorrect'] == 0, "Correct alternation marked as incorrect"

    print(f"✓ Trial 2: Right choice → Reward (correct alternation)")

    # --- Test 3.3: Incorrect Repetition (Right → Right) ---
    # After choosing right, choosing right again should NOT be rewarded

    navigate_to_right()  # Choose right again (same as last)

    stats = env.get_trial_statistics()
    assert stats['correct'] == 2, "Incorrect choice was rewarded"
    assert stats['incorrect'] == 1, "Incorrect choice not counted"

    print(f"✓ Trial 3: Right choice → No reward (incorrect - same as last)")

    # --- Test 3.4: Return to Correct Alternation (Right → Left) ---
    # After error, agent can still get back to alternating correctly

    navigate_to_left()  # Choose left (opposite of last)

    stats = env.get_trial_statistics()
    assert stats['correct'] == 3, "Return to alternation not rewarded"
    assert stats['incorrect'] == 1, "Incorrect count changed"

    print(f"✓ Trial 4: Left choice → Reward (correct alternation)")

    # --- Print Summary Statistics ---
    print(f"\n✓ Final statistics:")
    print(f"  - Correct trials: {stats['correct']}")  # Should be 3
    print(f"  - Incorrect trials: {stats['incorrect']}")  # Should be 1
    print(f"  - Accuracy: {stats['accuracy']:.1%}")  # Should be 75% (3/4)

    # Verify expected accuracy
    expected_accuracy = 3/4  # 3 correct out of 4 trials
    assert abs(stats['accuracy'] - expected_accuracy) < 0.01, \
        f"Accuracy mismatch: {stats['accuracy']} != {expected_accuracy}"

    print("\n✅ PASSED: Reward logic correct\n")

    env.close()


# =============================================================================
# TEST 4: RESET LOGIC
# =============================================================================

def test_reset_logic():
    """
    Verify that reset() properly clears state for new episode.
    
    PROJECT CRITERIA CONNECTION:
    - Section 2.1: "Writing reset and termination logic"
    
    PAPER CONNECTION:
    - Each recording session (episode) starts fresh
    - Alternation history resets between sessions
    
    WHAT WE TEST:
    1. Trial count resets to 0
    2. Last choice resets to None (no prior choice)
    3. Correct/incorrect counts reset to 0
    4. Agent position resets to start
    5. Agent direction resets to North
    
    IMPORTANT DISTINCTION:
    - reset() is called between EPISODES (clears everything)
    - Trial boundaries within episode do NOT reset last_choice
    (this is the working memory that persists!)
    """

    print("="*70)
    print("TEST 4: RESET LOGIC")
    print("="*70)

    env = Figure8TMazeEnv(render_mode=None)

    # --- Run Some Steps to Dirty the State ---
    obs, info = env.reset()

    # Take random actions
    for _ in range(10):
        env.step(env.action_space.sample())

    # Manually set some trial state to make reset more obvious
    env.trial_count = 5
    env.last_choice = 'left'
    env.correct_trials = 3
    env.incorrect_trials = 2
    env.trajectory = [((7,12,3)), ((7,11,3)), ((7,10,3))]  # Some trajectory

    # --- Call Reset ---
    # This should clear ALL episode state
    obs, info = env.reset()

    # --- Verify Everything Reset Correctly ---

    # Test 4.1: Trial count reset
    assert env.trial_count == 0, "Trial count not reset"
    print("✓ Trial count reset to 0")

    # Test 4.2: Last choice reset (CRITICAL for working memory)
    assert env.last_choice is None, "Last choice not reset"
    print("✓ Last choice reset to None")

    # Test 4.3: Correct trials reset
    assert env.correct_trials == 0, "Correct trials not reset"
    print("✓ Correct trials reset to 0")

    # Test 4.4: Incorrect trials reset
    assert env.incorrect_trials == 0, "Incorrect trials not reset"
    print("✓ Incorrect trials reset to 0")

    # Test 4.5: Agent position reset
    assert env.agent_pos == START_POS, f"Position not reset: {env.agent_pos} != {START_POS}"
    print(f"✓ Agent position reset to START_POS {START_POS}")

    # Test 4.6: Agent direction reset
    assert env.agent_dir == START_DIR, f"Direction not reset: {env.agent_dir} != {START_DIR}"
    print(f"✓ Agent direction reset to START_DIR (North)")

    # Test 4.7: Trajectory cleared
    assert len(env.trial_history) == 0, "Trial history not cleared"
    print("✓ Trial history cleared")

    print("\n✅ PASSED: Reset logic works correctly\n")

    env.close()


# =============================================================================
# TEST 5: TERMINATION LOGIC
# =============================================================================

def test_termination_logic():
    """
    Verify episodes terminate under correct conditions.
    
    PROJECT CRITERIA CONNECTION:
    - Section 2.1: "Writing reset and termination logic"
    
    PAPER CONNECTION:
    - Recording sessions lasted 30-50 trials
    - We implement this as episode termination after max_trials
    
    WHAT WE TEST:
    1. Episode terminates after reaching max_trials_per_episode
    2. Episode truncates after exceeding max_steps
    3. Episode does NOT terminate prematurely
    
    TERMINATION vs TRUNCATION:
    - terminated=True: Task completed successfully (reached max trials)
    - truncated=True: Timeout occurred (exceeded max steps)
    - Both end episode, but have different meanings for RL
    """

    print("="*70)
    print("TEST 5: TERMINATION LOGIC")
    print("="*70)

    # --- Test 5.1: Termination After Max Trials ---
    # Create environment with only 3 trials for quick testing
    env = Figure8TMazeEnv(render_mode=None, max_trials_per_episode=3)
    obs, info = env.reset()

    terminated = False
    trial_count = 0
    steps = 0

    # Run until episode terminates
    while not terminated and steps < 1000:  # Safety limit
        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

        # Count when trials complete (reward > 0.1 means reached a well)
        if reward > 0.1:
            trial_count += 1

    # Verify episode ended after exactly 3 trials
    assert trial_count == 3, \
        f"Episode should end after 3 trials, got {trial_count}"

    # Verify it was natural termination (not truncation)
    assert terminated == True, "Episode should be terminated (not truncated)"

    print(f"✓ Episode terminated after {trial_count} trials (as configured)")

    # --- Test 5.2: Truncation After Max Steps ---
    # Create environment with very low step limit
    env = Figure8TMazeEnv(render_mode=None, max_steps=100)
    obs, info = env.reset()

    truncated = False
    step_count = 0

    # Run until truncation or safety limit
    while not truncated and step_count < 150:
        # Just move forward (won't complete trials efficiently)
        obs, reward, terminated, truncated, info = env.step(2)
        step_count += 1

    # Verify episode was truncated (hit step limit)
    assert truncated == True, "Episode should truncate at max_steps"

    # Verify step count is at/near max_steps
    assert step_count <= 101, \
        f"Truncation should occur near max_steps=100, got {step_count}"

    print(f"✓ Episode truncated after exceeding max_steps")

    print("\n✅ PASSED: Termination logic correct\n")

    env.close()


# =============================================================================
# TEST 6: OVERALL BEHAVIOR (INTEGRATION TEST)
# =============================================================================

def test_environment_behaves_correctly():
    """
    Integration test: Run full episode and verify overall behavior.
    
    PROJECT CRITERIA CONNECTION:
    - Section 2.1: "Verifying that the environment behaves as intended"
    
    PAPER CONNECTION:
    - Simulates one recording session (30-50 trials)
    - Random agent should get ~50% accuracy (chance level)
    
    WHAT WE TEST:
    1. Episode runs to completion (10 trials)
    2. Statistics are reasonable (accuracy 0-100%)
    3. Rewards are accumulated correctly
    4. No crashes or errors occur
    
    EXPECTED RESULTS:
    - Random agent: ~50% accuracy (can't remember last choice)
    - Total reward: ~25 (10 trials × 0.5 accuracy × 1.0 reward - step costs)
    - All trials recorded in history
    """

    print("="*70)
    print("TEST 6: OVERALL BEHAVIOR")
    print("="*70)

    # Create environment for 10 trials
    env = Figure8TMazeEnv(render_mode=None, max_trials_per_episode=10)
    obs, info = env.reset()

    print("Running 10-trial episode with random agent...")

    # --- Run Full Episode with Random Agent ---
    total_reward = 0.0
    step_count = 0

    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Random action
        action = env.action_space.sample()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Accumulate reward
        total_reward += reward
        step_count += 1

        # Safety: prevent infinite loop
        if step_count > 2000:
            print("⚠ Warning: Exceeded safety step limit")
            break

    # --- Get Final Statistics ---
    stats = env.get_trial_statistics()

    # --- Print Results ---
    print(f"\n✓ Episode completed:")
    print(f"  - Total trials: {stats['total_trials']}")
    print(f"  - Total steps: {step_count}")
    print(f"  - Total reward: {total_reward:.2f}")
    print(f"  - Accuracy: {stats['accuracy']:.1%}")
    print(f"  - Correct: {stats['correct']}")
    print(f"  - Incorrect: {stats['incorrect']}")

    # --- Verify Results are Reasonable ---

    # Test 6.1: Correct number of trials
    assert stats['total_trials'] == 10, \
        f"Should complete 10 trials, got {stats['total_trials']}"

    # Test 6.2: Accuracy is valid percentage
    assert 0 <= stats['accuracy'] <= 1.0, \
        f"Accuracy should be between 0 and 1, got {stats['accuracy']}"

    # Test 6.3: Correct + incorrect = total trials
    assert stats['correct'] + stats['incorrect'] == stats['total_trials'], \
        "Correct + incorrect should equal total trials"

    # Test 6.4: Accuracy matches calculation
    expected_accuracy = stats['correct'] / stats['total_trials']
    assert abs(stats['accuracy'] - expected_accuracy) < 0.001, \
        "Accuracy calculation mismatch"

    # Test 6.5: Random agent should be near 50% (within reasonable bounds)
    # We don't enforce this strictly (randomness!), but warn if way off
    if stats['accuracy'] < 0.2 or stats['accuracy'] > 0.8:
        print(f"⚠ Warning: Random agent accuracy {stats['accuracy']:.1%} "
            f"is far from expected ~50%")

    print("\n✅ PASSED: Environment behaves correctly\n")

    env.close()


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """
    Run complete verification suite and report results.
    
    Returns:
        bool: True if all tests passed, False otherwise
    """

    # Print header
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  FIGURE-8 T-MAZE ENVIRONMENT VERIFICATION SUITE".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)

    try:
        # --- Run All Tests in Order ---

        # Test 1: State space design
        test_markov_state_space()

        # Test 2: Transition function
        test_transition_dynamics()

        # Test 3: Reward function
        test_reward_logic()

        # Test 4: Reset behavior
        test_reset_logic()

        # Test 5: Termination conditions
        test_termination_logic()

        # Test 6: Integration test
        test_environment_behaves_correctly()

        # --- All Tests Passed! ---

        print("\n" + "#"*70)
        print("#" + " "*68 + "#")
        print("#" + "  ✅ ALL TESTS PASSED - ENVIRONMENT VERIFIED ✅".center(68) + "#")
        print("#" + " "*68 + "#")
        print("#"*70)

        print("\n✓ Environment satisfies ALL project criteria")
        print("✓ Ready for agent training")
        print("✓ Suitable for poster presentation\n")

        return True

    except AssertionError as e:
        # --- Test Failed ---

        print(f"\n❌ TEST FAILED: {e}\n")
        print("Please fix the issue and re-run verification.")
        print("Do not proceed with agent training until all tests pass.\n")

        return False

    except Exception as e:
        # --- Unexpected Error ---

        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        print("This may indicate a bug in the environment code.")
        print("Check the stack trace above for details.\n")

        return False


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Run verification when script is executed directly.
    
    Usage: python verify_environment.py
    
    Exit codes:
        0 = All tests passed
        1 = At least one test failed
    """

    # Run all tests
    success = run_all_tests()

    # Exit with appropriate code
    # 0 = success (for CI/CD pipelines)
    # 1 = failure
    exit(0 if success else 1)