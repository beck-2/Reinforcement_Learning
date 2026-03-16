"""Quick smoke test: verify _at_well_side guard prevents re-triggering."""
from figure8_maze_env import Figure8TMazeEnv
from ppo import apply_stage

env = Figure8TMazeEnv(render_mode=None, max_trials_per_episode=20)
apply_stage(env, 1)
env.reset()

visits = 0
for _ in range(3000):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if info['trial_count'] > visits:
        visits = info['trial_count']
        print(f"  Visit {visits}: {info['last_choice']}  acc={info['accuracy']:.0%}")
    if visits >= 8 or term or trunc:
        break

env.close()
