from figure8_maze_env import Figure8TMazeEnv
from train import apply_stage

env = Figure8TMazeEnv(render_mode=None, max_trials_per_episode=20)
apply_stage(env, 2)
obs, _ = env.reset()

for _ in range(5000):
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    if term or trunc:
        break

print("Stage 2 trial history (first 6):")
for t in env.trial_history[:6]:
    print(f"  trial {t['trial']}: choice={t['choice']}, correct={t['correct']}, reward={t['reward']:.2f}")
print(f"Total trials: {env.trial_count}, correct: {env.correct_trials}, incorrect: {env.incorrect_trials}")
