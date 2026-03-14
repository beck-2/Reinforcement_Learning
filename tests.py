"""
Test suite covering environment correctness, model shapes, and training health.

Run with:  python tests.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from figure8_maze_env import Figure8TMazeEnv
from model import RecurrentActorCritic
from config import Config
from train import obs_to_tensor, compute_returns
from constants import START_POS, START_DIR, CORRECT_ALTERNATION_REWARD, INCORRECT_ALTERNATION_REWARD


# ── helpers ────────────────────────────────────────────────────────────────────

def make_env(**kwargs):
    return Figure8TMazeEnv(render_mode=None, step_cost=0.0, turn_cost=0.0, **kwargs)


def make_model(cfg=None):
    cfg = cfg or Config()
    return RecurrentActorCritic(cfg.obs_dim, cfg.hidden_size, cfg.num_actions)


def force_choice(env, choice: str) -> float:
    """
    Teleport the agent to a pose one step before a rewarded pose for the
    given choice arm, then execute a forward step to land on the reward.

    Valid pre-reward positions (verified against maze layout):
      left:  (6, 4) facing West  → forward lands on (5, 4, West) ∈ rewarded_poses_left
      right: (8, 4) facing East  → forward lands on (9, 4, East) ∈ rewarded_poses_right

    Clears _at_well_side before teleporting since this helper bypasses real navigation.
    """
    env._at_well_side = None     # simulate having navigated away from any prior well
    env._loop_phase = 0          # simulate having completed the full figure-8 loop
    env._loop_arm_bottom = None
    if choice == 'left':
        env.agent_pos = (6, 4)
        env.agent_dir = 2   # West
    else:
        env.agent_pos = (8, 4)
        env.agent_dir = 0   # East
    _, reward, _, _, _ = env.step(2)  # forward
    return reward


# ── Environment tests ──────────────────────────────────────────────────────────

def test_maze_resets_correctly():
    env = make_env()
    env.reset()
    assert env.agent_pos == START_POS, f"pos={env.agent_pos}"
    assert env.agent_dir == START_DIR, f"dir={env.agent_dir}"
    assert env.trial_count == 0
    assert env.last_choice is None
    print("  ✓ maze resets correctly")


def test_correct_alternation_reward():
    env = make_env()
    env.reset()
    r = force_choice(env, 'left')
    assert r == CORRECT_ALTERNATION_REWARD, f"first trial: expected {CORRECT_ALTERNATION_REWARD}, got {r}"
    r = force_choice(env, 'right')
    assert r == CORRECT_ALTERNATION_REWARD, f"alternation: expected {CORRECT_ALTERNATION_REWARD}, got {r}"
    print("  ✓ correct reward for valid alternation")


def test_no_reward_for_repeat():
    env = make_env()
    env.reset()
    force_choice(env, 'left')
    r = force_choice(env, 'left')   # repeat
    assert r == INCORRECT_ALTERNATION_REWARD, f"repeat: expected {INCORRECT_ALTERNATION_REWARD}, got {r}"
    print("  ✓ no reward for repeating same arm")


def test_episode_continues_after_error():
    env = make_env(max_trials_per_episode=10)
    env.reset()
    force_choice(env, 'left')
    force_choice(env, 'left')   # error
    assert env.trial_count == 2
    assert env.trial_count < env.max_trials_per_episode
    print("  ✓ episode continues after errors")


def test_repeated_traversal_loop():
    env = make_env(max_trials_per_episode=20)
    env.reset()
    choices = ['left', 'right'] * 10
    for c in choices:
        force_choice(env, c)
    assert env.trial_count == 20
    print("  ✓ repeated traversal loop works")


# ── Model tests ────────────────────────────────────────────────────────────────

def test_forward_pass_shapes():
    cfg = Config()
    model = make_model(cfg)
    obs = torch.zeros(1, cfg.obs_dim)
    h = model.init_hidden()
    logits, value, new_h = model(obs, h)
    assert logits.shape == (1, cfg.num_actions), f"logits: {logits.shape}"
    assert value.shape == (1, 1), f"value: {value.shape}"
    assert new_h.shape == (1, 1, cfg.hidden_size), f"hidden: {new_h.shape}"
    print("  ✓ forward pass tensor shapes correct")


def test_hidden_state_updates():
    cfg = Config()
    model = make_model(cfg)
    obs = torch.randn(1, cfg.obs_dim)
    h0 = model.init_hidden()
    _, _, h1 = model(obs, h0)
    assert not torch.allclose(h0, h1), "Hidden state should change after non-trivial input"
    print("  ✓ hidden state updates after forward pass")


def test_hidden_state_reset_between_episodes():
    cfg = Config()
    model = make_model(cfg)
    h1 = model.init_hidden()
    h2 = model.init_hidden()
    assert torch.allclose(h1, torch.zeros_like(h1))
    assert torch.allclose(h1, h2)
    print("  ✓ hidden state resets to zeros between episodes")


def test_actor_output_size():
    cfg = Config()
    model = make_model(cfg)
    logits, _, _ = model(torch.zeros(1, cfg.obs_dim), model.init_hidden())
    assert logits.shape[-1] == 3
    print("  ✓ actor output size matches action space (3)")


def test_critic_outputs_scalar():
    cfg = Config()
    model = make_model(cfg)
    _, value, _ = model(torch.zeros(1, cfg.obs_dim), model.init_hidden())
    assert value.shape == (1, 1)
    print("  ✓ critic outputs scalar value")


# ── Training health tests ──────────────────────────────────────────────────────

def _collect_mini_rollout(model, env, rollout_len=8):
    """Collect a short rollout, returning tensors needed for loss computation."""
    cfg = Config()
    obs, _ = env.reset()
    hidden = model.init_hidden()
    values, log_probs, rewards, dones, entropies = [], [], [], [], []

    for _ in range(rollout_len):
        obs_t = obs_to_tensor(obs)
        logits, value, hidden = model(obs_t, hidden)
        dist = Categorical(logits=logits)
        action = dist.sample()
        next_obs, reward, term, trunc, _ = env.step(action.item())
        done = term or trunc
        values.append(value)
        log_probs.append(dist.log_prob(action))
        rewards.append(float(reward))
        dones.append(done)
        entropies.append(dist.entropy())
        obs = next_obs if not done else env.reset()[0]
        if done:
            hidden = model.init_hidden()

    with torch.no_grad():
        _, last_val, _ = model(obs_to_tensor(obs), hidden)

    returns = torch.tensor(
        compute_returns(rewards, dones, last_val.item(), cfg.gamma)
    )
    return values, log_probs, returns, entropies


def test_gradients_finite():
    cfg = Config()
    model = make_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    env = make_env()

    values, log_probs, returns, entropies = _collect_mini_rollout(model, env)
    vt = torch.stack(values).squeeze(-1).squeeze(-1)
    lpt = torch.stack(log_probs)
    adv = (returns - vt.detach())
    loss = -(lpt * adv).mean() + F.mse_loss(vt, returns) - 0.01 * torch.stack(entropies).mean()

    optimizer.zero_grad()
    loss.backward()

    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "Non-finite gradient detected"
    print("  ✓ gradients remain finite")


def test_losses_finite():
    cfg = Config()
    model = make_model(cfg)
    env = make_env()

    values, _, returns, entropies = _collect_mini_rollout(model, env)
    vt = torch.stack(values).squeeze(-1).squeeze(-1)
    loss = F.mse_loss(vt, returns) - 0.01 * torch.stack(entropies).mean()
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    print("  ✓ losses remain finite")


def test_parameters_update():
    cfg = Config()
    model = make_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    params_before = [p.clone().detach() for p in model.parameters()]
    env = make_env()

    values, log_probs, returns, entropies = _collect_mini_rollout(model, env)
    vt = torch.stack(values).squeeze(-1).squeeze(-1)
    lpt = torch.stack(log_probs)
    adv = returns - vt.detach()
    loss = -(lpt * adv).mean() + F.mse_loss(vt, returns) - 0.01 * torch.stack(entropies).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    changed = any(
        not torch.allclose(pb, pa)
        for pb, pa in zip(params_before, model.parameters())
    )
    assert changed, "Parameters did not change after optimizer step"
    print("  ✓ parameters update during training")


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_all_tests():
    sections = [
        ("Environment", [
            test_maze_resets_correctly,
            test_correct_alternation_reward,
            test_no_reward_for_repeat,
            test_episode_continues_after_error,
            test_repeated_traversal_loop,
        ]),
        ("Model", [
            test_forward_pass_shapes,
            test_hidden_state_updates,
            test_hidden_state_reset_between_episodes,
            test_actor_output_size,
            test_critic_outputs_scalar,
        ]),
        ("Training health", [
            test_gradients_finite,
            test_losses_finite,
            test_parameters_update,
        ]),
    ]

    passed = failed = 0
    for section_name, tests in sections:
        print(f"\n{section_name} tests")
        print("-" * 40)
        for fn in tests:
            try:
                fn()
                passed += 1
            except Exception as e:
                print(f"  ✗ {fn.__name__}: {e}")
                failed += 1

    print("\n" + "=" * 40)
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 40)
    return failed == 0


if __name__ == "__main__":
    run_all_tests()
