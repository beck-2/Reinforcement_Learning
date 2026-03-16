"""
tests_ppo.py — Unit tests for recurrent PPO implementation.

Run: .venv/bin/python3 tests_ppo.py

Categories:
  1. Observation encoding
  2. Model architecture (shapes, init, hidden state)
  3. GAE computation (correctness, boundary handling)
  4. PPO loss (clip, value clip, entropy)
  5. Integration (rollout collection, ppo_update does not crash)
"""
import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from config import PPOConfig
from model import RecurrentActorCritic, ActorNet, CriticNet
from ppo import compute_gae, encode_obs, collect_rollout, ppo_update, make_env


# ---------------------------------------------------------------------------
# 1. Observation encoding
# ---------------------------------------------------------------------------

class TestObsEncoding(unittest.TestCase):

    def _make_obs(self, x=7, y=12, direction=0):
        return {
            "position_vector": np.array([x, y], dtype=np.float32),
            "direction": direction,
        }

    def test_output_shape(self):
        obs = encode_obs(self._make_obs())
        self.assertEqual(obs.shape, (6,))

    def test_position_normalized(self):
        obs = encode_obs(self._make_obs(x=14, y=0))
        self.assertAlmostEqual(obs[0].item(), 1.0, places=5)
        self.assertAlmostEqual(obs[1].item(), 0.0, places=5)

    def test_direction_onehot(self):
        for d in range(4):
            obs = encode_obs(self._make_obs(direction=d))
            dir_part = obs[2:].numpy()
            self.assertEqual(dir_part.argmax(), d)
            self.assertAlmostEqual(dir_part.sum(), 1.0, places=5)

    def test_dtype(self):
        obs = encode_obs(self._make_obs())
        self.assertEqual(obs.dtype, torch.float32)


# ---------------------------------------------------------------------------
# 2. Model architecture
# ---------------------------------------------------------------------------

class TestModelArchitecture(unittest.TestCase):

    def setUp(self):
        self.cfg = PPOConfig()
        self.model = RecurrentActorCritic(
            self.cfg.obs_dim, self.cfg.hidden_size, self.cfg.num_actions
        )

    def test_actor_output_shape(self):
        obs = torch.zeros(5, self.cfg.obs_dim)
        h = self.model.actor.init_hidden()
        logits, new_h = self.model.actor(obs, h)
        self.assertEqual(logits.shape, (5, self.cfg.num_actions))
        self.assertEqual(new_h[0].shape, (1, 1, self.cfg.hidden_size))

    def test_critic_output_shape(self):
        obs = torch.zeros(5, self.cfg.obs_dim)
        h = self.model.critic.init_hidden()
        values, new_h = self.model.critic(obs, h)
        self.assertEqual(values.shape, (5,))

    def test_joint_forward_shapes(self):
        obs = torch.zeros(3, self.cfg.obs_dim)
        h = self.model.init_hidden()
        logits, values, new_h = self.model(obs, h)
        self.assertEqual(logits.shape, (3, self.cfg.num_actions))
        self.assertEqual(values.shape, (3,))

    def test_forward_bias_initialized(self):
        # head.bias[2] > head.bias[0] and head.bias[2] > head.bias[1]
        bias = self.model.actor.head.bias.detach()
        self.assertGreater(bias[2].item(), bias[0].item())
        self.assertGreater(bias[2].item(), bias[1].item())

    def test_hidden_detach(self):
        h = self.model.init_hidden()
        dh = RecurrentActorCritic.detach_hidden(h)
        (h_a, c_a), (h_c, c_c) = dh
        self.assertFalse(h_a.requires_grad)
        self.assertFalse(c_a.requires_grad)

    def test_parameter_count(self):
        n = sum(p.numel() for p in self.model.parameters())
        # hidden=128: should be well over 50k parameters
        self.assertGreater(n, 50_000)

    def test_single_step_forward(self):
        obs = torch.zeros(1, self.cfg.obs_dim)
        h = self.model.init_hidden()
        logits, values, new_h = self.model(obs, h)
        self.assertEqual(logits.shape, (1, self.cfg.num_actions))
        self.assertEqual(values.shape, (1,))


# ---------------------------------------------------------------------------
# 3. GAE computation
# ---------------------------------------------------------------------------

class TestComputeGAE(unittest.TestCase):

    def test_no_discount_single_step(self):
        """With γ=1, λ=0: advantage = r + V(s') - V(s)."""
        rewards = torch.tensor([1.0])
        values = torch.tensor([0.5])
        next_value = torch.tensor(0.0)
        dones = torch.tensor([True])
        adv, ret = compute_gae(rewards, values, next_value, dones, gamma=1.0, lam=0.0)
        # done=True → mask=0 → delta = r + 0 - V = 1.0 - 0.5 = 0.5
        # returns = adv + V = 0.5 + 0.5 = 1.0  (no bootstrap since done)
        self.assertAlmostEqual(adv[0].item(), 0.5, places=5)
        self.assertAlmostEqual(ret[0].item(), 1.0, places=5)

    def test_episode_boundary_zeroes_bootstrap(self):
        """done=True should set bootstrap to 0, not next_value."""
        rewards = torch.tensor([1.0, 0.0])
        values = torch.tensor([0.0, 0.0])
        next_value = torch.tensor(100.0)   # should be ignored for step 0 (done)
        dones = torch.tensor([True, False])
        adv, _ = compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95)
        # step 0: done → mask=0 → delta = 1.0 + 0 - 0 = 1.0, gae=1.0
        self.assertAlmostEqual(adv[0].item(), 1.0, places=3)

    def test_returns_equal_advantages_plus_values(self):
        T = 10
        rewards = torch.rand(T)
        values = torch.rand(T)
        next_value = torch.tensor(0.5)
        dones = torch.zeros(T, dtype=torch.bool)
        adv, ret = compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95)
        diff = (ret - adv - values).abs().max().item()
        self.assertLess(diff, 1e-4)

    def test_shapes(self):
        T = 50
        rewards = torch.randn(T)
        values = torch.randn(T)
        next_value = torch.tensor(0.0)
        dones = torch.zeros(T, dtype=torch.bool)
        adv, ret = compute_gae(rewards, values, next_value, dones, 0.99, 0.95)
        self.assertEqual(adv.shape, (T,))
        self.assertEqual(ret.shape, (T,))

    def test_terminal_advantage_ignores_next_value(self):
        """Last step with done=True: advantage should not depend on next_value."""
        rewards = torch.tensor([1.0])
        values = torch.tensor([0.0])
        dones = torch.tensor([True])

        adv1, _ = compute_gae(rewards, values, torch.tensor(999.0), dones, 0.99, 0.95)
        adv2, _ = compute_gae(rewards, values, torch.tensor(0.0), dones, 0.99, 0.95)
        self.assertAlmostEqual(adv1[0].item(), adv2[0].item(), places=5)


# ---------------------------------------------------------------------------
# 4. PPO loss correctness
# ---------------------------------------------------------------------------

class TestPPOLoss(unittest.TestCase):

    def _make_fake_batch(self, T=8):
        """Return logits, old_lp, advantages, values, returns for a fake batch."""
        torch.manual_seed(0)
        logits = torch.randn(T, 3)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        old_lp = dist.log_prob(actions)
        advantages = torch.randn(T)
        values = torch.randn(T)
        returns = values + advantages
        return logits, actions, old_lp, advantages, values, returns

    def test_clip_ratio_within_bounds(self):
        """Ratio clipping should keep effective gradient within [1-ε, 1+ε]."""
        eps = 0.2
        logits, actions, old_lp, advantages, values, returns = self._make_fake_batch()
        logits2 = logits + 5.0   # deliberately different policy
        dist2 = Categorical(logits=logits2)
        new_lp = dist2.log_prob(actions)
        ratio = torch.exp(new_lp - old_lp)
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
        surr1 = ratio * advantages
        surr2 = clipped * advantages
        loss = -torch.min(surr1, surr2).mean()
        # Just ensure it's a finite scalar
        self.assertTrue(torch.isfinite(loss))

    def test_entropy_is_positive(self):
        logits = torch.zeros(4, 3)   # uniform policy
        dist = Categorical(logits=logits)
        self.assertGreater(dist.entropy().mean().item(), 0.0)

    def test_zero_advantage_zero_gradient(self):
        """With zero advantages the policy loss gradient w.r.t. logits is zero."""
        logits = torch.randn(4, 3, requires_grad=True)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        old_lp = dist.log_prob(actions).detach()

        advantages = torch.zeros(4)
        dist2 = Categorical(logits=logits)
        new_lp = dist2.log_prob(actions)
        ratio = torch.exp(new_lp - old_lp)
        loss = -torch.min(ratio * advantages,
                          torch.clamp(ratio, 0.8, 1.2) * advantages).mean()
        loss.backward()
        grad_norm = logits.grad.norm().item()
        self.assertAlmostEqual(grad_norm, 0.0, places=5)


# ---------------------------------------------------------------------------
# 5. Integration tests
# ---------------------------------------------------------------------------

class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.cfg = PPOConfig()
        self.cfg.rollout_length = 64   # fast
        self.cfg.ppo_epochs = 2
        self.cfg.num_train_steps = 128
        self.device = torch.device("cpu")
        self.model = RecurrentActorCritic(
            self.cfg.obs_dim, self.cfg.hidden_size, self.cfg.num_actions
        )

    def test_collect_rollout_returns_correct_shapes(self):
        env = make_env(self.cfg, stage=1)
        hidden = self.model.init_hidden()
        buf, new_hidden, stats = collect_rollout(env, self.model, hidden, self.cfg, self.device)
        T = self.cfg.rollout_length
        self.assertEqual(buf.obs.shape, (T, self.cfg.obs_dim))
        self.assertEqual(buf.actions.shape, (T,))
        self.assertEqual(buf.advantages.shape, (T,))
        env.close()

    def test_ppo_update_runs_without_error(self):
        env = make_env(self.cfg, stage=1)
        hidden = self.model.init_hidden()
        buf, _, _ = collect_rollout(env, self.model, hidden, self.cfg, self.device)
        actor_opt = torch.optim.Adam(self.model.actor.parameters(), lr=3e-4)
        critic_opt = torch.optim.Adam(self.model.critic.parameters(), lr=3e-5)
        stats = ppo_update(self.model, buf, actor_opt, critic_opt, self.cfg, self.device)
        self.assertIn("policy_loss", stats)
        self.assertTrue(np.isfinite(stats["policy_loss"]))
        env.close()

    def test_model_parameters_change_after_update(self):
        env = make_env(self.cfg, stage=1)
        hidden = self.model.init_hidden()
        buf, _, _ = collect_rollout(env, self.model, hidden, self.cfg, self.device)

        before = [p.clone().detach() for p in self.model.parameters()]

        actor_opt = torch.optim.Adam(self.model.actor.parameters(), lr=3e-4)
        critic_opt = torch.optim.Adam(self.model.critic.parameters(), lr=3e-5)
        ppo_update(self.model, buf, actor_opt, critic_opt, self.cfg, self.device)

        after = [p.clone().detach() for p in self.model.parameters()]
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        self.assertTrue(changed, "No parameters changed after PPO update")
        env.close()

    def test_stage_configs_all_create_valid_envs(self):
        for stage in [1, 2, 3]:
            env = make_env(self.cfg, stage)
            obs, _ = env.reset()
            self.assertIn("position_vector", obs)
            self.assertIn("direction", obs)
            env.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
