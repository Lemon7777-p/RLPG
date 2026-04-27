from __future__ import annotations

import math
import unittest
from copy import deepcopy

import numpy as np
import torch

from rl_benchmark.algorithms import PPOAlgorithm, PPOBatch
from rl_benchmark.runners import prepare_run_context


def collect_rollout(algorithm: PPOAlgorithm, env) -> PPOBatch:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[float] = []
    log_probs: list[float] = []
    values: list[float] = []

    observation, _ = env.reset()
    next_observation = observation
    for _ in range(algorithm.rollout_steps):
        output = algorithm.act(observation, deterministic=False)
        action = int(output.policy.actions.squeeze(0).cpu().item())
        log_prob = float(output.policy.log_probs.squeeze(0).detach().cpu().item())
        value = float(output.values.squeeze(0).detach().cpu().item())
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        observations.append(np.asarray(observation, dtype=np.float32))
        actions.append(action)
        rewards.append(float(reward))
        dones.append(float(done))
        log_probs.append(log_prob)
        values.append(value)

        observation = next_observation
        if done:
            break

    return PPOBatch(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        log_probs=np.asarray(log_probs, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
        next_observation=np.asarray(next_observation, dtype=np.float32),
    )


class PPOSmokeTests(unittest.TestCase):
    def test_advantage_and_return_shapes_match_rollout(self) -> None:
        context = prepare_run_context("ppo", "CartPole-v1", seed=7)
        config = deepcopy(context.config)
        config["algorithm"]["rollout_steps"] = 8
        config["algorithm"]["minibatch_size"] = 4
        algorithm = PPOAlgorithm(
            config=config,
            observation_dim=context.observation_dim,
            action_dim=context.action_dim,
            device="cpu",
        )

        advantages, returns = algorithm.compute_advantages_and_returns(
            rewards=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dones=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            values=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            next_value=torch.tensor(0.0),
        )

        self.assertEqual(tuple(advantages.shape), (3,))
        self.assertEqual(tuple(returns.shape), (3,))
        context.close()

    def test_single_cartpole_rollout_update_produces_finite_metrics(self) -> None:
        context = prepare_run_context("ppo", "CartPole-v1", seed=47)
        config = deepcopy(context.config)
        config["algorithm"]["rollout_steps"] = 16
        config["algorithm"]["minibatch_size"] = 8
        config["algorithm"]["update_epochs"] = 2
        algorithm = PPOAlgorithm(
            config=config,
            observation_dim=context.observation_dim,
            action_dim=context.action_dim,
            device="cpu",
        )

        before = next(algorithm.actor_critic.parameters()).detach().clone()
        batch = collect_rollout(algorithm, context.train_env.env)
        metrics = algorithm.update(batch)
        after = next(algorithm.actor_critic.parameters()).detach().clone()

        self.assertGreater(batch.rewards.shape[0], 0)
        self.assertFalse(torch.allclose(before, after))
        for key in ["policy_loss", "value_loss", "entropy", "grad_norm", "approx_kl", "rollout_return", "rollout_length"]:
            self.assertIn(key, metrics)
            self.assertTrue(math.isfinite(metrics[key]), f"Metric {key} is not finite")

        context.close()


if __name__ == "__main__":
    unittest.main()