from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from rl_benchmark.algorithms import A2CAlgorithm, RolloutBatch
from rl_benchmark.runners import prepare_run_context


def collect_rollout(algorithm: A2CAlgorithm, env) -> RolloutBatch:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[float] = []

    observation, _ = env.reset()
    next_observation = observation
    for _ in range(algorithm.rollout_steps):
        policy_output = algorithm.act(observation, deterministic=False)
        action = int(policy_output.policy.actions.squeeze(0).cpu().item())
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        observations.append(np.asarray(observation, dtype=np.float32))
        actions.append(action)
        rewards.append(float(reward))
        dones.append(float(done))

        observation = next_observation
        if done:
            break

    return RolloutBatch(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        next_observation=np.asarray(next_observation, dtype=np.float32),
    )


class A2CSmokeTests(unittest.TestCase):
    def test_compute_returns_matches_rollout_length(self) -> None:
        context = prepare_run_context("a2c", "CartPole-v1", seed=7)
        algorithm = A2CAlgorithm(
            config=context.config,
            observation_dim=context.observation_dim,
            action_dim=context.action_dim,
            device="cpu",
        )

        returns = algorithm.compute_returns(
            rewards=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dones=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            bootstrap_value=torch.tensor(0.0),
        )

        self.assertEqual(tuple(returns.shape), (3,))
        context.close()

    def test_single_cartpole_rollout_update_produces_finite_metrics(self) -> None:
        context = prepare_run_context("a2c", "CartPole-v1", seed=23)
        algorithm = A2CAlgorithm(
            config=context.config,
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
        for key in ["loss", "policy_loss", "value_loss", "entropy", "grad_norm", "rollout_return", "rollout_length"]:
            self.assertIn(key, metrics)
            self.assertTrue(math.isfinite(metrics[key]), f"Metric {key} is not finite")

        context.close()


if __name__ == "__main__":
    unittest.main()