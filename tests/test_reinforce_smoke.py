from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from rl_benchmark.algorithms import EpisodeBatch, ReinforceAlgorithm
from rl_benchmark.runners import prepare_run_context


def collect_episode(algorithm: ReinforceAlgorithm, env) -> EpisodeBatch:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []

    observation, _ = env.reset()
    done = False
    while not done:
        policy_output = algorithm.act(observation, deterministic=False)
        action = int(policy_output.actions.squeeze(0).cpu().item())
        next_observation, reward, terminated, truncated, _ = env.step(action)

        observations.append(np.asarray(observation, dtype=np.float32))
        actions.append(action)
        rewards.append(float(reward))

        observation = next_observation
        done = terminated or truncated

    return EpisodeBatch(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
    )


class ReinforceSmokeTests(unittest.TestCase):
    def test_discounted_returns_shape_matches_rewards(self) -> None:
        context = prepare_run_context("reinforce", "CartPole-v1", seed=7)
        algorithm = ReinforceAlgorithm(
            config=context.config,
            observation_dim=context.observation_dim,
            action_dim=context.action_dim,
            device="cpu",
        )

        returns = algorithm.compute_discounted_returns(np.array([1.0, 1.0, 1.0], dtype=np.float32))

        self.assertEqual(tuple(returns.shape), (3,))
        context.close()

    def test_single_cartpole_episode_update_produces_finite_metrics(self) -> None:
        context = prepare_run_context("reinforce", "CartPole-v1", seed=11)
        algorithm = ReinforceAlgorithm(
            config=context.config,
            observation_dim=context.observation_dim,
            action_dim=context.action_dim,
            device="cpu",
        )

        before = next(algorithm.policy.parameters()).detach().clone()
        batch = collect_episode(algorithm, context.train_env.env)
        metrics = algorithm.update(batch)
        after = next(algorithm.policy.parameters()).detach().clone()

        self.assertGreater(batch.rewards.shape[0], 0)
        self.assertFalse(torch.allclose(before, after))
        for key in ["loss", "policy_loss", "entropy", "grad_norm", "episode_return", "episode_length"]:
            self.assertIn(key, metrics)
            self.assertTrue(math.isfinite(metrics[key]), f"Metric {key} is not finite")

        context.close()


if __name__ == "__main__":
    unittest.main()