from __future__ import annotations

import unittest

import numpy as np

from rl_benchmark.envs.factory import (
    build_env,
    make_env,
    make_train_and_eval_envs,
    resolve_env_id,
)
from rl_benchmark.utils.seeding import set_global_seed


class SeedingTests(unittest.TestCase):
    def test_set_global_seed_repeats_numpy_sequence(self) -> None:
        set_global_seed(123)
        first = np.random.rand(5)

        set_global_seed(123)
        second = np.random.rand(5)

        np.testing.assert_allclose(first, second)


class EnvironmentFactoryTests(unittest.TestCase):
    def test_cartpole_initial_reset_is_reproducible(self) -> None:
        first = make_env("CartPole-v1", seed=123)
        second = make_env("CartPole-v1", seed=123)

        np.testing.assert_allclose(first.observation, second.observation)
        self.assertEqual(first.info, second.info)

        first.env.close()
        second.env.close()

    def test_train_and_eval_envs_use_different_initial_states(self) -> None:
        train_bundle, eval_bundle = make_train_and_eval_envs("CartPole-v1", seed=123)

        self.assertFalse(np.allclose(train_bundle.observation, eval_bundle.observation))

        train_bundle.env.close()
        eval_bundle.env.close()

    def test_requested_lunar_lander_id_is_resolved_explicitly(self) -> None:
        self.assertEqual(resolve_env_id("LunarLander-v2"), "LunarLander-v3")

    def test_lunar_lander_discrete_variant_can_be_created(self) -> None:
        bundle = make_env("LunarLander-v2", seed=123)

        self.assertEqual(bundle.requested_env_id, "LunarLander-v2")
        self.assertEqual(bundle.resolved_env_id, "LunarLander-v3")
        self.assertEqual(bundle.env.action_space.n, 4)
        bundle.env.close()


if __name__ == "__main__":
    unittest.main()
