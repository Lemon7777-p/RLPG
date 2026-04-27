from __future__ import annotations

import unittest

from rl_benchmark.runners import build_run_id, prepare_run_context


class TrainRunnerTests(unittest.TestCase):
    def test_build_run_id_is_stable(self) -> None:
        self.assertEqual(build_run_id("reinforce", "CartPole-v1", 7), "reinforce_CartPole-v1_seed7")

    def test_prepare_run_context_for_cartpole(self) -> None:
        context = prepare_run_context("reinforce", "CartPole-v1", seed=7)

        self.assertEqual(context.algorithm_name, "reinforce")
        self.assertEqual(context.env_id, "CartPole-v1")
        self.assertEqual(context.observation_dim, 4)
        self.assertEqual(context.action_dim, 2)
        self.assertTrue(context.output_dir.is_dir())
        self.assertEqual(context.train_env.requested_env_id, "CartPole-v1")
        self.assertEqual(context.eval_env.requested_env_id, "CartPole-v1")

        context.close()

    def test_prepare_run_context_tracks_runtime_env_alias(self) -> None:
        context = prepare_run_context("ppo", "LunarLander-v2", seed=11)

        self.assertEqual(context.config["run"]["runtime_env_id"], "LunarLander-v3")
        self.assertEqual(context.train_env.resolved_env_id, "LunarLander-v3")
        self.assertEqual(context.action_dim, 4)

        context.close()


if __name__ == "__main__":
    unittest.main()