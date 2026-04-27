from __future__ import annotations

import unittest

from rl_benchmark.config import (
    list_algorithms,
    list_environments,
    load_algorithm_config,
    load_defaults,
    load_environment_config,
    load_run_config,
)


class ConfigLoadingTests(unittest.TestCase):
    def test_defaults_include_main_environments(self) -> None:
        defaults = load_defaults()

        self.assertIn("experiment", defaults)
        self.assertEqual(defaults["experiment"]["main_envs"], ["Acrobot-v1", "LunarLander-v2"])

    def test_algorithm_config_can_be_loaded(self) -> None:
        ppo_config = load_algorithm_config("PPO")

        self.assertEqual(ppo_config["algorithm"]["name"], "ppo")
        self.assertEqual(ppo_config["algorithm"]["clip_coef"], 0.2)

    def test_environment_config_tracks_runtime_alias(self) -> None:
        lunar_lander_config = load_environment_config("LunarLander-v2")

        self.assertEqual(lunar_lander_config["runtime_env_id"], "LunarLander-v3")
        self.assertEqual(lunar_lander_config["action_space"], "discrete")

    def test_run_config_merges_defaults_algorithm_and_environment(self) -> None:
        run_config = load_run_config("a2c", "Acrobot-v1")

        self.assertEqual(run_config["algorithm"]["name"], "a2c")
        self.assertEqual(run_config["environment"]["role"], "benchmark")
        self.assertEqual(run_config["run"]["runtime_env_id"], "Acrobot-v1")
        self.assertEqual(run_config["project"]["seed_list"], [7, 11, 23, 47, 89])

    def test_available_names_are_discoverable(self) -> None:
        self.assertEqual(list_algorithms(), ["a2c", "ppo", "reinforce"])
        self.assertEqual(list_environments(), ["Acrobot-v1", "CartPole-v1", "LunarLander-v2"])


if __name__ == "__main__":
    unittest.main()
