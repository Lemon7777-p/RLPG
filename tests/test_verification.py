from __future__ import annotations

import unittest

from rl_benchmark.verification import (
    DEFAULT_ALGORITHMS,
    DEFAULT_MAIN_ENVIRONMENTS,
    DEFAULT_SEEDS,
    build_smoke_verification_jobs,
)


class VerificationPlanningTests(unittest.TestCase):
    def test_default_smoke_plan_covers_main_env_algorithm_seed_matrix(self) -> None:
        jobs = build_smoke_verification_jobs()

        self.assertEqual(
            len(jobs),
            len(DEFAULT_ALGORITHMS) * len(DEFAULT_MAIN_ENVIRONMENTS) * len(DEFAULT_SEEDS),
        )
        self.assertEqual({job.env_id for job in jobs}, set(DEFAULT_MAIN_ENVIRONMENTS))
        self.assertEqual({job.algorithm_name for job in jobs}, set(DEFAULT_ALGORITHMS))
        self.assertEqual({job.seed for job in jobs}, set(DEFAULT_SEEDS))
        self.assertTrue(all(job.checkpoint_interval_steps >= 8 for job in jobs))

        reinforce_jobs = [job for job in jobs if job.algorithm_name == "reinforce"]
        actor_critic_jobs = [job for job in jobs if job.algorithm_name in {"a2c", "ppo"}]
        self.assertTrue(all(job.train_steps == 1 for job in reinforce_jobs))
        self.assertTrue(all(job.train_steps == 16 for job in actor_critic_jobs))

    def test_smoke_plan_respects_custom_step_overrides(self) -> None:
        jobs = build_smoke_verification_jobs(
            algorithms=["reinforce", "a2c"],
            envs=["Acrobot-v1"],
            seeds=[3],
            reinforce_train_steps=2,
            actor_critic_train_steps=24,
        )

        self.assertEqual(len(jobs), 2)
        train_steps_by_algorithm = {job.algorithm_name: job.train_steps for job in jobs}
        self.assertEqual(train_steps_by_algorithm["reinforce"], 2)
        self.assertEqual(train_steps_by_algorithm["a2c"], 24)


if __name__ == "__main__":
    unittest.main()