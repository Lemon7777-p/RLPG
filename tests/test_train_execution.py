from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rl_benchmark.logging.aggregate import load_all_metrics, load_run_index
from rl_benchmark.logging.schema import read_manifest
from rl_benchmark.runners import run_training_job


class LoggedTrainingExecutionTests(unittest.TestCase):
    def test_reinforce_training_job_persists_manifest_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_training_job(
                "reinforce",
                "CartPole-v1",
                7,
                device="cpu",
                train_steps=80,
                eval_episodes=2,
                results_root=temp_dir,
            )

            self.assertEqual(result.manifest.status, "completed")
            self.assertGreater(result.manifest.total_steps, 0)
            self.assertGreater(len(result.metrics), 0)

            index_df = load_run_index(temp_dir)
            metrics_df = load_all_metrics(temp_dir)
            self.assertEqual(index_df.iloc[0]["algorithm_name"], "reinforce")
            self.assertTrue((metrics_df["run_id"] == result.run_id).any())

    def test_actor_critic_training_jobs_persist_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            a2c_result = run_training_job(
                "a2c",
                "CartPole-v1",
                11,
                device="cpu",
                train_steps=32,
                eval_episodes=2,
                results_root=temp_dir,
            )
            ppo_result = run_training_job(
                "ppo",
                "CartPole-v1",
                23,
                device="cpu",
                train_steps=48,
                eval_episodes=2,
                results_root=temp_dir,
            )

            self.assertEqual(a2c_result.manifest.status, "completed")
            self.assertEqual(ppo_result.manifest.status, "completed")
            self.assertLessEqual(a2c_result.manifest.total_steps, 32)
            self.assertLessEqual(ppo_result.manifest.total_steps, 48)

            metrics_df = load_all_metrics(temp_dir)
            self.assertIn("a2c", set(metrics_df["algorithm_name"]))
            self.assertIn("ppo", set(metrics_df["algorithm_name"]))
            self.assertTrue(metrics_df.loc[metrics_df["algorithm_name"] == "ppo", "approx_kl"].notna().any())

    def test_training_job_can_resume_from_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first = run_training_job(
                "a2c",
                "CartPole-v1",
                31,
                device="cpu",
                train_steps=16,
                eval_episodes=1,
                results_root=temp_dir,
                checkpoint_interval_steps=8,
            )
            self.assertIsNotNone(first.latest_checkpoint)

            resumed = run_training_job(
                "a2c",
                "CartPole-v1",
                31,
                device="cpu",
                train_steps=32,
                eval_episodes=1,
                results_root=temp_dir,
                checkpoint_interval_steps=8,
                resume=True,
            )

            self.assertEqual(resumed.manifest.status, "completed")
            self.assertLessEqual(first.manifest.total_steps, resumed.manifest.total_steps)
            self.assertGreater(resumed.manifest.total_updates, first.manifest.total_updates)
            self.assertGreaterEqual(resumed.manifest.checkpoint_count, 2)
            self.assertIsNotNone(resumed.manifest.resumed_from)

            manifest = read_manifest(resumed.output_dir / "manifest.json")
            self.assertEqual(manifest.status, "completed")
            self.assertIsNotNone(manifest.latest_checkpoint)
            metrics_df = load_all_metrics(temp_dir)
            resumed_rows = metrics_df.loc[metrics_df["run_id"] == resumed.run_id]
            self.assertEqual(len(resumed_rows), resumed.manifest.total_updates)

    def test_training_job_marks_manifest_failed_when_update_crashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(RuntimeError):
                with patch("rl_benchmark.runners.train._collect_batch", side_effect=RuntimeError("boom")):
                    run_training_job(
                        "a2c",
                        "CartPole-v1",
                        17,
                        device="cpu",
                        train_steps=16,
                        eval_episodes=1,
                        results_root=temp_dir,
                    )

            manifest = read_manifest(Path(temp_dir) / "a2c_CartPole-v1_seed17" / "manifest.json")
            self.assertEqual(manifest.status, "failed")
            self.assertEqual(manifest.failure_message, "RuntimeError: boom")

    def test_training_job_marks_manifest_failed_when_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(KeyboardInterrupt):
                with patch("rl_benchmark.runners.train._collect_batch", side_effect=KeyboardInterrupt()):
                    run_training_job(
                        "a2c",
                        "CartPole-v1",
                        18,
                        device="cpu",
                        train_steps=16,
                        eval_episodes=1,
                        results_root=temp_dir,
                    )

            manifest = read_manifest(Path(temp_dir) / "a2c_CartPole-v1_seed18" / "manifest.json")
            self.assertEqual(manifest.status, "failed")
            self.assertEqual(manifest.failure_message, "KeyboardInterrupt: ")

    def test_training_job_can_emit_progress_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("builtins.print") as mocked_print:
                result = run_training_job(
                    "a2c",
                    "CartPole-v1",
                    19,
                    device="cpu",
                    train_steps=16,
                    eval_episodes=1,
                    results_root=temp_dir,
                    checkpoint_interval_steps=8,
                    log_progress=True,
                    progress_interval_steps=8,
                )

            self.assertEqual(result.manifest.status, "completed")
            progress_messages = [
                call.args[0]
                for call in mocked_print.call_args_list
                if call.args and isinstance(call.args[0], str) and call.args[0].startswith("Progress ")
            ]
            self.assertGreaterEqual(len(progress_messages), 2)


if __name__ == "__main__":
    unittest.main()