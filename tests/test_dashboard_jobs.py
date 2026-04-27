from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rl_benchmark.dashboard.jobs import load_background_run_request, launch_background_training_job
from rl_benchmark.logging.schema import RunManifest, write_manifest


class DashboardJobLaunchTests(unittest.TestCase):
    def test_background_launcher_spawns_train_script_and_persists_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            process = MagicMock()
            process.pid = 4242

            with patch("rl_benchmark.dashboard.jobs.subprocess.Popen", return_value=process) as popen:
                launch = launch_background_training_job(
                    algorithm_name="a2c",
                    env_id="CartPole-v1",
                    seed=7,
                    device="cpu",
                    train_steps=32,
                    eval_episodes=2,
                    results_root=temp_dir,
                    notes="background launch",
                    checkpoint_interval_steps=8,
                    resume=False,
                )

            self.assertEqual(launch.run_id, "a2c_CartPole-v1_seed7")
            self.assertTrue(launch.log_path.is_file())
            self.assertTrue(launch.launch_metadata_path.is_file())
            metadata = json.loads(launch.launch_metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["pid"], 4242)
            self.assertIn("launched_at", metadata)
            self.assertEqual(metadata["request"]["train_steps"], 32)
            self.assertEqual(metadata["request"]["eval_episodes"], 2)
            self.assertIn("train_run.py", metadata["command"][1])
            self.assertIn("--train-steps", metadata["command"])
            self.assertEqual(popen.call_count, 1)

    def test_background_launcher_rejects_duplicate_running_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = RunManifest(
                run_id="a2c_CartPole-v1_seed7",
                algorithm_name="a2c",
                env_id="CartPole-v1",
                runtime_env_id="CartPole-v1",
                seed=7,
                status="running",
            )
            write_manifest(manifest, temp_dir)

            with self.assertRaises(RuntimeError):
                launch_background_training_job(
                    algorithm_name="a2c",
                    env_id="CartPole-v1",
                    seed=7,
                    device="cpu",
                    train_steps=32,
                    eval_episodes=2,
                    results_root=temp_dir,
                    notes="background launch",
                    checkpoint_interval_steps=8,
                    resume=False,
                )

    def test_background_request_loader_uses_manifest_fallback_without_launch_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = RunManifest(
                run_id="a2c_CartPole-v1_seed7",
                algorithm_name="a2c",
                env_id="CartPole-v1",
                runtime_env_id="CartPole-v1",
                seed=7,
                status="failed",
                total_steps=12,
                notes="fallback request",
                config_snapshot={
                    "environment": {"train_steps": 32},
                    "evaluation": {"episodes": 3},
                    "runtime": {"device": "cpu", "checkpoint_interval_steps": 8},
                },
            )
            write_manifest(manifest, temp_dir)

            request = load_background_run_request(Path(temp_dir) / manifest.run_id, resume=True)

            self.assertIsNotNone(request)
            assert request is not None
            self.assertEqual(request.algorithm_name, "a2c")
            self.assertEqual(request.train_steps, 32)
            self.assertEqual(request.eval_episodes, 3)
            self.assertEqual(request.device, "cpu")
            self.assertEqual(request.checkpoint_interval_steps, 8)
            self.assertTrue(request.resume)

    def test_background_request_loader_supports_legacy_launch_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "ppo_CartPole-v1_seed9"
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest = RunManifest(
                run_id=run_dir.name,
                algorithm_name="ppo",
                env_id="CartPole-v1",
                runtime_env_id="CartPole-v1",
                seed=9,
                status="failed",
                total_steps=40,
                notes="legacy launch",
                config_snapshot={
                    "environment": {"train_steps": 48},
                    "evaluation": {"episodes": 2},
                    "runtime": {"device": "auto", "checkpoint_interval_steps": 16},
                },
            )
            write_manifest(manifest, temp_dir)
            (run_dir / "background_launch.json").write_text(
                json.dumps(
                    {
                        "run_id": run_dir.name,
                        "pid": 9001,
                        "command": [
                            "python",
                            "scripts/train_run.py",
                            "--algorithm",
                            "ppo",
                            "--env",
                            "CartPole-v1",
                            "--seed",
                            "9",
                            "--device",
                            "cpu",
                            "--train-steps",
                            "64",
                            "--eval-episodes",
                            "4",
                            "--results-root",
                            temp_dir,
                            "--notes",
                            "legacy launch",
                            "--checkpoint-interval-steps",
                            "12",
                            "--resume",
                        ],
                        "log_path": str(run_dir / "background_train.log"),
                        "resume": True,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            request = load_background_run_request(run_dir)

            self.assertIsNotNone(request)
            assert request is not None
            self.assertEqual(request.device, "cpu")
            self.assertEqual(request.train_steps, 64)
            self.assertEqual(request.eval_episodes, 4)
            self.assertEqual(request.checkpoint_interval_steps, 12)
            self.assertTrue(request.resume)


if __name__ == "__main__":
    unittest.main()