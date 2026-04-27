from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rl_benchmark.dashboard.data import load_dashboard_data
from rl_benchmark.dashboard.jobs import BACKGROUND_LAUNCH_FILENAME, BACKGROUND_LOG_FILENAME
from rl_benchmark.dashboard.plots import (
    efficiency_figure,
    final_performance_figure,
    learning_curve_figure,
    stability_figure,
    wall_time_curve_figure,
)
from rl_benchmark.logging.demo import create_demo_results
from rl_benchmark.logging.schema import MetricRecord, RunManifest, write_manifest, write_metrics


class DashboardDataTests(unittest.TestCase):
    def test_demo_results_produce_dashboard_bundle_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            created = create_demo_results(temp_dir, overwrite=True)
            dashboard_data = load_dashboard_data(temp_dir)

            self.assertGreater(len(created), 0)
            self.assertTrue(dashboard_data.has_runs)
            self.assertFalse(dashboard_data.group_summary_df.empty)

            env_metrics = dashboard_data.metrics_df.loc[dashboard_data.metrics_df["env_id"] == "CartPole-v1"]
            env_summary = dashboard_data.run_summary_df.loc[dashboard_data.run_summary_df["env_id"] == "CartPole-v1"]
            env_group = dashboard_data.group_summary_df.loc[dashboard_data.group_summary_df["env_id"] == "CartPole-v1"]

            figures = [
                learning_curve_figure(env_metrics),
                wall_time_curve_figure(env_metrics),
                final_performance_figure(env_summary),
                efficiency_figure(env_group),
                stability_figure(env_group),
            ]
            for figure in figures:
                self.assertGreater(len(figure.data), 0)

    def test_running_manifest_and_checkpoint_metadata_appear_in_dashboard_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = RunManifest(
                run_id="live_run",
                algorithm_name="a2c",
                env_id="CartPole-v1",
                runtime_env_id="CartPole-v1",
                seed=3,
                status="running",
                total_steps=24,
                total_updates=6,
                latest_checkpoint="checkpoints/update_update_000006_step_00000024.pt",
                checkpoint_count=2,
            )
            metrics = [
                MetricRecord(step=8, update=1, wall_time_sec=0.2, train_episode_return=10.0),
                MetricRecord(step=24, update=6, wall_time_sec=1.1, train_episode_return=18.0, eval_episode_return=15.0),
            ]
            write_manifest(manifest, temp_dir)
            write_metrics(manifest.run_id, metrics, temp_dir)
            checkpoint_dir = Path(temp_dir) / manifest.run_id / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "update_update_000006_step_00000024.pt").write_text("stub", encoding="utf-8")
            run_dir = Path(temp_dir) / manifest.run_id
            (run_dir / BACKGROUND_LAUNCH_FILENAME).write_text(
                json.dumps(
                    {
                        "run_id": manifest.run_id,
                        "pid": 31337,
                        "command": ["python", "scripts/train_run.py", "--algorithm", "a2c"],
                        "log_path": str(run_dir / BACKGROUND_LOG_FILENAME),
                        "resume": False,
                        "launched_at": "2026-04-24T00:00:00+00:00",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (run_dir / BACKGROUND_LOG_FILENAME).write_text(
                "launching training\nstep 24\nwaiting for next rollout\n",
                encoding="utf-8",
            )

            dashboard_data = load_dashboard_data(temp_dir)

            self.assertTrue(dashboard_data.has_active_runs)
            self.assertEqual(int(dashboard_data.active_runs_df.iloc[0]["total_updates"]), 6)
            self.assertFalse(dashboard_data.run_checkpoint_df("live_run").empty)
            self.assertEqual(dashboard_data.run_manifest_row("live_run")["status"], "running")
            self.assertFalse(dashboard_data.active_background_runs_df.empty)
            self.assertEqual(int(dashboard_data.active_background_runs_df.iloc[0]["pid"]), 31337)
            self.assertEqual(dashboard_data.run_background_row("live_run")["launched_at"], "2026-04-24T00:00:00+00:00")
            self.assertIn("step 24", dashboard_data.run_background_log_tail("live_run") or "")


if __name__ == "__main__":
    unittest.main()
