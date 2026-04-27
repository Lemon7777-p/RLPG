from __future__ import annotations

import tempfile
import unittest

from rl_benchmark.logging.aggregate import build_group_summary, build_run_summary, load_all_metrics, load_run_index
from rl_benchmark.logging.schema import MetricRecord, RunManifest, append_metrics, write_manifest, write_metrics


class ResultSchemaTests(unittest.TestCase):
    def test_manifest_and_metrics_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = RunManifest(
                run_id="demo_run",
                algorithm_name="ppo",
                env_id="CartPole-v1",
                runtime_env_id="CartPole-v1",
                seed=7,
                source="demo",
                total_steps=10000,
                total_updates=2,
            )
            metrics = [
                MetricRecord(step=0, update=1, wall_time_sec=0.1, train_episode_return=20.0, eval_episode_return=18.0),
                MetricRecord(step=1000, update=2, wall_time_sec=1.2, train_episode_return=40.0, eval_episode_return=35.0, approx_kl=0.01),
            ]

            write_manifest(manifest, temp_dir)
            write_metrics(manifest.run_id, metrics, temp_dir)

            index_df = load_run_index(temp_dir)
            metrics_df = load_all_metrics(temp_dir)
            run_summary_df = build_run_summary(index_df, metrics_df)
            group_summary_df = build_group_summary(run_summary_df)

            self.assertEqual(index_df.iloc[0]["run_id"], "demo_run")
            self.assertEqual(metrics_df.iloc[-1]["eval_episode_return"], 35.0)
            self.assertEqual(run_summary_df.iloc[0]["final_return"], 35.0)
            self.assertEqual(group_summary_df.iloc[0]["runs"], 1)

    def test_append_metrics_preserves_existing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = RunManifest(
                run_id="append_run",
                algorithm_name="a2c",
                env_id="CartPole-v1",
                runtime_env_id="CartPole-v1",
                seed=13,
            )
            initial_records = [
                MetricRecord(step=5, update=1, wall_time_sec=0.1, train_episode_return=1.0),
                MetricRecord(step=10, update=2, wall_time_sec=0.2, train_episode_return=2.0),
            ]
            appended_record = MetricRecord(step=15, update=3, wall_time_sec=0.3, train_episode_return=3.0)

            write_manifest(manifest, temp_dir)
            write_metrics(manifest.run_id, initial_records, temp_dir)
            append_metrics(manifest.run_id, [appended_record], temp_dir)

            metrics_df = load_all_metrics(temp_dir)
            append_rows = metrics_df.loc[metrics_df["run_id"] == manifest.run_id]
            self.assertEqual(len(append_rows), 3)
            self.assertEqual(float(append_rows.iloc[-1]["train_episode_return"]), 3.0)


if __name__ == "__main__":
    unittest.main()
