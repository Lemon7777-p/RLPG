from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkScriptTests(unittest.TestCase):
    def test_run_benchmark_script_can_export_results_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir, tempfile.TemporaryDirectory() as export_dir:
            env = os.environ.copy()
            env["CV_SHOW"] = "1"
            env["PYTHONPATH"] = str(ROOT / "src")

            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_benchmark.py"),
                "--algorithms",
                "a2c",
                "ppo",
                "--envs",
                "CartPole-v1",
                "--seeds",
                "7",
                "--device",
                "cpu",
                "--train-steps",
                "32",
                "--eval-episodes",
                "1",
                "--results-root",
                results_dir,
                "--export-root",
                export_dir,
                "--figure-format",
                "html",
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertIn("Completed a2c_CartPole-v1_seed7", completed.stdout)
            self.assertIn("Completed ppo_CartPole-v1_seed7", completed.stdout)
            self.assertIn("Exported", completed.stdout)
            self.assertTrue((Path(export_dir) / "export_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()