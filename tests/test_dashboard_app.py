from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from rl_benchmark.logging.demo import create_demo_results


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "src" / "rl_benchmark" / "dashboard" / "app.py"


class DashboardAppTests(unittest.TestCase):
    def test_streamlit_dashboard_renders_with_demo_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            create_demo_results(temp_dir, overwrite=True)
            previous_root = os.environ.get("RL_BENCHMARK_RESULTS_ROOT")
            os.environ["RL_BENCHMARK_RESULTS_ROOT"] = temp_dir
            try:
                app = AppTest.from_file(str(APP_PATH))
                app.run(timeout=30)
            finally:
                if previous_root is None:
                    os.environ.pop("RL_BENCHMARK_RESULTS_ROOT", None)
                else:
                    os.environ["RL_BENCHMARK_RESULTS_ROOT"] = previous_root

            self.assertFalse(app.exception)
            self.assertGreaterEqual(len(app.radio), 1)


if __name__ == "__main__":
    unittest.main()