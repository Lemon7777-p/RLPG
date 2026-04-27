from __future__ import annotations

import tempfile
import unittest

from rl_benchmark.logging.demo import create_demo_results
from rl_benchmark.reporting import export_analysis_bundle


class ReportingExportTests(unittest.TestCase):
    def test_export_analysis_bundle_writes_tables_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir, tempfile.TemporaryDirectory() as export_dir:
            create_demo_results(results_dir, overwrite=True)
            result = export_analysis_bundle(
                results_root=results_dir,
                export_root=export_dir,
                environments=["CartPole-v1"],
                algorithms=["reinforce", "a2c", "ppo"],
                figure_format="html",
            )

            created_names = {path.name for path in result.created_files}
            self.assertIn("run_index.csv", created_names)
            self.assertIn("run_summary.csv", created_names)
            self.assertIn("group_summary.csv", created_names)
            self.assertIn("checkpoint_index.csv", created_names)
            self.assertIn("learning_curve.html", created_names)
            self.assertIn("final_performance.html", created_names)
            self.assertIn("export_manifest.json", created_names)

    def test_export_analysis_bundle_supports_static_svg_and_png(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            create_demo_results(results_dir, overwrite=True)

            for figure_format in ["svg", "png"]:
                with tempfile.TemporaryDirectory() as export_dir:
                    result = export_analysis_bundle(
                        results_root=results_dir,
                        export_root=export_dir,
                        environments=["CartPole-v1"],
                        algorithms=["reinforce", "a2c", "ppo"],
                        figure_format=figure_format,
                    )

                    created_names = {path.name for path in result.created_files}
                    self.assertIn(f"learning_curve.{figure_format}", created_names)
                    self.assertIn(f"final_performance.{figure_format}", created_names)


if __name__ == "__main__":
    unittest.main()