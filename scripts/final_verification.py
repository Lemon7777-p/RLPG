"""Run a final end-to-end smoke verification of the current project."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from rl_benchmark.dashboard.data import load_dashboard_data
from rl_benchmark.reporting import export_analysis_bundle
from rl_benchmark.runners import run_training_job
from rl_benchmark.verification import DEFAULT_ALGORITHMS, DEFAULT_MAIN_ENVIRONMENTS, DEFAULT_SEEDS, build_smoke_verification_jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an end-to-end RL benchmark smoke verification.")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--export-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--algorithms", nargs="+", default=list(DEFAULT_ALGORITHMS))
    parser.add_argument("--envs", nargs="+", default=list(DEFAULT_MAIN_ENVIRONMENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--reinforce-train-steps", type=int, default=1)
    parser.add_argument("--actor-critic-train-steps", type=int, default=16)
    parser.add_argument("--figure-format", default="svg", choices=["html", "svg", "png"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    temp_results_dir: tempfile.TemporaryDirectory[str] | None = None
    temp_export_dir: tempfile.TemporaryDirectory[str] | None = None

    if args.results_root is None:
        temp_results_dir = tempfile.TemporaryDirectory()
        results_root = Path(temp_results_dir.name)
    else:
        results_root = args.results_root
        results_root.mkdir(parents=True, exist_ok=True)

    if args.export_root is None:
        temp_export_dir = tempfile.TemporaryDirectory()
        export_root = Path(temp_export_dir.name)
    else:
        export_root = args.export_root
        export_root.mkdir(parents=True, exist_ok=True)

    try:
        jobs = build_smoke_verification_jobs(
            algorithms=args.algorithms,
            envs=args.envs,
            seeds=args.seeds,
            reinforce_train_steps=args.reinforce_train_steps,
            actor_critic_train_steps=args.actor_critic_train_steps,
        )
        print(f"Planned verification jobs: {len(jobs)}")
        for job in jobs:
            result = run_training_job(
                job.algorithm_name,
                job.env_id,
                job.seed,
                device=args.device,
                train_steps=job.train_steps,
                eval_episodes=1,
                results_root=results_root,
                checkpoint_interval_steps=job.checkpoint_interval_steps,
                notes="final verification",
            )
            print(f"Completed verification run: {result.run_id} ({result.manifest.total_steps} steps)")

        dashboard_data = load_dashboard_data(results_root)
        print(f"Dashboard data loaded: {len(dashboard_data.index_df)} runs")
        export_result = export_analysis_bundle(
            results_root=results_root,
            export_root=export_root,
            figure_format=args.figure_format,
        )
        print(f"Exported {len(export_result.created_files)} files to {export_result.export_root} using {args.figure_format} figures")
    finally:
        if temp_results_dir is not None:
            temp_results_dir.cleanup()
        if temp_export_dir is not None:
            temp_export_dir.cleanup()


if __name__ == "__main__":
    main()