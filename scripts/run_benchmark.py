"""Launch a small sequential benchmark sweep from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_benchmark.config import list_algorithms, load_defaults
from rl_benchmark.reporting import export_analysis_bundle
from rl_benchmark.runners import run_training_job


def build_parser() -> argparse.ArgumentParser:
    defaults = load_defaults()
    parser = argparse.ArgumentParser(description="Run a sequential benchmark sweep.")
    parser.add_argument("--algorithms", nargs="+", default=list_algorithms())
    parser.add_argument("--envs", nargs="+", default=defaults["experiment"]["main_envs"])
    parser.add_argument("--seeds", nargs="+", type=int, default=defaults["project"]["seed_list"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--export-root", type=Path, default=None)
    parser.add_argument("--figure-format", default="svg", choices=["html", "svg", "png"])
    parser.add_argument("--notes-prefix", default="benchmark")
    parser.add_argument("--checkpoint-interval-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    total_runs = len(args.algorithms) * len(args.envs) * len(args.seeds)
    completed = 0

    for algorithm_name in args.algorithms:
        for env_id in args.envs:
            for seed in args.seeds:
                completed += 1
                print(f"[{completed}/{total_runs}] Running {algorithm_name} on {env_id} with seed {seed}")
                result = run_training_job(
                    algorithm_name,
                    env_id,
                    seed,
                    device=args.device,
                    train_steps=args.train_steps,
                    eval_episodes=args.eval_episodes,
                    results_root=args.results_root,
                    notes=f"{args.notes_prefix}: {algorithm_name} {env_id} seed {seed}",
                    checkpoint_interval_steps=args.checkpoint_interval_steps,
                    resume=args.resume,
                    log_progress=True,
                )
                print(f"Completed {result.run_id} with {result.manifest.total_steps} steps")

    if args.export_root is not None:
        export_result = export_analysis_bundle(
            results_root=args.results_root or Path("results"),
            export_root=args.export_root,
            environments=args.envs,
            algorithms=args.algorithms,
            figure_format=args.figure_format,
        )
        print(f"Exported {len(export_result.created_files)} files to {export_result.export_root}")


if __name__ == "__main__":
    main()