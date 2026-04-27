"""Launch a single logged training run from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_benchmark.runners import run_training_job


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one logged RL training job.")
    parser.add_argument("--algorithm", required=True, choices=["reinforce", "a2c", "ppo"])
    parser.add_argument("--env", required=True, dest="env_id")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--checkpoint-interval-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_training_job(
        args.algorithm,
        args.env_id,
        args.seed,
        device=args.device,
        train_steps=args.train_steps,
        eval_episodes=args.eval_episodes,
        results_root=args.results_root,
        notes=args.notes,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        resume=args.resume,
        checkpoint_path=args.checkpoint_path,
        log_progress=True,
    )
    print(f"Completed run: {result.run_id}")
    print(f"Output dir: {result.output_dir}")
    print(f"Total steps: {result.manifest.total_steps}")
    print(f"Total updates: {result.manifest.total_updates}")
    print(f"Latest checkpoint: {result.latest_checkpoint}")
    if result.metrics:
        latest_metric = result.metrics[-1]
        print(f"Final train return: {latest_metric.train_episode_return}")
        print(f"Final eval return: {latest_metric.eval_episode_return}")


if __name__ == "__main__":
    main()
