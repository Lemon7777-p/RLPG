"""Deterministic demo results for dashboard development and validation."""

from __future__ import annotations

from math import exp, sin, pi
from pathlib import Path

import numpy as np

from rl_benchmark.config import load_run_config
from rl_benchmark.logging.schema import MetricRecord, RunManifest, write_manifest, write_metrics


DEMO_ENVIRONMENTS = ["CartPole-v1", "Acrobot-v1", "LunarLander-v2"]
DEMO_ALGORITHMS = ["reinforce", "a2c", "ppo"]
DEMO_SEEDS = [7, 11, 23]
DEMO_POINTS = 18
DEMO_PROFILE = {
    "CartPole-v1": {
        "start": 18.0,
        "targets": {"reinforce": 210.0, "a2c": 360.0, "ppo": 500.0},
        "sharpness": {"reinforce": 2.2, "a2c": 3.0, "ppo": 4.1},
        "noise": {"reinforce": 16.0, "a2c": 10.0, "ppo": 7.0},
        "time_factor": {"reinforce": 1.0, "a2c": 1.15, "ppo": 1.55},
    },
    "Acrobot-v1": {
        "start": -500.0,
        "targets": {"reinforce": -180.0, "a2c": -120.0, "ppo": -88.0},
        "sharpness": {"reinforce": 1.6, "a2c": 2.1, "ppo": 2.8},
        "noise": {"reinforce": 22.0, "a2c": 14.0, "ppo": 10.0},
        "time_factor": {"reinforce": 1.0, "a2c": 1.2, "ppo": 1.5},
    },
    "LunarLander-v2": {
        "start": -240.0,
        "targets": {"reinforce": 35.0, "a2c": 145.0, "ppo": 235.0},
        "sharpness": {"reinforce": 1.5, "a2c": 2.1, "ppo": 2.7},
        "noise": {"reinforce": 35.0, "a2c": 24.0, "ppo": 18.0},
        "time_factor": {"reinforce": 1.0, "a2c": 1.25, "ppo": 1.6},
    },
}


def create_demo_results(results_root: str | Path, overwrite: bool = False) -> list[str]:
    created_run_ids: list[str] = []
    for env_id in DEMO_ENVIRONMENTS:
        for algorithm_name in DEMO_ALGORITHMS:
            for seed in DEMO_SEEDS:
                run_id = f"demo_{algorithm_name}_{env_id}_seed{seed}"
                manifest_path = Path(results_root) / run_id / "manifest.json"
                if manifest_path.exists() and not overwrite:
                    continue

                run_config = load_run_config(algorithm_name, env_id)
                metrics = _build_demo_metrics(run_config, algorithm_name, env_id, seed)
                manifest = RunManifest(
                    run_id=run_id,
                    algorithm_name=algorithm_name,
                    env_id=env_id,
                    runtime_env_id=run_config["run"]["runtime_env_id"],
                    seed=seed,
                    source="demo",
                    total_steps=int(metrics[-1].step),
                    total_updates=len(metrics),
                    notes="Synthetic but deterministic demo data for the Streamlit dashboard.",
                    config_snapshot=run_config,
                )
                write_manifest(manifest, results_root)
                write_metrics(run_id, metrics, results_root)
                created_run_ids.append(run_id)

    return created_run_ids


def _build_demo_metrics(
    run_config: dict[str, object],
    algorithm_name: str,
    env_id: str,
    seed: int,
) -> list[MetricRecord]:
    profile = DEMO_PROFILE[env_id]
    total_steps = int(run_config["environment"]["train_steps"])
    rng = np.random.default_rng(seed + len(env_id) * 17 + len(algorithm_name) * 31)
    step_points = np.linspace(0, total_steps, num=DEMO_POINTS, dtype=int)

    start = profile["start"]
    target = profile["targets"][algorithm_name]
    sharpness = profile["sharpness"][algorithm_name]
    noise_scale = profile["noise"][algorithm_name]
    time_factor = profile["time_factor"][algorithm_name]

    metrics: list[MetricRecord] = []
    for index, step in enumerate(step_points, start=1):
        progress = 0.0 if total_steps == 0 else step / total_steps
        curve = 1.0 - exp(-sharpness * progress)
        baseline = start + (target - start) * curve
        oscillation = sin(progress * pi * 3.0) * noise_scale * 0.2
        train_return = baseline + oscillation + rng.normal(0.0, noise_scale)
        eval_return = baseline + rng.normal(0.0, noise_scale * 0.55)
        episode_length = max(20.0, 90.0 + progress * 240.0 + rng.normal(0.0, 12.0))
        wall_time_sec = float(progress * time_factor * 95.0 + rng.normal(0.0, 1.2))
        policy_loss = float((1.0 - curve) * (1.4 if algorithm_name == "reinforce" else 0.9) + rng.normal(0.0, 0.03))
        value_loss = (
            float((1.0 - curve) * (2.0 if env_id == "LunarLander-v2" else 1.2) + rng.normal(0.0, 0.04))
            if algorithm_name != "reinforce"
            else None
        )
        entropy = float(max(0.02, 0.85 - progress * 0.55 + rng.normal(0.0, 0.02)))
        grad_norm = float(max(0.05, 0.9 - progress * 0.35 + rng.normal(0.0, 0.03)))
        approx_kl = float(max(0.001, 0.012 + progress * 0.01 + rng.normal(0.0, 0.0015))) if algorithm_name == "ppo" else None

        metrics.append(
            MetricRecord(
                step=int(step),
                update=index,
                wall_time_sec=max(0.0, wall_time_sec),
                train_episode_return=float(train_return),
                eval_episode_return=float(eval_return),
                episode_length=float(episode_length),
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy=entropy,
                grad_norm=grad_norm,
                approx_kl=approx_kl,
            )
        )

    return metrics
