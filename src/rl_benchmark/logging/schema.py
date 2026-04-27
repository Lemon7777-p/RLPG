"""Result schema helpers for persisted experiment runs."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rl_benchmark import PROJECT_ROOT


MANIFEST_FILENAME = "manifest.json"
METRICS_FILENAME = "metrics.csv"
CHECKPOINT_DIRNAME = "checkpoints"
METRIC_COLUMNS = [
    "step",
    "update",
    "wall_time_sec",
    "train_episode_return",
    "eval_episode_return",
    "episode_length",
    "policy_loss",
    "value_loss",
    "entropy",
    "grad_norm",
    "approx_kl",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_results_root(results_root: str | Path | None = None) -> Path:
    if results_root is not None:
        root = Path(results_root)
    else:
        env_results_root = os.getenv("RL_BENCHMARK_RESULTS_ROOT")
        root = Path(env_results_root) if env_results_root else PROJECT_ROOT / "results"

    root.mkdir(parents=True, exist_ok=True)
    return root


def run_dir_for(run_id: str, results_root: str | Path | None = None) -> Path:
    run_dir = resolve_results_root(results_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def checkpoint_dir_for(run_id: str, results_root: str | Path | None = None) -> Path:
    checkpoint_dir = run_dir_for(run_id, results_root) / CHECKPOINT_DIRNAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


@dataclass(slots=True)
class RunManifest:
    run_id: str
    algorithm_name: str
    env_id: str
    runtime_env_id: str
    seed: int
    status: str = "completed"
    source: str = "training"
    created_at: str = field(default_factory=utc_now_iso)
    total_steps: int = 0
    total_updates: int = 0
    notes: str = ""
    metrics_file: str = METRICS_FILENAME
    latest_checkpoint: str | None = None
    checkpoint_count: int = 0
    resumed_from: str | None = None
    failure_message: str | None = None
    config_snapshot: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunManifest":
        return cls(**payload)


@dataclass(slots=True)
class MetricRecord:
    step: int
    update: int
    wall_time_sec: float
    train_episode_return: float | None = None
    eval_episode_return: float | None = None
    episode_length: float | None = None
    policy_loss: float | None = None
    value_loss: float | None = None
    entropy: float | None = None
    grad_norm: float | None = None
    approx_kl: float | None = None

    def to_row(self) -> dict[str, Any]:
        payload = asdict(self)
        return {column: payload.get(column) for column in METRIC_COLUMNS}

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "MetricRecord":
        return cls(
            step=int(row["step"]),
            update=int(row["update"]),
            wall_time_sec=float(row["wall_time_sec"]),
            train_episode_return=_optional_float(row.get("train_episode_return")),
            eval_episode_return=_optional_float(row.get("eval_episode_return")),
            episode_length=_optional_float(row.get("episode_length")),
            policy_loss=_optional_float(row.get("policy_loss")),
            value_loss=_optional_float(row.get("value_loss")),
            entropy=_optional_float(row.get("entropy")),
            grad_norm=_optional_float(row.get("grad_norm")),
            approx_kl=_optional_float(row.get("approx_kl")),
        )


def write_manifest(manifest: RunManifest, results_root: str | Path | None = None) -> Path:
    run_dir = run_dir_for(manifest.run_id, results_root)
    manifest_path = run_dir / MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2)
    return manifest_path


def read_manifest(path: str | Path) -> RunManifest:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return RunManifest.from_dict(payload)


def write_metrics(
    run_id: str,
    records: Iterable[MetricRecord],
    results_root: str | Path | None = None,
    filename: str = METRICS_FILENAME,
) -> Path:
    run_dir = run_dir_for(run_id, results_root)
    metrics_path = run_dir / filename
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())
    return metrics_path


def append_metrics(
    run_id: str,
    records: Iterable[MetricRecord],
    results_root: str | Path | None = None,
    filename: str = METRICS_FILENAME,
) -> Path:
    run_dir = run_dir_for(run_id, results_root)
    metrics_path = run_dir / filename
    needs_header = not metrics_path.is_file()
    with metrics_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        if needs_header:
            writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())
    return metrics_path


def read_metrics(path: str | Path) -> list[MetricRecord]:
    metrics_path = Path(path)
    if not metrics_path.is_file():
        return []

    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [MetricRecord.from_row(row) for row in reader]


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
