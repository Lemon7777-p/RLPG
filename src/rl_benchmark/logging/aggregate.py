"""Result loading and aggregation helpers for analysis and dashboards."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from rl_benchmark.logging.schema import (
    CHECKPOINT_DIRNAME,
    MANIFEST_FILENAME,
    METRIC_COLUMNS,
    RunManifest,
    read_manifest,
    resolve_results_root,
)


RUN_INDEX_COLUMNS = [
    "run_id",
    "algorithm_name",
    "env_id",
    "runtime_env_id",
    "seed",
    "status",
    "source",
    "created_at",
    "total_steps",
    "total_updates",
    "notes",
    "metrics_file",
    "latest_checkpoint",
    "checkpoint_count",
    "resumed_from",
    "failure_message",
]
RUN_SUMMARY_COLUMNS = [
    "run_id",
    "algorithm_name",
    "env_id",
    "seed",
    "source",
    "score_column",
    "start_return",
    "final_return",
    "best_return",
    "return_stability",
    "steps_to_target",
    "wall_time_to_target_sec",
    "final_wall_time_sec",
]
GROUP_SUMMARY_COLUMNS = [
    "algorithm_name",
    "env_id",
    "runs",
    "mean_final_return",
    "std_final_return",
    "mean_best_return",
    "mean_steps_to_target",
    "mean_wall_time_sec",
]
CHECKPOINT_INDEX_COLUMNS = [
    "run_id",
    "checkpoint_name",
    "checkpoint_path",
    "label",
    "update",
    "step",
    "modified_at",
    "size_bytes",
]


def discover_run_dirs(results_root: str | Path | None = None) -> list[Path]:
    root = resolve_results_root(results_root)
    return sorted(
        path for path in root.iterdir() if path.is_dir() and (path / MANIFEST_FILENAME).is_file()
    )


def load_run_index(results_root: str | Path | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in discover_run_dirs(results_root):
        manifest = read_manifest(run_dir / MANIFEST_FILENAME)
        rows.append(manifest.to_dict())

    if not rows:
        return pd.DataFrame(columns=RUN_INDEX_COLUMNS)

    frame = pd.DataFrame(rows)
    for column in RUN_INDEX_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[RUN_INDEX_COLUMNS]


def load_checkpoint_index(results_root: str | Path | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in discover_run_dirs(results_root):
        checkpoint_dir = run_dir / CHECKPOINT_DIRNAME
        if not checkpoint_dir.is_dir():
            continue

        for checkpoint_path in sorted(checkpoint_dir.glob("*.pt")):
            label, update, step = _parse_checkpoint_name(checkpoint_path.stem)
            stat = checkpoint_path.stat()
            rows.append(
                {
                    "run_id": run_dir.name,
                    "checkpoint_name": checkpoint_path.name,
                    "checkpoint_path": str(checkpoint_path),
                    "label": label,
                    "update": update,
                    "step": step,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "size_bytes": int(stat.st_size),
                }
            )

    if not rows:
        return pd.DataFrame(columns=CHECKPOINT_INDEX_COLUMNS)

    frame = pd.DataFrame(rows)
    for column in CHECKPOINT_INDEX_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[CHECKPOINT_INDEX_COLUMNS].sort_values(["run_id", "step", "update", "modified_at"]).reset_index(drop=True)


def load_all_metrics(results_root: str | Path | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in discover_run_dirs(results_root):
        manifest = read_manifest(run_dir / MANIFEST_FILENAME)
        metrics_path = run_dir / manifest.metrics_file
        if not metrics_path.is_file():
            continue

        metrics_frame = pd.read_csv(metrics_path)
        for column in METRIC_COLUMNS:
            if column not in metrics_frame.columns:
                metrics_frame[column] = np.nan

        metrics_frame["run_id"] = manifest.run_id
        metrics_frame["algorithm_name"] = manifest.algorithm_name
        metrics_frame["env_id"] = manifest.env_id
        metrics_frame["runtime_env_id"] = manifest.runtime_env_id
        metrics_frame["seed"] = manifest.seed
        metrics_frame["source"] = manifest.source
        frames.append(metrics_frame)

    if not frames:
        return pd.DataFrame(columns=METRIC_COLUMNS + ["run_id", "algorithm_name", "env_id", "runtime_env_id", "seed", "source"])

    return pd.concat(frames, ignore_index=True)


def build_run_summary(index_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    if index_df.empty:
        return pd.DataFrame(columns=RUN_SUMMARY_COLUMNS)

    rows: list[dict[str, Any]] = []
    for manifest_row in index_df.to_dict(orient="records"):
        run_metrics = metrics_df.loc[metrics_df["run_id"] == manifest_row["run_id"]].sort_values("step")
        if run_metrics.empty:
            rows.append(
                {
                    "run_id": manifest_row["run_id"],
                    "algorithm_name": manifest_row["algorithm_name"],
                    "env_id": manifest_row["env_id"],
                    "seed": manifest_row["seed"],
                    "source": manifest_row["source"],
                    "score_column": pd.NA,
                    "start_return": np.nan,
                    "final_return": np.nan,
                    "best_return": np.nan,
                    "return_stability": np.nan,
                    "steps_to_target": np.nan,
                    "wall_time_to_target_sec": np.nan,
                    "final_wall_time_sec": np.nan,
                }
            )
            continue

        score_column = _resolve_score_column(run_metrics)
        scores = run_metrics[score_column].dropna().reset_index(drop=True)
        if scores.empty:
            start_return = np.nan
            final_return = np.nan
            best_return = np.nan
            stability = np.nan
            steps_to_target = np.nan
            wall_time_to_target = np.nan
        else:
            start_return = float(scores.iloc[0])
            final_return = float(scores.iloc[-1])
            best_return = float(scores.max())
            stability_window = scores.tail(min(5, len(scores)))
            stability = float(stability_window.std(ddof=0)) if len(stability_window) > 1 else 0.0
            target_return = start_return + 0.9 * (best_return - start_return)
            if best_return >= start_return:
                reached_target = run_metrics.loc[run_metrics[score_column] >= target_return]
            else:
                reached_target = run_metrics.loc[run_metrics[score_column] <= target_return]
            steps_to_target = float(reached_target.iloc[0]["step"]) if not reached_target.empty else np.nan
            wall_time_to_target = (
                float(reached_target.iloc[0]["wall_time_sec"]) if not reached_target.empty else np.nan
            )

        rows.append(
            {
                "run_id": manifest_row["run_id"],
                "algorithm_name": manifest_row["algorithm_name"],
                "env_id": manifest_row["env_id"],
                "seed": manifest_row["seed"],
                "source": manifest_row["source"],
                "score_column": score_column,
                "start_return": start_return,
                "final_return": final_return,
                "best_return": best_return,
                "return_stability": stability,
                "steps_to_target": steps_to_target,
                "wall_time_to_target_sec": wall_time_to_target,
                "final_wall_time_sec": float(run_metrics["wall_time_sec"].dropna().iloc[-1]),
            }
        )

    return pd.DataFrame(rows, columns=RUN_SUMMARY_COLUMNS)


def build_group_summary(run_summary_df: pd.DataFrame) -> pd.DataFrame:
    if run_summary_df.empty:
        return pd.DataFrame(columns=GROUP_SUMMARY_COLUMNS)

    grouped = (
        run_summary_df.groupby(["algorithm_name", "env_id"], dropna=False)
        .agg(
            runs=("run_id", "count"),
            mean_final_return=("final_return", "mean"),
            std_final_return=("final_return", "std"),
            mean_best_return=("best_return", "mean"),
            mean_steps_to_target=("steps_to_target", "mean"),
            mean_wall_time_sec=("final_wall_time_sec", "mean"),
        )
        .reset_index()
    )
    grouped["std_final_return"] = grouped["std_final_return"].fillna(0.0)
    return grouped[GROUP_SUMMARY_COLUMNS]


def _resolve_score_column(metrics_df: pd.DataFrame) -> str:
    eval_returns = metrics_df["eval_episode_return"].dropna()
    if not eval_returns.empty:
        return "eval_episode_return"
    return "train_episode_return"


def _parse_checkpoint_name(stem: str) -> tuple[str, int | None, int | None]:
    match = re.match(r"(?P<label>.+)_update_(?P<update>\d+)_step_(?P<step>\d+)$", stem)
    if not match:
        return stem, None, None
    return match.group("label"), int(match.group("update")), int(match.group("step"))
