"""Dashboard data-loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from rl_benchmark.dashboard.jobs import load_background_run_info, read_background_log_tail
from rl_benchmark.logging.aggregate import (
    build_group_summary,
    build_run_summary,
    load_all_metrics,
    load_checkpoint_index,
    load_run_index,
)
from rl_benchmark.logging.schema import resolve_results_root


BACKGROUND_RUN_COLUMNS = [
    "run_id",
    "algorithm_name",
    "env_id",
    "seed",
    "status",
    "total_steps",
    "total_updates",
    "pid",
    "resume",
    "launched_at",
    "log_path",
    "launch_metadata_path",
    "log_exists",
    "log_size_bytes",
    "log_modified_at",
    "command",
]


@dataclass(slots=True)
class DashboardData:
    results_root: Path
    index_df: pd.DataFrame
    metrics_df: pd.DataFrame
    run_summary_df: pd.DataFrame
    group_summary_df: pd.DataFrame
    checkpoint_df: pd.DataFrame
    background_runs_df: pd.DataFrame
    loaded_at: datetime

    @property
    def has_runs(self) -> bool:
        return not self.index_df.empty

    @property
    def active_runs_df(self) -> pd.DataFrame:
        if self.index_df.empty or "status" not in self.index_df:
            return self.index_df.iloc[0:0]
        return self.index_df.loc[self.index_df["status"] == "running"].copy()

    @property
    def has_active_runs(self) -> bool:
        return not self.active_runs_df.empty

    @property
    def has_background_runs(self) -> bool:
        return not self.background_runs_df.empty

    @property
    def active_background_runs_df(self) -> pd.DataFrame:
        if self.background_runs_df.empty or "status" not in self.background_runs_df:
            return self.background_runs_df.iloc[0:0]
        return self.background_runs_df.loc[self.background_runs_df["status"] == "running"].copy()

    def run_manifest_row(self, run_id: str) -> dict[str, Any]:
        return self.index_df.loc[self.index_df["run_id"] == run_id].iloc[0].to_dict()

    def run_metrics(self, run_id: str) -> pd.DataFrame:
        return self.metrics_df.loc[self.metrics_df["run_id"] == run_id].sort_values("step").copy()

    def run_summary_row(self, run_id: str) -> dict[str, Any] | None:
        run_summary = self.run_summary_df.loc[self.run_summary_df["run_id"] == run_id]
        if run_summary.empty:
            return None
        return run_summary.iloc[0].to_dict()

    def run_checkpoint_df(self, run_id: str) -> pd.DataFrame:
        if self.checkpoint_df.empty:
            return self.checkpoint_df.iloc[0:0]
        return self.checkpoint_df.loc[self.checkpoint_df["run_id"] == run_id].copy()

    def run_background_row(self, run_id: str) -> dict[str, Any] | None:
        if self.background_runs_df.empty:
            return None
        background_rows = self.background_runs_df.loc[self.background_runs_df["run_id"] == run_id]
        if background_rows.empty:
            return None
        return background_rows.iloc[0].to_dict()

    def run_background_log_tail(self, run_id: str, *, max_lines: int = 40) -> str | None:
        return read_background_log_tail(self.results_root / run_id, max_lines=max_lines)


def load_dashboard_data(results_root: str | Path | None = None) -> DashboardData:
    resolved_root = resolve_results_root(results_root)
    index_df = load_run_index(resolved_root)
    metrics_df = load_all_metrics(resolved_root)
    run_summary_df = build_run_summary(index_df, metrics_df)
    group_summary_df = build_group_summary(run_summary_df)
    checkpoint_df = load_checkpoint_index(resolved_root)
    background_runs_df = _load_background_runs(index_df, resolved_root)
    return DashboardData(
        results_root=resolved_root,
        index_df=index_df,
        metrics_df=metrics_df,
        run_summary_df=run_summary_df,
        group_summary_df=group_summary_df,
        checkpoint_df=checkpoint_df,
        background_runs_df=background_runs_df,
        loaded_at=datetime.now(timezone.utc),
    )


def _load_background_runs(index_df: pd.DataFrame, results_root: Path) -> pd.DataFrame:
    if index_df.empty:
        return pd.DataFrame(columns=BACKGROUND_RUN_COLUMNS)

    rows: list[dict[str, Any]] = []
    for manifest_row in index_df.to_dict(orient="records"):
        run_id = manifest_row.get("run_id")
        if not run_id:
            continue

        background_info = load_background_run_info(results_root / str(run_id))
        if background_info is None:
            continue

        rows.append(
            {
                "run_id": run_id,
                "algorithm_name": manifest_row.get("algorithm_name"),
                "env_id": manifest_row.get("env_id"),
                "seed": manifest_row.get("seed"),
                "status": manifest_row.get("status"),
                "total_steps": manifest_row.get("total_steps"),
                "total_updates": manifest_row.get("total_updates"),
                **background_info.to_row(),
            }
        )

    if not rows:
        return pd.DataFrame(columns=BACKGROUND_RUN_COLUMNS)

    frame = pd.DataFrame(rows)
    for column in BACKGROUND_RUN_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[BACKGROUND_RUN_COLUMNS].sort_values("launched_at", ascending=False, na_position="last").reset_index(drop=True)
