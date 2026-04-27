"""Utilities for exporting report-ready artifacts from run outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from plotly.graph_objects import Figure

from rl_benchmark.dashboard.data import DashboardData, load_dashboard_data
from rl_benchmark.dashboard.plots import (
    efficiency_figure,
    final_performance_figure,
    learning_curve_figure,
    stability_figure,
    wall_time_curve_figure,
)


@dataclass(slots=True)
class ExportResult:
    export_root: Path
    created_files: list[Path]
    environments: list[str]
    algorithms: list[str]


def export_analysis_bundle(
    *,
    results_root: str | Path,
    export_root: str | Path,
    environments: Iterable[str] | None = None,
    algorithms: Iterable[str] | None = None,
    figure_format: str = "html",
) -> ExportResult:
    data = load_dashboard_data(results_root)
    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    selected_environments = list(environments) if environments is not None else sorted(data.index_df["env_id"].dropna().unique())
    selected_algorithms = list(algorithms) if algorithms is not None else sorted(data.index_df["algorithm_name"].dropna().unique())
    created_files: list[Path] = []

    created_files.extend(_export_table_bundle(export_root, data))
    for env_id in selected_environments:
        created_files.extend(
            _export_environment_bundle(
                export_root=export_root,
                env_id=env_id,
                data=data,
                algorithms=selected_algorithms,
                figure_format=figure_format,
            )
        )

    manifest_path = export_root / "export_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "results_root": str(Path(results_root)),
                "export_root": str(export_root),
                "environments": selected_environments,
                "algorithms": selected_algorithms,
                "created_files": [str(path) for path in created_files],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    created_files.append(manifest_path)
    return ExportResult(
        export_root=export_root,
        created_files=created_files,
        environments=selected_environments,
        algorithms=selected_algorithms,
    )


def _export_table_bundle(export_root: Path, data: DashboardData) -> list[Path]:
    tables_dir = export_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_paths = [
        _write_csv(tables_dir / "run_index.csv", data.index_df),
        _write_csv(tables_dir / "run_summary.csv", data.run_summary_df),
        _write_csv(tables_dir / "group_summary.csv", data.group_summary_df),
        _write_csv(tables_dir / "checkpoint_index.csv", data.checkpoint_df),
    ]
    if data.has_active_runs:
        table_paths.append(_write_csv(tables_dir / "active_runs.csv", data.active_runs_df))
    return table_paths


def _export_environment_bundle(
    *,
    export_root: Path,
    env_id: str,
    data: DashboardData,
    algorithms: list[str],
    figure_format: str,
) -> list[Path]:
    env_dir = export_root / _slugify(env_id)
    env_dir.mkdir(parents=True, exist_ok=True)

    env_metrics = data.metrics_df.loc[
        (data.metrics_df["env_id"] == env_id)
        & (data.metrics_df["algorithm_name"].isin(algorithms))
    ]
    env_run_summary = data.run_summary_df.loc[
        (data.run_summary_df["env_id"] == env_id)
        & (data.run_summary_df["algorithm_name"].isin(algorithms))
    ]
    env_group_summary = data.group_summary_df.loc[
        (data.group_summary_df["env_id"] == env_id)
        & (data.group_summary_df["algorithm_name"].isin(algorithms))
    ]

    created_files = [
        _write_csv(env_dir / f"{_slugify(env_id)}_run_summary.csv", env_run_summary),
        _write_csv(env_dir / f"{_slugify(env_id)}_group_summary.csv", env_group_summary),
    ]

    figure_builders = {
        "learning_curve": learning_curve_figure(env_metrics),
        "wall_time": wall_time_curve_figure(env_metrics),
        "final_performance": final_performance_figure(env_run_summary),
        "efficiency": efficiency_figure(env_group_summary),
        "stability": stability_figure(env_group_summary),
    }
    for name, figure in figure_builders.items():
        created_files.append(_write_figure(env_dir / f"{name}.{figure_format}", figure, figure_format))

    return created_files


def _write_csv(path: Path, dataframe) -> Path:
    dataframe.to_csv(path, index=False)
    return path


def _write_figure(path: Path, figure: Figure, figure_format: str) -> Path:
    if figure_format == "html":
        figure.write_html(path, include_plotlyjs="cdn", full_html=True)
        return path
    try:
        figure.write_image(path, format=figure_format)
    except Exception as error:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            f"Static image export for '{figure_format}' requires Plotly image support. Install the 'kaleido' package and retry."
        ) from error
    return path


def _slugify(value: str) -> str:
    return value.replace("/", "-").replace(" ", "_")