"""Plotly chart builders for the Streamlit dashboard."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLOR_MAP = {
    "reinforce": "#d04f30",
    "a2c": "#1f6f8b",
    "ppo": "#2f855a",
}


def empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font={"size": 16})
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.update_layout(template="plotly_white", margin={"l": 20, "r": 20, "t": 20, "b": 20})
    return figure


def learning_curve_figure(metrics_df: pd.DataFrame) -> go.Figure:
    if metrics_df.empty:
        return empty_figure("No metrics available for the selected filter.")

    score_column = _resolve_score_column(metrics_df)
    plot_df = metrics_df.dropna(subset=[score_column]).copy()
    if plot_df.empty:
        return empty_figure("No return values available for the selected filter.")

    grouped = (
        plot_df.groupby(["algorithm_name", "step"], as_index=False)[score_column]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_score", "std": "std_score"})
    )
    grouped["std_score"] = grouped["std_score"].fillna(0.0)

    figure = go.Figure()
    for algorithm_name in grouped["algorithm_name"].unique():
        algorithm_df = grouped.loc[grouped["algorithm_name"] == algorithm_name].sort_values("step")
        color = COLOR_MAP.get(algorithm_name, "#4c566a")
        upper = algorithm_df["mean_score"] + algorithm_df["std_score"]
        lower = algorithm_df["mean_score"] - algorithm_df["std_score"]

        figure.add_trace(
            go.Scatter(
                x=algorithm_df["step"],
                y=upper,
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=algorithm_df["step"],
                y=lower,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=_rgba(color, 0.14),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=algorithm_df["step"],
                y=algorithm_df["mean_score"],
                mode="lines+markers",
                name=algorithm_name.upper(),
                line={"color": color, "width": 3},
                marker={"size": 6},
            )
        )

    figure.update_layout(
        title="Learning Curve by Environment Steps",
        template="plotly_white",
        xaxis_title="Environment steps",
        yaxis_title=_axis_label_for_score(score_column),
        legend_title="Algorithm",
        margin={"l": 30, "r": 20, "t": 55, "b": 30},
    )
    return figure


def wall_time_curve_figure(metrics_df: pd.DataFrame) -> go.Figure:
    if metrics_df.empty:
        return empty_figure("No metrics available for the selected filter.")

    score_column = _resolve_score_column(metrics_df)
    plot_df = metrics_df.dropna(subset=[score_column, "wall_time_sec"]).copy()
    if plot_df.empty:
        return empty_figure("No wall-time metrics available for the selected filter.")

    grouped = (
        plot_df.groupby(["algorithm_name", "wall_time_sec"], as_index=False)[score_column]
        .mean()
        .rename(columns={score_column: "mean_score"})
    )
    figure = px.line(
        grouped,
        x="wall_time_sec",
        y="mean_score",
        color="algorithm_name",
        markers=True,
        color_discrete_map=COLOR_MAP,
        template="plotly_white",
        title="Return by Wall-Clock Time",
        labels={"wall_time_sec": "Wall time (s)", "mean_score": _axis_label_for_score(score_column), "algorithm_name": "Algorithm"},
    )
    figure.update_layout(margin={"l": 30, "r": 20, "t": 55, "b": 30})
    return figure


def final_performance_figure(run_summary_df: pd.DataFrame) -> go.Figure:
    if run_summary_df.empty:
        return empty_figure("No runs available for final-performance comparison.")

    plot_df = run_summary_df.dropna(subset=["final_return"])
    if plot_df.empty:
        return empty_figure("No final-return values available for the selected filter.")

    figure = px.box(
        plot_df,
        x="algorithm_name",
        y="final_return",
        color="algorithm_name",
        points="all",
        color_discrete_map=COLOR_MAP,
        template="plotly_white",
        title="Final Performance Distribution",
        labels={"algorithm_name": "Algorithm", "final_return": "Final return"},
    )
    figure.update_layout(showlegend=False, margin={"l": 30, "r": 20, "t": 55, "b": 30})
    return figure


def efficiency_figure(group_summary_df: pd.DataFrame) -> go.Figure:
    if group_summary_df.empty:
        return empty_figure("No aggregated runs available for efficiency comparison.")

    plot_df = group_summary_df.dropna(subset=["mean_steps_to_target"])
    if plot_df.empty:
        return empty_figure("No efficiency metric is available for the selected filter.")

    figure = px.bar(
        plot_df,
        x="algorithm_name",
        y="mean_steps_to_target",
        color="algorithm_name",
        color_discrete_map=COLOR_MAP,
        template="plotly_white",
        title="Learning Efficiency: Mean Steps to Reach 90% of Best Return",
        labels={"algorithm_name": "Algorithm", "mean_steps_to_target": "Steps to target"},
    )
    figure.update_layout(showlegend=False, margin={"l": 30, "r": 20, "t": 55, "b": 30})
    return figure


def stability_figure(group_summary_df: pd.DataFrame) -> go.Figure:
    if group_summary_df.empty:
        return empty_figure("No aggregated runs available for stability comparison.")

    figure = px.bar(
        group_summary_df,
        x="algorithm_name",
        y="std_final_return",
        color="algorithm_name",
        color_discrete_map=COLOR_MAP,
        template="plotly_white",
        title="Training Stability: Final Return Standard Deviation Across Seeds",
        labels={"algorithm_name": "Algorithm", "std_final_return": "Std. of final return"},
        hover_data=["runs", "mean_final_return"],
    )
    figure.update_layout(showlegend=False, margin={"l": 30, "r": 20, "t": 55, "b": 30})
    return figure


def _resolve_score_column(metrics_df: pd.DataFrame) -> str:
    if "eval_episode_return" in metrics_df and metrics_df["eval_episode_return"].notna().any():
        return "eval_episode_return"
    return "train_episode_return"


def _axis_label_for_score(score_column: str) -> str:
    return "Evaluation return" if score_column == "eval_episode_return" else "Training return"


def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    red, green, blue = tuple(int(hex_color[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha})"
