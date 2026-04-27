"""Comparison page for the Streamlit dashboard."""

from __future__ import annotations

import streamlit as st

from rl_benchmark.dashboard.data import DashboardData
from rl_benchmark.dashboard.plots import (
    efficiency_figure,
    final_performance_figure,
    learning_curve_figure,
    stability_figure,
    wall_time_curve_figure,
)


def render(data: DashboardData) -> None:
    st.subheader("Compare Algorithms")
    if not data.has_runs:
        st.info("No runs are available to compare yet.")
        return

    environments = sorted(data.index_df["env_id"].unique())
    algorithms = sorted(data.index_df["algorithm_name"].unique())
    statuses = sorted(data.index_df["status"].dropna().unique())
    sources = sorted(data.index_df["source"].dropna().unique())
    control_left, control_middle, control_right, control_fourth = st.columns([1, 2, 1, 1])
    selected_env = control_left.selectbox("Environment", options=environments)
    selected_algorithms = control_middle.multiselect(
        "Algorithms",
        options=algorithms,
        default=algorithms,
    )
    selected_statuses = control_right.multiselect("Statuses", options=statuses, default=statuses)
    selected_sources = control_fourth.multiselect("Sources", options=sources, default=sources)

    filtered_metrics = data.metrics_df.loc[
        (data.metrics_df["env_id"] == selected_env)
        & (data.metrics_df["algorithm_name"].isin(selected_algorithms))
        & (data.metrics_df["source"].isin(selected_sources))
    ]
    filtered_run_summary = data.run_summary_df.loc[
        (data.run_summary_df["env_id"] == selected_env)
        & (data.run_summary_df["algorithm_name"].isin(selected_algorithms))
        & (data.run_summary_df["source"].isin(selected_sources))
    ]
    filtered_group_summary = data.group_summary_df.loc[
        (data.group_summary_df["env_id"] == selected_env)
        & (data.group_summary_df["algorithm_name"].isin(selected_algorithms))
    ]
    allowed_run_ids = set(
        data.index_df.loc[
            (data.index_df["env_id"] == selected_env)
            & (data.index_df["algorithm_name"].isin(selected_algorithms))
            & (data.index_df["status"].isin(selected_statuses))
            & (data.index_df["source"].isin(selected_sources)),
            "run_id",
        ]
    )
    filtered_metrics = filtered_metrics.loc[filtered_metrics["run_id"].isin(allowed_run_ids)]
    filtered_run_summary = filtered_run_summary.loc[filtered_run_summary["run_id"].isin(allowed_run_ids)]
    filtered_group_summary = filtered_group_summary.loc[
        filtered_group_summary["algorithm_name"].isin(set(filtered_run_summary["algorithm_name"]))
    ]

    if filtered_run_summary.empty:
        st.warning("The current filter returned no runs.")
        return

    if (data.active_runs_df["env_id"] == selected_env).any():
        st.info("Active runs are present for this environment. Enable auto-refresh in the sidebar to keep the view current.")

    st.plotly_chart(learning_curve_figure(filtered_metrics), width="stretch")

    chart_left, chart_right = st.columns(2)
    chart_left.plotly_chart(efficiency_figure(filtered_group_summary), width="stretch")
    chart_right.plotly_chart(stability_figure(filtered_group_summary), width="stretch")

    detail_left, detail_right = st.columns(2)
    detail_left.plotly_chart(final_performance_figure(filtered_run_summary), width="stretch")
    detail_right.plotly_chart(wall_time_curve_figure(filtered_metrics), width="stretch")

    st.markdown("### Aggregated Summary")
    st.dataframe(filtered_group_summary, width="stretch", hide_index=True)
    st.download_button(
        "Download aggregated summary CSV",
        data=filtered_group_summary.to_csv(index=False).encode("utf-8"),
        file_name=f"summary_{selected_env}.csv",
        mime="text/csv",
    )
