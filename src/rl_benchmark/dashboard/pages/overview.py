"""Overview page for the Streamlit dashboard."""

from __future__ import annotations

import streamlit as st

from rl_benchmark.dashboard.data import DashboardData


def render(data: DashboardData) -> None:
    st.subheader("Overview")
    if not data.has_runs:
        st.info("No result artifacts were found yet. Use the Experiment Builder page to generate demo results or point the dashboard at a populated results directory.")
        return

    run_count = int(len(data.index_df))
    env_count = int(data.index_df["env_id"].nunique())
    algorithm_count = int(data.index_df["algorithm_name"].nunique())
    active_count = int(len(data.active_runs_df))
    checkpointed_count = int((data.index_df["checkpoint_count"].fillna(0) > 0).sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Runs", run_count)
    col2.metric("Algorithms", algorithm_count)
    col3.metric("Environments", env_count)
    col4.metric("Active runs", active_count)

    support_left, support_right = st.columns(2)
    support_left.metric("Checkpointed runs", checkpointed_count)
    support_right.metric("Loaded checkpoints", int(len(data.checkpoint_df)))

    if data.has_active_runs:
        st.markdown("### Active Runs")
        active_columns = [
            column
            for column in ["run_id", "algorithm_name", "env_id", "status", "total_steps", "total_updates", "latest_checkpoint"]
            if column in data.active_runs_df.columns
        ]
        st.dataframe(
            data.active_runs_df[active_columns].sort_values("created_at", ascending=False),
            width="stretch",
            hide_index=True,
        )

    if not data.active_background_runs_df.empty:
        st.markdown("### Active Background Jobs")
        background_columns = [
            column
            for column in [
                "run_id",
                "algorithm_name",
                "env_id",
                "pid",
                "launched_at",
                "log_size_bytes",
                "total_steps",
                "total_updates",
            ]
            if column in data.active_background_runs_df.columns
        ]
        st.dataframe(
            data.active_background_runs_df[background_columns].sort_values("launched_at", ascending=False, na_position="last"),
            width="stretch",
            hide_index=True,
        )

    st.markdown("### Benchmark Coverage")
    coverage = (
        data.index_df.groupby(["env_id", "algorithm_name"], as_index=False)["run_id"]
        .count()
        .rename(columns={"run_id": "runs"})
        .sort_values(["env_id", "algorithm_name"])
    )
    st.dataframe(coverage, width="stretch", hide_index=True)

    st.markdown("### Run Status")
    status_breakdown = (
        data.index_df.groupby(["status", "source"], as_index=False)["run_id"]
        .count()
        .rename(columns={"run_id": "runs"})
        .sort_values(["status", "source"])
    )
    st.dataframe(status_breakdown, width="stretch", hide_index=True)

    st.markdown("### Latest Runs")
    latest_runs = data.index_df.sort_values("created_at", ascending=False).head(10)
    st.dataframe(latest_runs, width="stretch", hide_index=True)
