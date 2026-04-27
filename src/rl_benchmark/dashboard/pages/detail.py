"""Run detail page for the Streamlit dashboard."""

from __future__ import annotations

from typing import Any

import plotly.express as px
import streamlit as st

from rl_benchmark.dashboard.data import DashboardData
from rl_benchmark.dashboard.jobs import load_background_run_request, relaunch_background_training_job


def render(data: DashboardData) -> None:
    st.subheader("Run Detail")
    if not data.has_runs:
        st.info("No run artifacts are available yet.")
        return

    last_recovery_launch = st.session_state.pop("last_detail_background_launch", None)
    if last_recovery_launch is not None:
        st.success(
            f"Launched {last_recovery_launch['run_id']} in the background via {last_recovery_launch['action']}. PID: {last_recovery_launch['pid']}."
        )

    run_options = list(data.index_df.sort_values("created_at", ascending=False)["run_id"])
    selected_run_id = st.selectbox("Run", options=run_options)
    manifest_row = data.run_manifest_row(selected_run_id)
    run_metrics = data.run_metrics(selected_run_id)
    run_summary = data.run_summary_row(selected_run_id)
    checkpoint_df = data.run_checkpoint_df(selected_run_id)
    background_row = data.run_background_row(selected_run_id)
    background_log_tail = data.run_background_log_tail(selected_run_id)
    background_request = load_background_run_request(data.results_root / selected_run_id)

    status = str(manifest_row.get("status", "unknown"))
    if status == "running":
        st.info("This run is currently marked as running. Enable auto-refresh in the sidebar to keep its metrics current.")
    elif status == "failed":
        failure_message = manifest_row.get("failure_message")
        if failure_message:
            st.error(f"This run failed: {failure_message}")
        else:
            st.error("This run failed. Check the run manifest and background log for details.")
        _render_recovery_controls(
            results_root=data.results_root,
            run_id=selected_run_id,
            has_checkpoint=bool(manifest_row.get("latest_checkpoint")),
            background_request=background_request,
        )

    stat_left, stat_middle, stat_right = st.columns(3)
    stat_left.metric("Status", status)
    stat_middle.metric("Checkpoints", int(manifest_row.get("checkpoint_count") or 0))
    stat_right.metric("Updates", int(manifest_row.get("total_updates") or 0))

    if run_summary is not None:
        summary_left, summary_middle, summary_right = st.columns(3)
        summary_left.metric("Final return", f"{run_summary['final_return']:.2f}" if run_summary.get("final_return") == run_summary.get("final_return") else "n/a")
        summary_middle.metric("Best return", f"{run_summary['best_return']:.2f}" if run_summary.get("best_return") == run_summary.get("best_return") else "n/a")
        summary_right.metric("Stability", f"{run_summary['return_stability']:.2f}" if run_summary.get("return_stability") == run_summary.get("return_stability") else "n/a")

    top_left, top_right = st.columns([1, 1])
    top_left.json(manifest_row)

    if run_metrics.empty:
        top_right.info("This run does not have a metrics file yet.")
        _render_background_diagnostics(background_row, background_log_tail)
        return

    return_figure = px.line(
        run_metrics,
        x="step",
        y=[column for column in ["train_episode_return", "eval_episode_return"] if column in run_metrics.columns],
        template="plotly_white",
        title="Run Returns Over Time",
        labels={"value": "Return", "step": "Environment steps", "variable": "Series"},
    )
    top_right.plotly_chart(return_figure, width="stretch")

    loss_columns = [column for column in ["policy_loss", "value_loss", "entropy", "approx_kl"] if column in run_metrics.columns]
    available_loss_columns = [column for column in loss_columns if run_metrics[column].notna().any()]
    if available_loss_columns:
        loss_figure = px.line(
            run_metrics,
            x="step",
            y=available_loss_columns,
            template="plotly_white",
            title="Optimization Metrics",
            labels={"value": "Metric value", "step": "Environment steps", "variable": "Metric"},
        )
        st.plotly_chart(loss_figure, width="stretch")

    if not checkpoint_df.empty:
        st.markdown("### Checkpoint History")
        st.dataframe(
            checkpoint_df[[column for column in ["checkpoint_name", "label", "update", "step", "modified_at", "size_bytes"] if column in checkpoint_df.columns]],
            width="stretch",
            hide_index=True,
        )

    _render_background_diagnostics(background_row, background_log_tail)

    st.download_button(
        "Download run metrics CSV",
        data=run_metrics.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_run_id}_metrics.csv",
        mime="text/csv",
    )

    st.markdown("### Raw Metrics")
    st.dataframe(run_metrics, width="stretch", hide_index=True)


def _render_background_diagnostics(background_row: dict[str, Any] | None, background_log_tail: str | None) -> None:
    if background_row is None and background_log_tail is None:
        return

    st.markdown("### Background Job Diagnostics")
    diagnostics_left, diagnostics_right = st.columns([1, 1])
    if background_row is not None:
        command = background_row.get("command")
        metadata_row = {key: value for key, value in background_row.items() if key != "command"}
        diagnostics_left.json(metadata_row)
        if command:
            diagnostics_left.caption("Launch command")
            diagnostics_left.code(" ".join(str(part) for part in command), language="text")
    else:
        diagnostics_left.info("No background launch metadata is available for this run.")

    if background_log_tail:
        diagnostics_right.caption("Recent log tail")
        diagnostics_right.code(background_log_tail, language="text")
    else:
        diagnostics_right.info("No background log output is available yet.")


def _render_recovery_controls(
    *,
    results_root,
    run_id: str,
    has_checkpoint: bool,
    background_request,
) -> None:
    st.markdown("### Recovery Controls")
    if background_request is None:
        st.info("No persisted launch request is available to retry this run automatically.")
        return

    st.caption(
        f"Recovered launch settings: train steps {background_request.train_steps}, eval episodes {background_request.eval_episodes}, device {background_request.device}."
    )
    retry_column, resume_column = st.columns(2)
    if retry_column.button("Retry In Background", key=f"retry_background_{run_id}"):
        _trigger_background_recovery(results_root=results_root, run_id=run_id, resume=False, action_label="retry")

    if has_checkpoint:
        if resume_column.button("Resume In Background", key=f"resume_background_{run_id}"):
            _trigger_background_recovery(results_root=results_root, run_id=run_id, resume=True, action_label="resume")
    else:
        resume_column.caption("No checkpoint is available to resume this run.")


def _trigger_background_recovery(*, results_root, run_id: str, resume: bool, action_label: str) -> None:
    try:
        launch = relaunch_background_training_job(results_root / run_id, resume=resume)
    except (FileNotFoundError, RuntimeError) as error:
        st.error(str(error))
        return

    st.session_state["last_detail_background_launch"] = {
        "run_id": launch.run_id,
        "pid": launch.pid,
        "action": action_label,
    }
    st.rerun()
