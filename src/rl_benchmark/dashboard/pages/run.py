"""Experiment-builder page for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from rl_benchmark.config import list_algorithms, list_environments, load_run_config
from rl_benchmark.dashboard.jobs import launch_background_training_job
from rl_benchmark.logging.demo import create_demo_results
from rl_benchmark.runners import build_run_id, run_training_job


def render(results_root: Path) -> None:
    st.subheader("Experiment Builder")
    st.caption("This page previews run configuration, can launch synchronous or background local training jobs, and can seed the dashboard with deterministic demo results.")

    last_background_launch = st.session_state.pop("last_background_launch", None)
    if last_background_launch is not None:
        st.success(
            f"Launched {last_background_launch['run_id']} in the background. PID: {last_background_launch['pid']}. Log: {last_background_launch['log_path']}"
        )

    algorithms = list_algorithms()
    environments = list_environments()
    control_left, control_middle, control_right = st.columns(3)
    algorithm_name = control_left.selectbox("Algorithm", options=algorithms)
    env_id = control_middle.selectbox("Environment", options=environments)
    seed = int(control_right.number_input("Seed", value=7, min_value=0, step=1))

    run_config = load_run_config(algorithm_name, env_id)
    run_id = build_run_id(algorithm_name, env_id, seed)
    default_train_steps = int(run_config["environment"]["train_steps"])
    default_eval_episodes = int(run_config["evaluation"].get("episodes", 10))

    execution_left, execution_middle, execution_right = st.columns(3)
    train_steps = int(
        execution_left.number_input(
            "Train steps",
            value=default_train_steps,
            min_value=1,
            step=1000,
        )
    )
    eval_episodes = int(
        execution_middle.number_input(
            "Eval episodes",
            value=default_eval_episodes,
            min_value=1,
            step=1,
        )
    )
    device = execution_right.selectbox("Device", options=["auto", "cpu"])

    checkpoint_left, checkpoint_middle = st.columns(2)
    checkpoint_interval = int(
        checkpoint_left.number_input(
            "Checkpoint interval steps",
            value=int(run_config["runtime"].get("checkpoint_interval_steps", 25000)),
            min_value=0,
            step=1000,
        )
    )
    resume_existing = checkpoint_middle.checkbox("Resume existing run", value=False)

    notes = st.text_input("Run notes", value="")

    info_left, info_right = st.columns(2)
    info_left.metric("Resolved run id", run_id)
    info_right.metric("Runtime environment", run_config["run"]["runtime_env_id"])

    action_left, action_middle, action_right = st.columns(3)
    if action_left.button("Launch Background Job", type="primary"):
        try:
            launch = launch_background_training_job(
                algorithm_name=algorithm_name,
                env_id=env_id,
                seed=seed,
                device=device,
                train_steps=train_steps,
                eval_episodes=eval_episodes,
                results_root=results_root,
                notes=notes,
                checkpoint_interval_steps=checkpoint_interval,
                resume=resume_existing,
            )
        except (FileNotFoundError, RuntimeError) as error:
            st.error(str(error))
        else:
            st.session_state["last_background_launch"] = {
                "run_id": launch.run_id,
                "pid": launch.pid,
                "log_path": str(launch.log_path),
            }
            st.rerun()

    if action_middle.button("Run Training Job Now"):
        try:
            with st.spinner("Running training job..."):
                result = run_training_job(
                    algorithm_name,
                    env_id,
                    seed,
                    device=device,
                    train_steps=train_steps,
                    eval_episodes=eval_episodes,
                    results_root=results_root,
                    notes=notes,
                    checkpoint_interval_steps=checkpoint_interval,
                    resume=resume_existing,
                )
        except FileNotFoundError as error:
            st.error(str(error))
            result = None
        if result is None:
            return
        latest_metric = result.metrics[-1] if result.metrics else None
        st.success(f"Completed {result.run_id} with {result.manifest.total_updates} updates and {result.manifest.total_steps} steps.")
        if latest_metric is not None:
            metric_left, metric_right, metric_third = st.columns(3)
            metric_left.metric("Final train return", f"{latest_metric.train_episode_return:.2f}" if latest_metric.train_episode_return is not None else "n/a")
            metric_right.metric("Final eval return", f"{latest_metric.eval_episode_return:.2f}" if latest_metric.eval_episode_return is not None else "n/a")
            metric_third.metric("Wall time (s)", f"{latest_metric.wall_time_sec:.2f}")
        st.caption(f"Latest checkpoint: {result.latest_checkpoint}")
        st.caption("For long jobs, use Launch Background Job so the page remains responsive while the overview and detail pages refresh from persisted artifacts.")

    st.markdown("### Run Configuration Preview")
    st.json(run_config)

    if action_right.button("Generate Demo Results"):
        created = create_demo_results(results_root, overwrite=False)
        st.success(f"Generated or confirmed {len(created)} demo runs under {results_root}.")
        st.caption("Reload the page or switch to Compare to inspect the generated results.")
