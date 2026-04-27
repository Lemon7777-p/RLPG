"""Streamlit dashboard entry point."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from rl_benchmark.dashboard.data import load_dashboard_data
from rl_benchmark.dashboard.pages import compare, detail, overview, run
from rl_benchmark.dashboard.refresh import schedule_page_refresh
from rl_benchmark.logging.schema import resolve_results_root


PAGES = {
    "Overview": overview.render,
    "Compare": compare.render,
    "Run Detail": detail.render,
    "Experiment Builder": run.render,
}


def main() -> None:
    st.set_page_config(
        page_title="RL Benchmark Dashboard",
        page_icon="chart_with_upwards_trend",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    default_results_root = resolve_results_root()
    st.sidebar.title("RL Benchmark")
    results_root_text = st.sidebar.text_input("Results root", value=str(default_results_root))
    selected_page = st.sidebar.radio("Page", options=list(PAGES.keys()))
    if st.sidebar.button("Refresh now"):
        st.rerun()

    data = load_dashboard_data(Path(results_root_text))

    refresh_enabled = st.sidebar.checkbox(
        "Auto-refresh while runs are active",
        value=False,
        help="Reload the page periodically when there are active runs writing metrics.",
    )
    refresh_interval_seconds = st.sidebar.slider(
        "Refresh interval (s)",
        min_value=5,
        max_value=60,
        value=10,
        step=5,
        disabled=not refresh_enabled,
    )
    if refresh_enabled and data.has_active_runs:
        schedule_page_refresh(refresh_interval_seconds)

    st.title("RL Benchmark Dashboard")
    st.caption("Interactive comparison surface for REINFORCE, A2C, and PPO across classic-control tasks, with support for demo data and repository-backed results.")
    st.caption(f"Data loaded at {data.loaded_at.isoformat()}")

    if data.has_runs:
        source_counts = data.index_df["source"].value_counts().to_dict()
        active_count = int(len(data.active_runs_df))
        st.markdown(
            f"<div class='banner-card'><strong>{len(data.index_df)}</strong> runs loaded from <code>{data.results_root}</code>. Sources: {source_counts}. Active runs: {active_count}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='banner-card banner-empty'>No completed run artifacts found under <code>{data.results_root}</code>. Use the Experiment Builder page to generate demo results or point the dashboard at a populated results directory.</div>",
            unsafe_allow_html=True,
        )

    page_renderer = PAGES[selected_page]
    if selected_page == "Experiment Builder":
        page_renderer(data.results_root)
    else:
        page_renderer(data)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(17, 138, 178, 0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(208, 79, 48, 0.12), transparent 24%),
                linear-gradient(180deg, #f5f1e8 0%, #fbfaf7 100%);
        }
        .banner-card {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(31, 111, 139, 0.20);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 10px 30px rgba(24, 39, 75, 0.06);
        }
        .banner-empty {
            border-color: rgba(208, 79, 48, 0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
