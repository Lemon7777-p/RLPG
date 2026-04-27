"""Refresh helpers for the Streamlit dashboard."""

from __future__ import annotations

import streamlit.components.v1 as components


def schedule_page_refresh(interval_seconds: int) -> None:
    interval_ms = max(1000, int(interval_seconds * 1000))
    components.html(
        f"""
        <script>
        const intervalMs = {interval_ms};
        setTimeout(() => {{
            const targetWindow = window.parent && window.parent.location ? window.parent : window;
            targetWindow.location.reload();
        }}, intervalMs);
        </script>
        """,
        height=0,
        width=0,
    )