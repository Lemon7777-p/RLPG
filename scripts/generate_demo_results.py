"""Generate deterministic demo results for the Streamlit dashboard."""

from __future__ import annotations

from rl_benchmark.logging.demo import create_demo_results
from rl_benchmark.logging.schema import resolve_results_root


def main() -> None:
    results_root = resolve_results_root()
    created_run_ids = create_demo_results(results_root, overwrite=True)
    print(f"Generated {len(created_run_ids)} demo runs in {results_root}")


if __name__ == "__main__":
    main()
