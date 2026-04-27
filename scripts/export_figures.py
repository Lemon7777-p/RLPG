"""Export figures and tables from persisted run artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_benchmark.reporting import export_analysis_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export benchmark figures and tables.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--export-root", type=Path, default=Path("exports"))
    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument("--algorithms", nargs="+", default=None)
    parser.add_argument("--figure-format", default="html", choices=["html", "svg", "png"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = export_analysis_bundle(
        results_root=args.results_root,
        export_root=args.export_root,
        environments=args.envs,
        algorithms=args.algorithms,
        figure_format=args.figure_format,
    )
    print(f"Export root: {result.export_root}")
    print(f"Exported files: {len(result.created_files)}")
    for path in result.created_files:
        print(path)


if __name__ == "__main__":
    main()
