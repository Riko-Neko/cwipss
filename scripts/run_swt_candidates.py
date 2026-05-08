#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ce4_period_search import run_swt_scan
from ce4_period_search.config import SWTScanConfig, load_swt_config, resolve_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run block-wise time-axis SWT period-candidate detection.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config.")
    parser.add_argument("--input", type=str, default=None, help="Input data path supported by the active reader.")
    parser.add_argument("--f-start", type=float, default=None, help="Frequency start in MHz.")
    parser.add_argument("--f-stop", type=float, default=None, help="Frequency stop in MHz.")
    parser.add_argument("--t-start", type=int, default=None, help="Record start index.")
    parser.add_argument("--t-stop", type=int, default=None, help="Record stop index.")
    parser.add_argument("--wavelet", type=str, default=None, help="Discrete wavelet used by SWT.")
    parser.add_argument("--levels", type=int, default=None, help="SWT levels.")
    parser.add_argument("--block-channels", type=int, default=None, help="Frequency channels per block.")
    parser.add_argument("--threshold", type=float, default=None, help="Local robust S/N threshold.")
    parser.add_argument("--min-pixels", type=int, default=None, help="Minimum connected-component size.")
    parser.add_argument("--local-time", type=int, default=None, help="Local median window in records.")
    parser.add_argument("--local-freq", type=int, default=None, help="Local median window in channels.")
    parser.add_argument(
        "--veto-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write candidates_reviewed.csv with configured veto rules.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional stable run directory name.")
    parser.add_argument(
        "--max-candidates-per-block",
        type=int,
        default=None,
        help="Cap components retained per block and level.",
    )
    parser.add_argument("--visualize", action="store_true", default=None, help="Write staged visualization PNGs.")
    parser.add_argument("--viz-max-blocks", type=int, default=None, help="Maximum blocks to visualize; 0 means all.")
    parser.add_argument("--viz-max-levels", type=int, default=None, help="Maximum SWT levels per block to visualize; 0 means all.")
    parser.add_argument("--viz-top-candidates", type=int, default=None, help="Maximum candidate overlays/scatter points.")
    parser.add_argument("--viz-dpi", type=int, default=None, help="Visualization image DPI.")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> SWTScanConfig:
    overrides = {}
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        mapped_key = {
            "visualize": "visualization_enabled",
            "viz_max_blocks": "visualization_max_blocks",
            "viz_max_levels": "visualization_max_levels",
            "viz_top_candidates": "visualization_top_candidates",
            "viz_dpi": "visualization_dpi",
        }.get(key, key)
        overrides[mapped_key] = value
    config = load_swt_config(args.config, overrides=overrides)
    config = resolve_output_dir(config, PROJECT_DIR)
    if not config.input:
        raise SystemExit("--input is required unless provided by --config")
    return config


def main() -> None:
    config = resolve_config(parse_args())
    run_dir = run_swt_scan(config)
    print(f"Run directory: {run_dir}")
    print(f"Candidates: {run_dir / 'candidates_raw.csv'}")
    print(f"Summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
