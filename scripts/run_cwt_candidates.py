#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ce4_period_search import run_cwt_search
from ce4_period_search.config import CWTSearchConfig, load_cwt_config, resolve_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CWT period-channel candidate detection.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config.")
    parser.add_argument("--input", type=str, default=None, help="Input data path supported by the active reader.")
    parser.add_argument("--f-start", type=float, default=None, help="Frequency start in MHz.")
    parser.add_argument("--f-stop", type=float, default=None, help="Frequency stop in MHz.")
    parser.add_argument("--t-start", type=int, default=None, help="Record start index.")
    parser.add_argument("--t-stop", type=int, default=None, help="Record stop index.")
    parser.add_argument("--wavelet", type=str, default=None, help="PyWavelets CWT wavelet.")
    parser.add_argument("--cwt-method", choices=["conv", "fft"], default=None, help="PyWavelets CWT computation method.")
    parser.add_argument("--period-min-records", type=float, default=None, help="Minimum CWT period in records.")
    parser.add_argument("--period-max-records", type=float, default=None, help="Maximum CWT period in records.")
    parser.add_argument("--period-count", type=int, default=None, help="Number of CWT periods.")
    parser.add_argument("--period-spacing", choices=["log", "linear"], default=None, help="CWT period spacing.")
    parser.add_argument("--block-channels", type=int, default=None, help="Frequency channels per block.")
    parser.add_argument("--time-aggregation", type=str, default=None, help="Time aggregation for period-channel response.")
    parser.add_argument("--aggregation-percentile", type=float, default=None, help="Percentile for percentile aggregation.")
    parser.add_argument("--threshold", type=float, default=None, help="Minimum channel-wise period peak score.")
    parser.add_argument("--min-prominence", type=float, default=None, help="Minimum 1D period-peak prominence.")
    parser.add_argument("--dog-sigma-peak", type=float, default=None, help="Narrow Gaussian sigma for period-profile DoG.")
    parser.add_argument("--dog-sigma-background", type=float, default=None, help="Broad Gaussian sigma for period-profile DoG.")
    parser.add_argument("--min-width-bins", type=float, default=None, help="Minimum 1D peak width in period bins.")
    parser.add_argument("--max-width-bins", type=float, default=None, help="Maximum 1D peak width in period bins.")
    parser.add_argument("--min-distance-bins", type=int, default=None, help="Minimum separation between peaks in period bins.")
    parser.add_argument("--max-candidates-per-channel", type=int, default=None, help="Cap retained peaks per frequency channel.")
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
        help="Cap peaks retained per block.",
    )
    parser.add_argument("--visualize", action="store_true", default=None, help="Write staged visualization PNGs.")
    parser.add_argument("--viz-max-blocks", type=int, default=None, help="Maximum blocks to visualize; 0 means all.")
    parser.add_argument("--viz-max-channels", type=int, default=None, help="Representative channels per block for period-time CWT plots.")
    parser.add_argument("--viz-top-candidates", type=int, default=None, help="Maximum candidate overlays/scatter points.")
    parser.add_argument("--viz-dpi", type=int, default=None, help="Visualization image DPI.")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show CWT channel-progress tqdm.",
    )
    parser.add_argument(
        "--progress-leave",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Leave the tqdm progress bar on screen when finished.",
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> CWTSearchConfig:
    overrides = {}
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        mapped_key = {
            "visualize": "visualization_enabled",
            "viz_max_blocks": "visualization_max_blocks",
            "viz_max_channels": "visualization_max_channels",
            "viz_top_candidates": "visualization_top_candidates",
            "viz_dpi": "visualization_dpi",
            "progress": "progress_enabled",
            "progress_leave": "progress_leave",
        }.get(key, key)
        overrides[mapped_key] = value
    config = load_cwt_config(args.config, overrides=overrides)
    config = resolve_output_dir(config, PROJECT_DIR)
    if not config.input:
        raise SystemExit("--input is required unless provided by --config")
    return config


def main() -> None:
    config = resolve_config(parse_args())
    run_dir = run_cwt_search(config)
    print(f"Run directory: {run_dir}")
    print(f"Candidates: {run_dir / 'candidates_raw.csv'}")
    if config.visualization_enabled:
        print(f"Visualization: {run_dir / 'visualization' / 'index.md'}")
    print(f"Summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
