#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cwipss import run_cwt_search
from cwipss.config import CWTSearchConfig, load_cwt_config, resolve_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CWT period-candidate detection.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config.")
    parser.add_argument("--input", type=str, default=None, help="Input data path supported by the active reader.")
    parser.add_argument("--f-start", type=float, default=None, help="Frequency start in MHz.")
    parser.add_argument("--f-stop", type=float, default=None, help="Frequency stop in MHz.")
    parser.add_argument("--t-start", type=int, default=None, help="Record start index.")
    parser.add_argument("--t-stop", type=int, default=None, help="Record stop index.")
    parser.add_argument("--wavelet", type=str, default=None, help="PyWavelets CWT wavelet.")
    parser.add_argument("--cwt-method", choices=["conv", "fft"], default=None, help="PyWavelets CWT computation method.")
    parser.add_argument("--cwt-backend", choices=["cpu", "cuda", "auto"], default=None, help="CWT compute backend.")
    parser.add_argument("--cuda-device", type=int, default=None, help="CUDA device index for --cwt-backend cuda/auto.")
    parser.add_argument("--period-min-records", type=float, default=None, help="Minimum CWT period in records.")
    parser.add_argument("--period-max-records", type=float, default=None, help="Maximum CWT period in records.")
    parser.add_argument("--period-count", type=int, default=None, help="Number of CWT periods.")
    parser.add_argument("--period-spacing", choices=["log", "linear"], default=None, help="CWT period spacing.")
    parser.add_argument("--block-channels", type=int, default=None, help="Frequency channels per block.")
    parser.add_argument("--time-aggregation", type=str, default=None, help="Time aggregation for period-channel response.")
    parser.add_argument("--aggregation-percentile", type=float, default=None, help="Percentile for percentile aggregation.")
    parser.add_argument("--noise-floor-fraction", type=float, default=None, help="Lowest fraction of valid CWT power used as channel noise floor.")
    parser.add_argument("--excess-eps-fraction", type=float, default=None, help="Fractional epsilon added to the noise floor.")
    parser.add_argument("--structure-baseline-quantile", type=float, default=None, help="Low time-quantile background for per-period structure z-score.")
    parser.add_argument("--structure-scale-quantile", type=float, default=None, help="Low time-quantile subset used to estimate per-period structure scale.")
    parser.add_argument("--structure-z-threshold", type=float, default=None, help="Per-period robust z threshold for 2D CWT structure support.")
    parser.add_argument("--structure-time-support-records", type=int, default=None, help="Time-neighborhood width for 2D structure support.")
    parser.add_argument("--structure-period-support-bins", type=int, default=None, help="Period-neighborhood width for 2D structure support.")
    parser.add_argument("--structure-min-support-fraction", type=float, default=None, help="Minimum local 2D support fraction before CWT texture is retained.")
    parser.add_argument("--activity-trim-low", type=float, default=None, help="Lower period-axis trim fraction for signed activity.")
    parser.add_argument("--activity-trim-high", type=float, default=None, help="Upper period-axis trim fraction for signed activity.")
    parser.add_argument("--activity-smooth-records", type=int, default=None, help="Moving-average width for activity curves.")
    parser.add_argument("--pelt-penalty", type=float, default=None, help="PELT mean-shift penalty.")
    parser.add_argument("--pelt-min-size-records", type=int, default=None, help="PELT minimum segment size.")
    parser.add_argument("--window-min-duration-records", type=int, default=None, help="Minimum retained PELT window duration.")
    parser.add_argument("--window-min-activity-mean", type=float, default=None, help="Minimum retained standardized activity mean.")
    parser.add_argument("--window-min-activity-raw-mean", type=float, default=None, help="Minimum retained raw structured activity mean before robust standardization.")
    parser.add_argument("--window-merge-gap-records", type=int, default=None, help="Merge PELT windows separated by at most this gap.")
    parser.add_argument("--profile-min-prominence", type=float, default=None, help="Minimum windowed period-profile peak prominence.")
    parser.add_argument("--profile-max-peaks-per-window", type=int, default=None, help="Maximum period peaks retained per time window.")
    parser.add_argument("--candidate-period-min-records", type=float, default=None, help="Reject candidates below this period in records.")
    parser.add_argument("--candidate-period-max-records", type=float, default=None, help="Reject candidates above this period in records.")
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
    parser.add_argument("--viz-max-channels", type=int, default=None, help="Representative channels per block for period-time CWT plots; 0 means all channels.")
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
    parser.add_argument(
        "--timing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print per-block CWT pipeline timing diagnostics.",
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
            "timing": "timing_enabled",
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
    print(f"Time windows: {run_dir / 'time_windows.csv'}")
    print(f"Candidates: {run_dir / 'candidates_raw.csv'}")
    if config.visualization_enabled:
        print(f"Visualization: {run_dir / 'visualization' / 'index.md'}")
    print(f"Summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
