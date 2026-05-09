#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ce4_period_search.benchmark import (
    CWTBenchmarkConfig,
    MatchConfig,
    make_background_from_args,
    run_injection_benchmark,
)
from ce4_period_search.injection_config import load_injection_config, make_injections_from_config
from ce4_period_search.validation import ValidationConfig
from ce4_period_search.visualization import CWTVisualizationConfig
from ce4_period_search.veto import VetoConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic/injected CWT period-search benchmark.")
    parser.add_argument("--background", choices=["synthetic", "ce4"], default="synthetic", help="Background source.")
    parser.add_argument("--input", type=str, default=None, help="Input file for --background ce4.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to runs/injection_<run-id>.")
    parser.add_argument("--run-id", type=str, default="injection_benchmark", help="Run id written to output tables.")

    parser.add_argument("--records", type=int, default=1024, help="Synthetic record count.")
    parser.add_argument("--channels", type=int, default=64, help="Synthetic channel count.")
    parser.add_argument("--noise-std", type=float, default=1.0, help="Synthetic Gaussian noise sigma.")
    parser.add_argument("--seed", type=int, default=12345, help="Synthetic noise RNG seed.")
    parser.add_argument("--f-start", type=float, default=None, help="Frequency/channel-coordinate start.")
    parser.add_argument("--f-stop", type=float, default=None, help="Frequency/channel-coordinate stop.")
    parser.add_argument("--t-start", type=int, default=None, help="Record start for CE-4 background.")
    parser.add_argument("--t-stop", type=int, default=None, help="Record stop for CE-4 background.")

    parser.add_argument("--injection-config", type=Path, required=True, help="JSON config for simulation injections.")

    parser.add_argument("--wavelet", type=str, default="cmor1.5-1.0", help="PyWavelets CWT wavelet.")
    parser.add_argument("--cwt-method", choices=["conv", "fft"], default="fft", help="PyWavelets CWT computation method.")
    parser.add_argument("--period-min-records", type=float, default=2.0, help="Minimum CWT period in records.")
    parser.add_argument("--period-max-records", type=float, default=512.0, help="Maximum CWT period in records.")
    parser.add_argument("--period-count", type=int, default=96, help="Number of CWT periods.")
    parser.add_argument("--period-spacing", choices=["log", "linear"], default="log", help="CWT period spacing.")
    parser.add_argument("--block-channels", type=int, default=128, help="Frequency channels per block.")
    parser.add_argument("--time-aggregation", type=str, default="p95", help="Time aggregation for period-channel response.")
    parser.add_argument("--aggregation-percentile", type=float, default=95.0, help="Percentile for percentile aggregation.")
    parser.add_argument("--threshold", type=float, default=2.5, help="Minimum per-channel scalogram region score.")
    parser.add_argument("--dog-sigma-peak", type=float, default=1.0, help="Narrow period-axis Gaussian sigma for scalogram DoG.")
    parser.add_argument("--dog-sigma-background", type=float, default=10.0, help="Broad period-axis Gaussian sigma for scalogram DoG.")
    parser.add_argument("--time-smooth-sigma", type=float, default=1.0, help="Time-axis smoothing sigma for scalogram region scoring.")
    parser.add_argument("--min-duration-records", type=int, default=8, help="Minimum candidate time length in records.")
    parser.add_argument("--min-width-bins", type=float, default=1.0, help="Minimum candidate period width in bins.")
    parser.add_argument("--max-width-bins", type=float, default=10.0, help="Maximum candidate period width in bins.")
    parser.add_argument("--max-candidates-per-channel", type=int, default=2, help="Candidate cap per frequency channel.")
    parser.add_argument("--max-candidates-per-block", type=int, default=50, help="Candidate cap per block.")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show CWT channel-progress tqdm.",
    )
    parser.add_argument(
        "--progress-leave",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Leave the tqdm progress bar on screen when finished.",
    )

    parser.add_argument("--validation-max-candidates", type=int, default=25, help="Maximum candidates to validate.")
    parser.add_argument("--validation-window-periods", type=int, default=128, help="Validation window in periods.")
    parser.add_argument("--validation-min-window-records", type=int, default=256, help="Minimum validation window.")
    parser.add_argument("--validation-max-window-records", type=int, default=4096, help="Maximum validation window.")
    parser.add_argument("--validation-radius", type=float, default=2.0, help="Period-search radius around CWT candidate period.")
    parser.add_argument("--validation-fold-bins", type=int, default=16, help="Fold profile bin count.")
    parser.add_argument("--validation-shuffle-trials", type=int, default=100, help="Shuffle/null trials.")
    parser.add_argument("--validation-include-vetoed", action="store_true", help="Validate vetoed candidates too.")

    parser.add_argument("--min-time-overlap", type=float, default=0.30, help="Truth match time-overlap threshold.")
    parser.add_argument("--min-freq-overlap", type=float, default=0.30, help="Truth match frequency-overlap threshold.")
    parser.add_argument("--max-period-error", type=float, default=0.50, help="Validated period fractional error threshold.")

    parser.add_argument("--visualize", action="store_true", help="Write staged visualization PNGs and index.md.")
    parser.add_argument("--viz-max-blocks", type=int, default=2, help="Maximum blocks to visualize; 0 means all.")
    parser.add_argument("--viz-max-channels", type=int, default=4, help="Representative channels per block for period-time CWT plots.")
    parser.add_argument("--viz-top-candidates", type=int, default=50, help="Maximum candidate overlays/scatter points.")
    parser.add_argument("--viz-dpi", type=int, default=140, help="Visualization image DPI.")
    return parser.parse_args()


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = Path("runs") / f"injection_{args.run_id}"
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = _output_dir(args)
    background = make_background_from_args(
        mode=args.background,
        input_path=args.input,
        records=args.records,
        channels=args.channels,
        seed=args.seed,
        noise_std=args.noise_std,
        f_start=args.f_start,
        f_stop=args.f_stop,
        t_start=args.t_start,
        t_stop=args.t_stop,
    )
    injection_payload = load_injection_config(args.injection_config)
    injections = make_injections_from_config(
        injection_payload,
        records=background.data.shape[0],
        channels=background.data.shape[1],
        freqs_mhz=background.freqs_mhz,
        default_seed=args.seed,
    )
    search_config = CWTBenchmarkConfig(
        wavelet=args.wavelet,
        cwt_method=args.cwt_method,
        period_min_records=args.period_min_records,
        period_max_records=args.period_max_records,
        period_count=args.period_count,
        period_spacing=args.period_spacing,
        block_channels=args.block_channels,
        time_aggregation=args.time_aggregation,
        aggregation_percentile=args.aggregation_percentile,
        threshold=args.threshold,
        dog_sigma_peak=args.dog_sigma_peak,
        dog_sigma_background=args.dog_sigma_background,
        time_smooth_sigma=args.time_smooth_sigma,
        min_duration_records=args.min_duration_records,
        min_width_bins=args.min_width_bins,
        max_width_bins=args.max_width_bins,
        max_candidates_per_channel=args.max_candidates_per_channel,
        max_candidates_per_block=args.max_candidates_per_block,
        progress_enabled=args.progress,
        progress_leave=args.progress_leave,
    )
    validation_config = ValidationConfig(
        include_vetoed=args.validation_include_vetoed,
        max_candidates=args.validation_max_candidates,
        window_periods=args.validation_window_periods,
        min_window_records=args.validation_min_window_records,
        max_window_records=args.validation_max_window_records,
        period_search_radius=args.validation_radius,
        fold_bins=args.validation_fold_bins,
        shuffle_trials=args.validation_shuffle_trials,
        random_seed=args.seed,
    )
    match_config = MatchConfig(
        min_time_overlap=args.min_time_overlap,
        min_freq_overlap=args.min_freq_overlap,
        max_period_error_fraction=args.max_period_error,
    )
    output_dir = run_injection_benchmark(
        background=background,
        injections=injections,
        output_dir=output_dir,
        run_id=args.run_id,
        search_config=search_config,
        veto_config=VetoConfig(),
        validation_config=validation_config,
        match_config=match_config,
        visualization_config=CWTVisualizationConfig(
            enabled=bool(args.visualize),
            max_blocks=args.viz_max_blocks,
            max_channels=args.viz_max_channels,
            top_candidates=args.viz_top_candidates,
            dpi=args.viz_dpi,
        ),
    )
    (output_dir / "injection_config.json").write_text(json.dumps(injection_payload, indent=2, ensure_ascii=True))
    summary_path = output_dir / "injection_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary["injection_config"] = {
        "source_path": str(args.injection_config),
        "resolved_path": str(output_dir / "injection_config.json"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Injection benchmark directory: {output_dir}")
    print(f"Truth: {output_dir / 'injection_truth.csv'}")
    print(f"Results: {output_dir / 'injection_results.csv'}")
    print(f"Performance: {output_dir / 'injection_performance.csv'}")
    if args.visualize:
        print(f"Visualization: {output_dir / 'visualization' / 'index.md'}")
    print(f"Summary: {output_dir / 'injection_summary.json'}")


if __name__ == "__main__":
    main()
