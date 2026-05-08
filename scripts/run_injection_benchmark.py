#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ce4_period_search.benchmark import (
    MatchConfig,
    MatrixSearchConfig,
    make_background_from_args,
    make_default_injections,
    run_injection_benchmark,
)
from ce4_period_search.validation import ValidationConfig
from ce4_period_search.visualization import VisualizationConfig
from ce4_period_search.veto import VetoConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic/injected SWT period-search benchmark.")
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

    parser.add_argument("--period-records", type=float, nargs="+", default=[16.0], help="Injected periods in records.")
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[6.0], help="Injected amplitudes.")
    parser.add_argument("--grid", action="store_true", help="Inject the full period x amplitude grid.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat each injection setting.")
    parser.add_argument(
        "--signal-model",
        choices=[
            "pulsed_periodic",
            "intermittent_periodic",
            "sinusoidal_narrowband",
            "band_limited_periodic",
            "drifting_ridge",
        ],
        default="pulsed_periodic",
        help="Injected signal morphology.",
    )

    parser.add_argument("--wavelet", type=str, default="db4", help="SWT wavelet.")
    parser.add_argument("--levels", type=int, default=5, help="SWT decomposition levels.")
    parser.add_argument("--block-channels", type=int, default=128, help="Frequency channels per block.")
    parser.add_argument("--threshold", type=float, default=5.0, help="Local robust S/N threshold.")
    parser.add_argument("--min-pixels", type=int, default=12, help="Minimum connected-component size.")
    parser.add_argument("--local-time", type=int, default=513, help="Median/MAD local time window.")
    parser.add_argument("--local-freq", type=int, default=9, help="Median/MAD local frequency window.")
    parser.add_argument("--max-candidates-per-block", type=int, default=200, help="Candidate cap per block and level.")

    parser.add_argument("--validation-max-candidates", type=int, default=50, help="Maximum candidates to validate.")
    parser.add_argument("--validation-window-periods", type=int, default=128, help="Validation window in periods.")
    parser.add_argument("--validation-min-window-records", type=int, default=256, help="Minimum validation window.")
    parser.add_argument("--validation-max-window-records", type=int, default=4096, help="Maximum validation window.")
    parser.add_argument("--validation-radius", type=float, default=2.0, help="Period-search radius around SWT scale.")
    parser.add_argument("--validation-fold-bins", type=int, default=16, help="Fold profile bin count.")
    parser.add_argument("--validation-shuffle-trials", type=int, default=100, help="Shuffle/null trials.")
    parser.add_argument("--validation-include-vetoed", action="store_true", help="Validate vetoed candidates too.")

    parser.add_argument("--min-time-overlap", type=float, default=0.30, help="Truth match time-overlap threshold.")
    parser.add_argument("--min-freq-overlap", type=float, default=0.30, help="Truth match frequency-overlap threshold.")
    parser.add_argument("--max-period-error", type=float, default=0.50, help="Validated period fractional error threshold.")

    parser.add_argument("--visualize", action="store_true", help="Write staged visualization PNGs and index.md.")
    parser.add_argument("--viz-max-blocks", type=int, default=2, help="Maximum blocks to visualize; 0 means all.")
    parser.add_argument("--viz-max-levels", type=int, default=3, help="Maximum SWT levels per block to visualize; 0 means all.")
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
    injections = make_default_injections(
        periods=list(args.period_records),
        amplitudes=list(args.amplitudes),
        records=background.data.shape[0],
        channels=background.data.shape[1],
        model=args.signal_model,
        grid=args.grid,
        repeats=args.repeats,
    )
    search_config = MatrixSearchConfig(
        wavelet=args.wavelet,
        levels=args.levels,
        block_channels=args.block_channels,
        threshold=args.threshold,
        min_pixels=args.min_pixels,
        local_time=args.local_time,
        local_freq=args.local_freq,
        max_candidates_per_block=args.max_candidates_per_block,
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
        output_dir=_output_dir(args),
        run_id=args.run_id,
        search_config=search_config,
        veto_config=VetoConfig(),
        validation_config=validation_config,
        match_config=match_config,
        visualization_config=VisualizationConfig(
            enabled=bool(args.visualize),
            max_blocks=args.viz_max_blocks,
            max_levels=args.viz_max_levels,
            top_candidates=args.viz_top_candidates,
            dpi=args.viz_dpi,
        ),
    )
    print(f"Injection benchmark directory: {output_dir}")
    print(f"Truth: {output_dir / 'injection_truth.csv'}")
    print(f"Results: {output_dir / 'injection_results.csv'}")
    print(f"Performance: {output_dir / 'injection_performance.csv'}")
    if args.visualize:
        print(f"Visualization: {output_dir / 'visualization' / 'index.md'}")
    print(f"Summary: {output_dir / 'injection_summary.json'}")


if __name__ == "__main__":
    main()
