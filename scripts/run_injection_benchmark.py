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

from cwipss.benchmark import (
    CWTBenchmarkConfig,
    MatchConfig,
    make_background_from_args,
    run_injection_benchmark,
)
from cwipss.injection_config import load_injection_config, make_injections_from_config
from cwipss.validation import ValidationConfig
from cwipss.visualization import CWTVisualizationConfig
from cwipss.veto import VetoConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic/injected CWT period-search benchmark.")
    parser.add_argument(
        "--background",
        choices=["synthetic", "ce4"],
        default="synthetic",
        help="Background source; ce4 means the currently supported CE4 .2C/.2CL data format.",
    )
    parser.add_argument("--input", type=str, default=None, help="CE4 .2C input file for --background ce4.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to runs/injection_<run-id>.")
    parser.add_argument("--run-id", type=str, default="injection_benchmark", help="Run id written to output tables.")

    parser.add_argument("--records", type=int, default=1024, help="Synthetic record count.")
    parser.add_argument("--channels", type=int, default=64, help="Synthetic channel count.")
    parser.add_argument("--noise-std", type=float, default=1.0, help="Synthetic Gaussian noise sigma.")
    parser.add_argument("--seed", type=int, default=12345, help="Synthetic noise RNG seed.")
    parser.add_argument("--f-start", type=float, default=None, help="Frequency/channel-coordinate start.")
    parser.add_argument("--f-stop", type=float, default=None, help="Frequency/channel-coordinate stop.")
    parser.add_argument("--t-start", type=int, default=None, help="Record start for CE4-format background.")
    parser.add_argument("--t-stop", type=int, default=None, help="Record stop for CE4-format background.")

    parser.add_argument("--injection-config", type=Path, required=True, help="JSON config for simulation injections.")

    parser.add_argument("--wavelet", type=str, default="cmor1.5-1.0", help="PyWavelets CWT wavelet.")
    parser.add_argument("--cwt-method", choices=["conv", "fft"], default="fft", help="PyWavelets CWT computation method.")
    parser.add_argument("--cwt-backend", choices=["cpu", "cuda", "auto"], default="cpu", help="CWT compute backend.")
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device index for --cwt-backend cuda/auto.")
    parser.add_argument("--period-min-records", type=float, default=2.0, help="Minimum CWT period in records.")
    parser.add_argument("--period-max-records", type=float, default=512.0, help="Maximum CWT period in records.")
    parser.add_argument("--period-count", type=int, default=96, help="Number of CWT periods.")
    parser.add_argument("--period-spacing", choices=["log", "linear"], default="log", help="CWT period spacing.")
    parser.add_argument("--block-channels", type=int, default=128, help="Frequency channels per block.")
    parser.add_argument("--time-aggregation", type=str, default="p95", help="Time aggregation for period-channel response.")
    parser.add_argument("--aggregation-percentile", type=float, default=95.0, help="Percentile for percentile aggregation.")
    parser.add_argument("--noise-floor-fraction", type=float, default=0.20, help="Lowest fraction of valid CWT power used as channel noise floor.")
    parser.add_argument("--excess-eps-fraction", type=float, default=1e-6, help="Fractional epsilon added to the noise floor.")
    parser.add_argument("--structure-baseline-quantile", type=float, default=0.10, help="Low time-quantile background for per-period structure z-score.")
    parser.add_argument("--structure-scale-quantile", type=float, default=0.20, help="Low time-quantile subset used to estimate per-period structure scale.")
    parser.add_argument("--structure-z-threshold", type=float, default=1.0, help="Per-period robust z threshold for 2D CWT structure support.")
    parser.add_argument("--structure-time-support-records", type=int, default=64, help="Time-neighborhood width for 2D structure support.")
    parser.add_argument("--structure-period-support-bins", type=int, default=3, help="Period-neighborhood width for 2D structure support.")
    parser.add_argument("--structure-min-support-fraction", type=float, default=0.10, help="Minimum local 2D support fraction before CWT texture is retained.")
    parser.add_argument("--activity-trim-low", type=float, default=0.05, help="Lower period-axis trim fraction for signed activity.")
    parser.add_argument("--activity-trim-high", type=float, default=0.95, help="Upper period-axis trim fraction for signed activity.")
    parser.add_argument("--activity-smooth-records", type=int, default=16, help="Moving-average width for activity curves.")
    parser.add_argument("--pelt-penalty", type=float, default=16.0, help="PELT mean-shift penalty.")
    parser.add_argument("--pelt-min-size-records", type=int, default=384, help="PELT minimum segment size.")
    parser.add_argument("--pelt-jump-records", type=int, default=1, help="PELT endpoint/candidate grid stride in records; 1 keeps exact current behavior.")
    parser.add_argument("--pelt-threads", type=int, default=1, help="CPU worker threads for native PELT across channels; 1 keeps sequential behavior.")
    parser.add_argument("--window-min-duration-records", type=int, default=384, help="Minimum retained PELT window duration.")
    parser.add_argument("--window-min-activity-mean", type=float, default=0.05, help="Minimum retained standardized activity mean.")
    parser.add_argument("--window-min-activity-raw-mean", type=float, default=25.0, help="Minimum retained raw structured activity mean before robust standardization.")
    parser.add_argument("--window-merge-gap-records", type=int, default=256, help="Merge PELT windows separated by at most this gap.")
    parser.add_argument("--profile-min-prominence", type=float, default=0.5, help="Minimum windowed period-profile peak prominence.")
    parser.add_argument("--profile-max-peaks-per-window", type=int, default=1, help="Maximum period peaks retained per time window.")
    parser.add_argument("--candidate-period-min-records", type=float, default=10.0, help="Reject candidates below this period in records.")
    parser.add_argument("--candidate-period-max-records", type=float, default=200.0, help="Reject candidates above this period in records.")
    parser.add_argument("--max-candidates-per-channel", type=str, default="auto", help="Hard cap per channel, or 'auto' to use --max-candidates-per-record.")
    parser.add_argument("--max-candidates-per-record", type=float, default=3.0 / 4096.0, help="Per-record candidate rate used when --max-candidates-per-channel auto.")
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
    parser.add_argument("--viz-max-channels", type=int, default=4, help="Representative channels per block for period-time CWT plots; 0 means all channels.")
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
        cwt_backend=args.cwt_backend,
        cuda_device=args.cuda_device,
        period_min_records=args.period_min_records,
        period_max_records=args.period_max_records,
        period_count=args.period_count,
        period_spacing=args.period_spacing,
        block_channels=args.block_channels,
        time_aggregation=args.time_aggregation,
        aggregation_percentile=args.aggregation_percentile,
        noise_floor_fraction=args.noise_floor_fraction,
        excess_eps_fraction=args.excess_eps_fraction,
        structure_baseline_quantile=args.structure_baseline_quantile,
        structure_scale_quantile=args.structure_scale_quantile,
        structure_z_threshold=args.structure_z_threshold,
        structure_time_support_records=args.structure_time_support_records,
        structure_period_support_bins=args.structure_period_support_bins,
        structure_min_support_fraction=args.structure_min_support_fraction,
        activity_trim_low=args.activity_trim_low,
        activity_trim_high=args.activity_trim_high,
        activity_smooth_records=args.activity_smooth_records,
        pelt_penalty=args.pelt_penalty,
        pelt_min_size_records=args.pelt_min_size_records,
        pelt_jump_records=args.pelt_jump_records,
        pelt_threads=args.pelt_threads,
        window_min_duration_records=args.window_min_duration_records,
        window_min_activity_mean=args.window_min_activity_mean,
        window_min_activity_raw_mean=args.window_min_activity_raw_mean,
        window_merge_gap_records=args.window_merge_gap_records,
        profile_min_prominence=args.profile_min_prominence,
        profile_max_peaks_per_window=args.profile_max_peaks_per_window,
        candidate_period_min_records=args.candidate_period_min_records,
        candidate_period_max_records=args.candidate_period_max_records,
        max_candidates_per_channel=args.max_candidates_per_channel,
        max_candidates_per_record=args.max_candidates_per_record,
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
    print(f"Time windows: {output_dir / 'time_windows.csv'}")
    print(f"Results: {output_dir / 'injection_results.csv'}")
    print(f"Performance: {output_dir / 'injection_performance.csv'}")
    if args.visualize:
        print(f"Visualization: {output_dir / 'visualization' / 'index.md'}")
    print(f"Summary: {output_dir / 'injection_summary.json'}")


if __name__ == "__main__":
    main()
