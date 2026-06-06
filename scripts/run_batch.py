#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cwipss.batch import (
    BatchConfig,
    BatchJob,
    default_batch_id,
    discover_input_files,
    ensure_run_ids,
    read_batch_manifest,
    run_batch,
)
from cwipss.config import CWTSearchConfig, load_cwt_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CWT period-candidate search over multiple input files.")
    parser.add_argument("--config", type=Path, default=None, help="Base scan JSON config.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory to search for input files.")
    parser.add_argument("--pattern", type=str, default="*.2C", help="Input glob pattern for --input-dir.")
    parser.add_argument("--manifest", type=Path, default=None, help="Batch manifest CSV.")
    parser.add_argument("--inputs", nargs="*", default=None, help="Explicit input files.")
    parser.add_argument("--batch-id", type=str, default=None, help="Batch output directory id.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Base output directory. Defaults to runs/.")
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True, help="Run validation.")
    parser.add_argument("--stats", action=argparse.BooleanOptionalAction, default=True, help="Run statistics.")

    parser.add_argument("--f-start", type=float, default=None, help="Frequency/channel-coordinate start.")
    parser.add_argument("--f-stop", type=float, default=None, help="Frequency/channel-coordinate stop.")
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
    parser.add_argument("--time-aggregation", type=str, default=None, help="Time aggregation for period-channel response.")
    parser.add_argument("--aggregation-percentile", type=float, default=None, help="Percentile for percentile aggregation.")
    parser.add_argument("--block-channels", type=int, default=None, help="Frequency channels per block.")
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
    parser.add_argument("--pelt-jump-records", type=int, default=None, help="PELT endpoint/candidate grid stride in records; 1 keeps exact current behavior.")
    parser.add_argument("--pelt-threads", type=int, default=None, help="CPU worker threads for native PELT across channels; 1 keeps sequential behavior.")
    parser.add_argument("--cuda-structure-batch", action=argparse.BooleanOptionalAction, default=None, help="Batch CUDA structure/activity preprocessing across channel chunks.")
    parser.add_argument("--cuda-structure-batch-channels", type=int, default=None, help="Channels per CUDA structure/activity batch chunk; omit to process the full block at once.")
    parser.add_argument("--cuda-max-pending-blocks", type=int, default=None, help="Prepared CUDA blocks retained while CPU PELT runs; 1 is sequential, 2 overlaps one following block.")
    parser.add_argument("--window-min-duration-records", type=int, default=None, help="Minimum retained PELT window duration.")
    parser.add_argument("--window-min-activity-mean", type=float, default=None, help="Minimum retained standardized activity mean.")
    parser.add_argument("--window-min-activity-raw-mean", type=float, default=None, help="Minimum retained raw structured activity mean before robust standardization.")
    parser.add_argument("--window-merge-gap-records", type=int, default=None, help="Merge PELT windows separated by at most this gap.")
    parser.add_argument("--profile-min-prominence", type=float, default=None, help="Minimum windowed period-profile peak prominence.")
    parser.add_argument("--profile-max-peaks-per-window", type=int, default=None, help="Maximum period peaks retained per time window.")
    parser.add_argument("--candidate-period-min-records", type=float, default=None, help="Reject candidates below this period in records.")
    parser.add_argument("--candidate-period-max-records", type=float, default=None, help="Reject candidates above this period in records.")
    parser.add_argument("--max-candidates-per-channel", type=str, default=None, help="Hard cap per channel, or 'auto' to use --max-candidates-per-record.")
    parser.add_argument("--max-candidates-per-record", type=float, default=None, help="Per-record candidate rate used when --max-candidates-per-channel auto.")
    parser.add_argument("--validation-max-candidates", type=int, default=None, help="Maximum candidates to validate per file.")
    parser.add_argument("--validation-shuffle-trials", type=int, default=None, help="Shuffle/null trials per candidate.")
    parser.add_argument(
        "--validation-include-vetoed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Validate candidates already marked vetoed.",
    )
    parser.add_argument("--visualize", action="store_true", default=None, help="Write staged visualization PNGs for each file run.")
    parser.add_argument("--viz-max-blocks", type=int, default=None, help="Maximum blocks to visualize per file; 0 means all.")
    parser.add_argument("--viz-max-channels", type=int, default=None, help="Representative channels per block for period-time CWT plots; 0 means all channels.")
    parser.add_argument("--viz-top-candidates", type=int, default=None, help="Maximum candidate overlays/scatter points.")
    parser.add_argument("--viz-dpi", type=int, default=None, help="Visualization image DPI.")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show CWT channel-progress tqdm for each file.",
    )
    parser.add_argument(
        "--progress-leave",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Leave CWT progress bars on screen when each file finishes.",
    )
    parser.add_argument(
        "--timing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print per-block timing diagnostics for each file run.",
    )
    return parser.parse_args()


def _output_dir(path: Path | None) -> Path:
    output_dir = Path("runs") if path is None else path
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    return output_dir


def _scan_overrides(args: argparse.Namespace) -> dict:
    names = [
        "f_start",
        "f_stop",
        "t_start",
        "t_stop",
        "wavelet",
        "cwt_method",
        "cwt_backend",
        "cuda_device",
        "period_min_records",
        "period_max_records",
        "period_count",
        "period_spacing",
        "block_channels",
        "time_aggregation",
        "aggregation_percentile",
        "noise_floor_fraction",
        "excess_eps_fraction",
        "structure_baseline_quantile",
        "structure_scale_quantile",
        "structure_z_threshold",
        "structure_time_support_records",
        "structure_period_support_bins",
        "structure_min_support_fraction",
        "activity_trim_low",
        "activity_trim_high",
        "activity_smooth_records",
        "pelt_penalty",
        "pelt_min_size_records",
        "pelt_jump_records",
        "pelt_threads",
        "cuda_structure_batch",
        "cuda_structure_batch_channels",
        "cuda_max_pending_blocks",
        "window_min_duration_records",
        "window_min_activity_mean",
        "window_min_activity_raw_mean",
        "window_merge_gap_records",
        "profile_min_prominence",
        "profile_max_peaks_per_window",
        "candidate_period_min_records",
        "candidate_period_max_records",
        "max_candidates_per_channel",
        "max_candidates_per_record",
        "validation_max_candidates",
        "validation_shuffle_trials",
        "validation_include_vetoed",
        "visualization_enabled",
        "visualization_max_blocks",
        "visualization_max_channels",
        "visualization_top_candidates",
        "visualization_dpi",
        "progress_enabled",
        "progress_leave",
        "timing_enabled",
    ]
    mapped = {
        "visualize": "visualization_enabled",
        "viz_max_blocks": "visualization_max_blocks",
        "viz_max_channels": "visualization_max_channels",
        "viz_top_candidates": "visualization_top_candidates",
        "viz_dpi": "visualization_dpi",
        "progress": "progress_enabled",
        "progress_leave": "progress_leave",
        "timing": "timing_enabled",
    }
    values = vars(args)
    overrides = {name: values[name] for name in names if name in values and values[name] is not None}
    for source, target in mapped.items():
        if values.get(source) is not None:
            overrides[target] = values[source]
    return overrides


def _load_base_config(args: argparse.Namespace) -> CWTSearchConfig:
    config = load_cwt_config(args.config, overrides=_scan_overrides(args))
    return replace(config, input=None, run_id=None)


def _collect_jobs(args: argparse.Namespace) -> list[BatchJob]:
    jobs: list[BatchJob] = []
    if args.manifest is not None:
        jobs.extend(read_batch_manifest(args.manifest))
    inputs: list[str] = []
    if args.input_dir is not None:
        inputs.extend(discover_input_files(args.input_dir, pattern=args.pattern, project_dir=PROJECT_DIR))
    if args.inputs:
        inputs.extend(args.inputs)
    jobs.extend(BatchJob(input=value) for value in inputs)
    if not jobs:
        raise SystemExit("Provide --input-dir, --manifest, or --inputs")
    return ensure_run_ids(jobs)


def main() -> None:
    args = parse_args()
    batch_config = BatchConfig(
        batch_id=args.batch_id or default_batch_id(),
        output_dir=str(_output_dir(args.output_dir)),
        validate=bool(args.validate),
        stats=bool(args.stats),
    )
    batch_dir = run_batch(
        jobs=_collect_jobs(args),
        base_config=_load_base_config(args),
        batch_config=batch_config,
        project_dir=PROJECT_DIR,
    )
    print(f"Batch directory: {batch_dir}")
    print(f"Manifest: {batch_dir / 'manifest.csv'}")
    print(f"Candidates: {batch_dir / 'candidates_reviewed.all.csv'}")
    print(f"Time windows: {batch_dir / 'time_windows.all.csv'}")
    print(f"Validation: {batch_dir / 'validation_reviewed.all.csv'}")


if __name__ == "__main__":
    main()
