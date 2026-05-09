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

from ce4_period_search.batch import (
    BatchConfig,
    BatchJob,
    default_batch_id,
    discover_input_files,
    ensure_run_ids,
    read_batch_manifest,
    run_batch,
)
from ce4_period_search.config import CWTSearchConfig, load_cwt_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CWT period-channel candidate search over multiple input files.")
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
    parser.add_argument("--period-min-records", type=float, default=None, help="Minimum CWT period in records.")
    parser.add_argument("--period-max-records", type=float, default=None, help="Maximum CWT period in records.")
    parser.add_argument("--period-count", type=int, default=None, help="Number of CWT periods.")
    parser.add_argument("--period-spacing", choices=["log", "linear"], default=None, help="CWT period spacing.")
    parser.add_argument("--time-aggregation", type=str, default=None, help="Time aggregation for period-channel response.")
    parser.add_argument("--aggregation-percentile", type=float, default=None, help="Percentile for percentile aggregation.")
    parser.add_argument("--block-channels", type=int, default=None, help="Frequency channels per block.")
    parser.add_argument("--threshold", type=float, default=None, help="Local robust S/N threshold.")
    parser.add_argument("--min-pixels", type=int, default=None, help="Minimum connected-component size.")
    parser.add_argument("--max-candidates-per-block", type=int, default=None, help="Cap components per block and level.")
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
    parser.add_argument("--viz-max-channels", type=int, default=None, help="Representative channels per block for period-time CWT plots.")
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
        "period_min_records",
        "period_max_records",
        "period_count",
        "period_spacing",
        "block_channels",
        "time_aggregation",
        "aggregation_percentile",
        "threshold",
        "min_pixels",
        "max_candidates_per_block",
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
    ]
    mapped = {
        "visualize": "visualization_enabled",
        "viz_max_blocks": "visualization_max_blocks",
        "viz_max_channels": "visualization_max_channels",
        "viz_top_candidates": "visualization_top_candidates",
        "viz_dpi": "visualization_dpi",
        "progress": "progress_enabled",
        "progress_leave": "progress_leave",
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
    print(f"Validation: {batch_dir / 'validation_reviewed.all.csv'}")


if __name__ == "__main__":
    main()
