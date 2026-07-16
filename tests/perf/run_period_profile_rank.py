#!/usr/bin/env python3
"""Run period-profile rank behind a frozen activity algorithm and native PELT."""

from __future__ import annotations

import argparse
from pathlib import Path

from period_profile_benchmark import (
    PROJECT_DIR,
    PeriodProfileBenchmarkConfig,
    run_period_profile_benchmark,
)


def _channels(value: str) -> tuple[int, ...]:
    if value.strip().lower() == "all":
        return tuple(range(2048))
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one negative channel is required")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "runs/period_profile_rank_initial",
    )
    parser.add_argument(
        "--activity-algorithm",
        default="sm_cpro_w769",
        help="Frozen strict single-map activity algorithm used before mandatory PELT.",
    )
    parser.add_argument("--candidate-period-max-records", type=float, default=1000.0)
    parser.add_argument(
        "--stage3-min-window-records",
        type=int,
        default=96,
        help="Reject PELT windows shorter than this before period-axis evaluation.",
    )
    parser.add_argument("--profile-threshold-snr", type=float, default=32.0)
    parser.add_argument("--profile-texture-quantile", type=float, default=0.9375)
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--injections",
        type=Path,
        default=PROJECT_DIR / "configs/injection_lowfreq_random_100.json",
    )
    parser.add_argument(
        "--cwt-config",
        type=Path,
        default=PROJECT_DIR / "configs/cwt_default.json",
    )
    parser.add_argument("--max-positive-cases", type=int, default=0)
    parser.add_argument(
        "--negative-channels",
        type=_channels,
        default=(3, 4, 5, 6, 7, 8, 9, 10),
        help="Comma-separated real no-injection CE4 channels, or 'all' for all 2048 channels.",
    )
    parser.add_argument("--max-negative-windows-per-channel", type=int, default=0)
    parser.add_argument("--algorithm", action="append", default=[])
    parser.add_argument("--cwt-backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_period_profile_benchmark(
        PeriodProfileBenchmarkConfig(
            output_dir=args.output,
            input_path=args.input,
            injection_config=args.injections,
            cwt_config=args.cwt_config,
            activity_algorithm=args.activity_algorithm,
            candidate_period_max_records=args.candidate_period_max_records,
            stage3_min_window_records=args.stage3_min_window_records,
            profile_threshold_snr=args.profile_threshold_snr,
            profile_texture_quantile=args.profile_texture_quantile,
            algorithms=tuple(args.algorithm),
            max_positive_cases=args.max_positive_cases,
            negative_channel_indices=tuple(args.negative_channels),
            max_negative_windows_per_channel=args.max_negative_windows_per_channel,
            cwt_backend=args.cwt_backend,
            cuda_device=args.cuda_device,
            progress_every=args.progress_every,
        )
    )
    print(f"Cases: {result.cases_csv}")
    print(f"Rank: {result.summary_csv}")
    print(f"Model strata: {result.model_summary_csv}")
    print(f"Summary: {result.summary_json}")


if __name__ == "__main__":
    main()
