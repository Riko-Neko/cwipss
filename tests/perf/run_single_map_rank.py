#!/usr/bin/env python3
"""Run a reproducible strict single-CWT-map activity ranking experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from compression_config_rank import largest_complete_2c
from cwt_activity_algorithms import resolve_cwt_activity_algorithms
from cwt_activity_rank import CWTActivityRun, run_cwt_activity_rank
from single_map_activity_algorithms import (
    single_map_absolute_persistence_names,
    single_map_algorithm_names,
    single_map_false_window_names,
    single_map_false_window_refinement_names,
    single_map_ridge_refinement_names,
    single_map_sparse_algorithm_names,
    single_map_cpro_names,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
LINEAR_SINGLE_MAP_CONTROLS = (
    "raw_max_power_ratio",
    "row_mad_max_z",
    "row_mad_topk_mean",
    "row_mad_topk_ratio",
    "row_mad_horizontal_filter",
    "ridge_cfar_no_time",
    "ridge_cfar_2cycle",
    "ridge_cfar_4cycle",
    "ridge_cfar_6cycle",
    "ridge_cfar_wide_4cycle",
    "spectral_entropy_deficit",
    "js_background_divergence",
    "row_mad_kurtosis_window",
    "connected_component_mass",
    "hough_horizontal_vote",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--injections", type=Path, default=PROJECT_DIR / "configs/injection_lowfreq_random_100.json")
    parser.add_argument("--cwt-config", type=Path, default=PROJECT_DIR / "configs/cwt_default.json")
    parser.add_argument("--algorithms", type=str, default="all-single-map")
    parser.add_argument("--max-groups-per-family", type=int, default=0)
    parser.add_argument("--negative-max-channels", type=int, default=0)
    parser.add_argument(
        "--negative-channels",
        type=str,
        default="",
        help="Comma-separated channel indices for targeted PELT stress screening.",
    )
    parser.add_argument("--negative-window-method", choices=("pelt",), default="pelt")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--pelt-threads", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    negative_channels = tuple(
        int(value.strip()) for value in args.negative_channels.split(",") if value.strip()
    )
    input_path = args.input or largest_complete_2c(PROJECT_DIR / "data/CE4")
    if args.algorithms.strip() == "all-single-map":
        names = (*single_map_algorithm_names(), *LINEAR_SINGLE_MAP_CONTROLS)
    elif args.algorithms.strip() == "sparse-single-map":
        names = (*single_map_sparse_algorithm_names(), *LINEAR_SINGLE_MAP_CONTROLS)
    elif args.algorithms.strip() == "ridge-refinement":
        names = (*single_map_ridge_refinement_names(), "ridge_cfar_wide_4cycle")
    elif args.algorithms.strip() == "false-window-suppression":
        names = (*single_map_false_window_names(), "sm_pcfar_i05_o17_c04_l150_k1")
    elif args.algorithms.strip() == "false-window-refinement":
        names = (*single_map_false_window_refinement_names(), "sm_pcfar_i05_o17_c04_l150_k1")
    elif args.algorithms.strip() == "absolute-persistence":
        names = (
            *single_map_absolute_persistence_names(),
            "sm_pcfar_i05_o17_c04_l150_k1",
            "row_mad_topk_mean",
            "raw_max_power_ratio",
        )
    elif args.algorithms.strip() == "cpro-ablation":
        names = (*single_map_cpro_names(), "sm_pcfar_i05_o17_c04_l150_k1")
    else:
        names = tuple(value.strip() for value in args.algorithms.split(",") if value.strip())
    algorithms = resolve_cwt_activity_algorithms(names)
    invalid = [
        item.name
        for item in algorithms
        if item.input_denoiser not in {"none", "absolute"} or item.complexity != "O(P*T)"
    ]
    if invalid:
        raise SystemExit("invalid strict single-map algorithms: " + ", ".join(invalid))
    result = run_cwt_activity_rank(
        CWTActivityRun(
            output_dir=args.output,
            input_path=input_path,
            injection_config=args.injections,
            cwt_config=args.cwt_config,
            algorithms=names,
            cwt_backend=args.backend,
            pelt_threads=args.pelt_threads,
            candidate_period_max_records=1000.0,
            progress_every=args.progress_every,
            negative_control=True,
            negative_f_start_mhz=0.15,
            negative_f_stop_mhz=1.90,
            negative_max_channels=args.negative_max_channels,
            negative_channel_indices=negative_channels,
            negative_window_method=args.negative_window_method,
            strict_single_map=True,
            max_groups_per_family=args.max_groups_per_family,
        )
    )
    with (args.output / "cwt_activity_summary.csv").open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    print(f"best={result['best_algorithm']} algorithms={len(algorithms)} groups={result['group_count']}")
    for index, row in enumerate(rows[:10], start=1):
        print(
            f"{index:02d} {row['algorithm']} score={float(row['rank_score']):.6f} "
            f"paired_activity={float(row['paired_activity_detection_rate']):.3f} "
            f"paired_band={float(row['paired_band_detection_rate']):.3f} "
            f"false_windows={float(row['false_window_count']):.0f}"
        )


if __name__ == "__main__":
    main()
