#!/usr/bin/env python3
"""Rank CPRO single-channel CWT-to-time compression candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parents[1]
SRC_DIR = PROJECT_DIR / "src"
for path in (THIS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compression_config_rank import injection_group_id, largest_complete_2c  # noqa: E402
from cwt_activity_rank import (  # noqa: E402
    CWTActivityRun,
    _limit_specs_by_family,
    activity_config_from_cwt,
    prepare_activity_component,
)
from cwipss.analysis.injection_config import (  # noqa: E402
    load_injection_config,
    make_injections_from_config,
)
from cwipss.data.readers import open_spectrum_reader  # noqa: E402
from cwipss.signal.activity import crop_valid_periods  # noqa: E402
from cwipss.signal.cwt import cwt_power_cube, period_grid_records  # noqa: E402
from persistent_occupancy import (  # noqa: E402
    PersistentOccupancyParameters,
    difference_noise_std,
    impulse_cwt_noise_gain,
    persistent_occupancy_catalog,
    persistent_occupancy_windows,
)


COMPONENT_FIELDS = [
    "algorithm",
    "component_id",
    "group_id",
    "channel_index",
    "frequency_mhz",
    "period_records",
    "duration_records",
    "truth_window_hit",
    "truth_coverage",
    "paired_new_hit",
    "paired_new_coverage",
    "peak_in_truth",
    "paired_activity_detected",
    "paired_band_detected",
    "baseline_truth_coverage",
    "injected_window_count",
    "baseline_window_count",
    "algorithm_seconds",
]

NEGATIVE_FIELDS = [
    "algorithm",
    "channel_index",
    "frequency_mhz",
    "records",
    "window_count",
    "windows_per_10000_records",
    "active_fraction",
    "median_window_duration",
    "max_window_duration",
    "algorithm_seconds",
]

SUMMARY_FIELDS = [
    "rank",
    "algorithm",
    "hard_gate_pass",
    "group_count",
    "group_truth_window_hit_rate",
    "group_paired_new_hit_rate",
    "group_peak_in_truth_rate",
    "group_paired_activity_detection_rate",
    "group_paired_band_detection_rate",
    "mean_truth_coverage",
    "mean_paired_new_coverage",
    "mean_baseline_truth_coverage",
    "negative_channel_count",
    "false_window_count",
    "false_windows_per_channel",
    "false_windows_p95",
    "false_windows_max",
    "false_windows_per_10000_mean",
    "false_windows_per_10000_p95",
    "false_windows_per_10000_max",
    "false_active_fraction_mean",
    "false_active_fraction_p95",
    "false_active_fraction_max",
    "false_window_duration_median",
    "mean_algorithm_seconds",
]


HARD_GATES = {
    "min_group_truth_window_hit_rate": 0.90,
    "min_group_paired_new_hit_rate": 0.85,
    "min_group_peak_in_truth_rate": 0.85,
    "max_false_windows_per_10000_mean": 0.50,
    "max_false_windows_per_10000_p95": 1.25,
    "max_false_windows_per_10000_any_channel": 2.00,
    "max_false_active_fraction_mean": 0.05,
    "max_false_active_fraction_p95": 0.15,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _safe_quantile(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, quantile)) if finite.size else 0.0


def _overlaps_truth(windows: tuple[dict[str, float | int], ...], start: int, stop: int) -> bool:
    return any(
        int(window["record_start"]) < stop and int(window["record_stop"]) > start
        for window in windows
    )


def _component_metrics(
    *,
    params: PersistentOccupancyParameters,
    prepared: Any,
    baseline: np.ndarray,
    noise_gain: np.ndarray,
) -> tuple[dict[str, Any], float]:
    started = perf_counter()
    noise_std = difference_noise_std(baseline)
    baseline_result = persistent_occupancy_windows(
        prepared.baseline_cwt_power["absolute"],
        noise_std=noise_std,
        noise_gain=noise_gain,
        params=params,
    )
    result = persistent_occupancy_windows(
        prepared.cwt_power["absolute"],
        noise_std=noise_std,
        noise_gain=noise_gain,
        params=params,
    )
    elapsed = perf_counter() - started
    start = max(0, int(prepared.truth["record_start"]))
    stop = min(int(result.active_mask.size), int(prepared.truth["record_stop"]))
    truth_slice = slice(start, max(start + 1, stop))
    truth_mask = result.active_mask[truth_slice]
    baseline_truth_mask = baseline_result.active_mask[truth_slice]
    paired_new = truth_mask & ~baseline_truth_mask
    peak_record = int(np.argmax(result.activity)) if np.any(result.activity > 0.0) else -1

    truth_period = float(prepared.truth["period_records"])
    center = int(
        np.argmin(
            np.abs(
                np.log(np.maximum(prepared.valid_periods, 1e-12) / max(truth_period, 1e-12))
            )
        )
    )
    period_slice = slice(max(0, center - 1), min(result.score_map.shape[0], center + 2))
    truth_band = result.score_map[period_slice, truth_slice]
    return (
        {
            "truth_window_hit": int(_overlaps_truth(result.windows, start, stop)),
            "truth_coverage": float(np.mean(truth_mask)),
            "paired_new_hit": int(np.any(paired_new)),
            "paired_new_coverage": float(np.mean(paired_new)),
            "peak_in_truth": int(start <= peak_record < stop),
            "paired_activity_detected": int(
                _safe_quantile(result.activity[truth_slice], 0.95)
                > _safe_quantile(baseline_result.activity, 0.999)
            ),
            "paired_band_detected": int(
                _safe_quantile(truth_band, 0.95)
                > _safe_quantile(baseline_result.score_map, 0.999)
            ),
            "baseline_truth_coverage": float(np.mean(baseline_truth_mask)),
            "injected_window_count": len(result.windows),
            "baseline_window_count": len(baseline_result.windows),
        },
        elapsed,
    )


def _negative_channels(reader: Any, text: str, maximum: int) -> list[int]:
    if text.strip():
        return sorted({int(value.strip()) for value in text.split(",") if value.strip()})
    frequencies = np.asarray(reader.freqs_mhz)
    eligible = np.flatnonzero((frequencies >= 0.15) & (frequencies <= 1.90))
    if maximum > 0 and eligible.size > maximum:
        positions = np.linspace(0, eligible.size - 1, maximum).round().astype(int)
        eligible = eligible[positions]
    return [int(value) for value in eligible]


def _group_summary(
    component_rows: list[dict[str, Any]],
    negative_rows: list[dict[str, Any]],
    algorithms: tuple[PersistentOccupancyParameters, ...],
) -> list[dict[str, Any]]:
    component_buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in component_rows:
        component_buckets[(str(row["algorithm"]), str(row["group_id"]))].append(row)
    summary: list[dict[str, Any]] = []
    for params in algorithms:
        groups = [
            rows for (algorithm, _group), rows in component_buckets.items() if algorithm == params.name
        ]
        negative = [row for row in negative_rows if row["algorithm"] == params.name]
        group_metric = lambda field: [max(float(row[field]) for row in rows) for rows in groups]
        window_counts = np.asarray([float(row["window_count"]) for row in negative])
        window_density = np.asarray(
            [float(row["windows_per_10000_records"]) for row in negative]
        )
        active = np.asarray([float(row["active_fraction"]) for row in negative])
        durations = [
            float(row["median_window_duration"])
            for row in negative
            if float(row["window_count"]) > 0.0
        ]
        row = {
            "algorithm": params.name,
            "group_count": len(groups),
            "group_truth_window_hit_rate": float(np.mean(group_metric("truth_window_hit"))),
            "group_paired_new_hit_rate": float(np.mean(group_metric("paired_new_hit"))),
            "group_peak_in_truth_rate": float(np.mean(group_metric("peak_in_truth"))),
            "group_paired_activity_detection_rate": float(
                np.mean(group_metric("paired_activity_detected"))
            ),
            "group_paired_band_detection_rate": float(np.mean(group_metric("paired_band_detected"))),
            "mean_truth_coverage": float(np.mean(group_metric("truth_coverage"))),
            "mean_paired_new_coverage": float(np.mean(group_metric("paired_new_coverage"))),
            "mean_baseline_truth_coverage": float(np.mean(group_metric("baseline_truth_coverage"))),
            "negative_channel_count": len(negative),
            "false_window_count": int(np.sum(window_counts)),
            "false_windows_per_channel": float(np.mean(window_counts)),
            "false_windows_p95": _safe_quantile(window_counts, 0.95),
            "false_windows_max": float(np.max(window_counts)) if window_counts.size else 0.0,
            "false_windows_per_10000_mean": (
                float(np.mean(window_density)) if window_density.size else 0.0
            ),
            "false_windows_per_10000_p95": _safe_quantile(window_density, 0.95),
            "false_windows_per_10000_max": (
                float(np.max(window_density)) if window_density.size else 0.0
            ),
            "false_active_fraction_mean": float(np.mean(active)) if active.size else 0.0,
            "false_active_fraction_p95": _safe_quantile(active, 0.95),
            "false_active_fraction_max": float(np.max(active)) if active.size else 0.0,
            "false_window_duration_median": float(np.median(durations)) if durations else 0.0,
            "mean_algorithm_seconds": float(
                np.mean(
                    [
                        float(item["algorithm_seconds"])
                        for item in component_rows
                        if item["algorithm"] == params.name
                    ]
                )
            ),
        }
        row["hard_gate_pass"] = int(
            row["group_truth_window_hit_rate"] >= HARD_GATES["min_group_truth_window_hit_rate"]
            and row["group_paired_new_hit_rate"] >= HARD_GATES["min_group_paired_new_hit_rate"]
            and row["group_peak_in_truth_rate"] >= HARD_GATES["min_group_peak_in_truth_rate"]
            and row["false_windows_per_10000_mean"]
            <= HARD_GATES["max_false_windows_per_10000_mean"]
            and row["false_windows_per_10000_p95"]
            <= HARD_GATES["max_false_windows_per_10000_p95"]
            and row["false_windows_per_10000_max"]
            <= HARD_GATES["max_false_windows_per_10000_any_channel"]
            and row["false_active_fraction_mean"] <= HARD_GATES["max_false_active_fraction_mean"]
            and row["false_active_fraction_p95"] <= HARD_GATES["max_false_active_fraction_p95"]
        )
        summary.append(row)
    summary.sort(
        key=lambda row: (
            -int(row["hard_gate_pass"]),
            -float(row["group_paired_new_hit_rate"]),
            -float(row["group_truth_window_hit_rate"]),
            float(row["false_windows_per_channel"]),
            float(row["false_active_fraction_mean"]),
            -float(row["group_peak_in_truth_rate"]),
        )
    )
    for rank, row in enumerate(summary, start=1):
        row["rank"] = rank
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--injections",
        type=Path,
        default=PROJECT_DIR / "configs/injection_lowfreq_random_100.json",
    )
    parser.add_argument("--cwt-config", type=Path, default=PROJECT_DIR / "configs/cwt_default.json")
    parser.add_argument("--algorithms", help="Comma-separated CPRO catalog names.")
    parser.add_argument("--max-groups-per-family", type=int, default=0)
    parser.add_argument("--negative-channels", default="")
    parser.add_argument("--negative-max-channels", type=int, default=0)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    input_path = args.input or largest_complete_2c(PROJECT_DIR / "data/CE4")
    catalog = {params.name: params for params in persistent_occupancy_catalog()}
    names = (
        [value.strip() for value in args.algorithms.split(",") if value.strip()]
        if args.algorithms
        else list(catalog)
    )
    missing = sorted(set(names) - set(catalog))
    if missing:
        raise SystemExit("unknown CPRO algorithms: " + ", ".join(missing))
    algorithms = tuple(catalog[name] for name in names)

    reader = open_spectrum_reader(input_path)
    run = CWTActivityRun(
        output_dir=args.output,
        input_path=input_path,
        injection_config=args.injections,
        cwt_config=args.cwt_config,
        cwt_backend=args.backend,
        candidate_period_max_records=1000.0,
        max_groups_per_family=args.max_groups_per_family,
    )
    config = activity_config_from_cwt(run)
    periods = period_grid_records(
        config.period_min_records,
        config.period_max_records,
        config.period_count,
        config.period_spacing,
    )
    specs = make_injections_from_config(
        load_injection_config(args.injections),
        records=reader.n_records,
        channels=reader.n_channels,
        freqs_mhz=reader.freqs_mhz,
    )
    specs = _limit_specs_by_family(specs, args.max_groups_per_family)
    component_rows: list[dict[str, Any]] = []
    noise_gain: np.ndarray | None = None
    started = perf_counter()
    for index, spec in enumerate(specs, start=1):
        prepared, baseline, _injected = prepare_activity_component(
            reader=reader,
            spec=spec,
            config=config,
            periods=periods,
            input_denoisers=("absolute",),
        )
        if noise_gain is None:
            noise_gain = impulse_cwt_noise_gain(prepared.valid_periods, wavelet=config.wavelet)
        for params in algorithms:
            metrics, elapsed = _component_metrics(
                params=params,
                prepared=prepared,
                baseline=baseline,
                noise_gain=noise_gain,
            )
            channel = int(round(float(spec.channel_center)))
            component_rows.append(
                {
                    "algorithm": params.name,
                    "component_id": spec.injection_id,
                    "group_id": injection_group_id(spec.injection_id),
                    "channel_index": channel,
                    "frequency_mhz": float(reader.freqs_mhz[channel]),
                    "period_records": float(spec.period_records),
                    "duration_records": int(prepared.truth["duration_records"]),
                    **metrics,
                    "algorithm_seconds": elapsed,
                }
            )
        if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0 or index == len(specs)):
            print(f"[cpro-injection] {index}/{len(specs)} elapsed={perf_counter() - started:.1f}s", flush=True)

    negative_rows: list[dict[str, Any]] = []
    channels = _negative_channels(reader, args.negative_channels, args.negative_max_channels)
    for offset, channel in enumerate(channels, start=1):
        series = np.asarray(
            reader.read_block(slice(0, reader.n_records), slice(channel, channel + 1)).data[:, 0],
            dtype=np.float32,
        )
        cwt = cwt_power_cube(
            series[:, None],
            periods,
            wavelet=config.wavelet,
            normalize_channels=False,
            method=config.cwt_method,
            backend=args.backend,
            cuda_device=config.cuda_device,
        )
        valid_power, valid_periods, _mask = crop_valid_periods(
            cwt,
            periods,
            config.candidate_period_min_records,
            config.candidate_period_max_records,
        )
        if noise_gain is None:
            noise_gain = impulse_cwt_noise_gain(valid_periods, wavelet=config.wavelet)
        noise_std = difference_noise_std(series)
        for params in algorithms:
            algorithm_started = perf_counter()
            result = persistent_occupancy_windows(
                valid_power[:, :, 0],
                noise_std=noise_std,
                noise_gain=noise_gain,
                params=params,
            )
            durations = [int(window["duration_records"]) for window in result.windows]
            negative_rows.append(
                {
                    "algorithm": params.name,
                    "channel_index": channel,
                    "frequency_mhz": float(reader.freqs_mhz[channel]),
                    "records": int(series.size),
                    "window_count": len(result.windows),
                    "windows_per_10000_records": (
                        10_000.0 * float(len(result.windows)) / float(max(1, series.size))
                    ),
                    "active_fraction": float(np.mean(result.active_mask)),
                    "median_window_duration": float(np.median(durations)) if durations else 0.0,
                    "max_window_duration": max(durations, default=0),
                    "algorithm_seconds": perf_counter() - algorithm_started,
                }
            )
        if args.progress_every > 0 and (offset == 1 or offset % args.progress_every == 0 or offset == len(channels)):
            print(f"[cpro-negative] {offset}/{len(channels)} elapsed={perf_counter() - started:.1f}s", flush=True)

    summary = _group_summary(component_rows, negative_rows, algorithms)
    _write_csv(args.output / "component_cases.csv", COMPONENT_FIELDS, component_rows)
    _write_csv(args.output / "negative_control_cases.csv", NEGATIVE_FIELDS, negative_rows)
    _write_csv(args.output / "summary.csv", SUMMARY_FIELDS, summary)
    (args.output / "algorithm_map.json").write_text(
        json.dumps({params.name: params.to_dict() for params in algorithms}, indent=2, ensure_ascii=True)
    )
    shutil.copy2(args.injections, args.output / "injection_config.json")
    shutil.copy2(args.cwt_config, args.output / "cwt_config.json")
    result = {
        "method": "Calibrated Persistent Ridge Occupancy (CPRO)",
        "status": (
            "full_scientific_validation"
            if args.max_groups_per_family == 0 and len(channels) >= 90
            else "scientific_screen_only"
        ),
        "best_algorithm": summary[0]["algorithm"] if summary else None,
        "hard_gate_pass_count": sum(int(row["hard_gate_pass"]) for row in summary),
        "component_count": len(specs),
        "group_count": len({injection_group_id(spec.injection_id) for spec in specs}),
        "negative_channels": channels,
        "hard_gates": HARD_GATES,
        "elapsed_seconds": perf_counter() - started,
        "reproducibility": {
            "input": str(input_path),
            "input_sha256": _sha256(input_path),
            "injection_sha256": _sha256(args.injections),
            "cwt_config_sha256": _sha256(args.cwt_config),
            "command": " ".join(sys.argv),
        },
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=True))
    print(json.dumps(result, indent=2, ensure_ascii=True))
    for row in summary[:10]:
        print(
            f"{int(row['rank']):02d} {row['algorithm']} pass={int(row['hard_gate_pass'])} "
            f"hit={float(row['group_truth_window_hit_rate']):.3f} "
            f"new={float(row['group_paired_new_hit_rate']):.3f} "
            f"false={float(row['false_windows_per_channel']):.2f}/ch "
            f"active={float(row['false_active_fraction_mean']):.4f}"
        )


if __name__ == "__main__":
    main()
