"""Compression ranking on configured injections over a real CE4 background."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from compression_benchmark import (  # noqa: E402
    CASE_FIELDNAMES,
    DEFAULT_TOP10_ALGORITHMS,
    SUMMARY_FIELDNAMES,
    CompressionBenchmarkConfig,
    _PreparedCase,
    _best_band_timeseries,
    _evaluate_algorithm,
    _resolve_algorithms,
    _summary_rows,
    compression_algorithm_map,
)
from cwipss.analysis.injection_config import (  # noqa: E402
    load_injection_config,
    make_injections_from_config,
)
from cwipss.analysis.simulation import InjectionSpec, inject_periodic_signal, injection_truth  # noqa: E402
from cwipss.config import load_cwt_config  # noqa: E402
from cwipss.data.readers import CE4_RECORD_LEN, open_spectrum_reader  # noqa: E402
from cwipss.signal.activity import (  # noqa: E402
    coherent_structure_map,
    crop_valid_periods,
    low_fraction_noise_floor,
    relative_excess,
)
from cwipss.signal.cwt import cwt_power_cube, period_grid_records  # noqa: E402


GROUP_EXTRA_FIELDS = [
    "group_id",
    "component_count",
    "component_ids",
    "component_channels",
    "component_frequencies_mhz",
]

SCIENTIFIC_RANK_WEIGHTS = {
    "peak_in_truth_rate": 0.20,
    "truth_window_hit_rate": 0.15,
    "mean_time_contrast_z": 0.20,
    "mean_peak_global_band_contrast": 0.20,
    "mean_peak_period_concentration": 0.15,
    "mean_truth_window_local_band_contrast": 0.05,
}


@dataclass(frozen=True)
class ConfiguredCompressionRun:
    output_dir: Path
    input_path: Path
    injection_config: Path
    cwt_config: Path
    algorithms: tuple[str, ...] = DEFAULT_TOP10_ALGORITHMS
    cwt_backend: str = "cpu"
    cuda_device: int = 0
    candidate_period_max_records: float = 1000.0
    progress_every: int = 10


def largest_complete_2c(input_dir: str | Path) -> Path:
    root = Path(input_dir)
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".2c")
    complete = [
        path
        for path in files
        if path.stat().st_size > 0 and path.stat().st_size % CE4_RECORD_LEN == 0
    ]
    if not complete:
        raise FileNotFoundError(f"No complete CE4 .2C files found under: {root}")
    return max(complete, key=lambda path: path.stat().st_size)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def injection_group_id(injection_id: str) -> str:
    without_sequence = re.sub(r"^inj_[0-9]+_", "", str(injection_id))
    return re.sub(r"_c[0-9]+$", "", without_sequence)


def local_record_window(spec: InjectionSpec, total_records: int) -> tuple[int, int]:
    signal_start = int(spec.record_start)
    signal_stop = signal_start + int(spec.duration_records or 0)
    signal_len = max(1, signal_stop - signal_start)
    margin = max(1, int(math.ceil(0.5 * signal_len)))
    start = max(0, signal_start - margin)
    stop = min(int(total_records), signal_stop + margin)
    if stop <= start:
        raise ValueError(f"Invalid local record window for {spec.injection_id}: {start}:{stop}")
    return start, stop


def _physical_noise_std(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    median = float(np.nanmedian(data))
    mad = float(np.nanmedian(np.abs(data - median)))
    robust = 1.4826 * mad
    if np.isfinite(robust) and robust > np.finfo(np.float32).tiny:
        return robust
    fallback = float(np.nanstd(data))
    return fallback if np.isfinite(fallback) and fallback > 0.0 else 1.0


def compression_config_from_cwt(
    run: ConfiguredCompressionRun,
) -> CompressionBenchmarkConfig:
    cwt = load_cwt_config(
        run.cwt_config,
        overrides={
            "cwt_backend": run.cwt_backend,
            "cuda_device": run.cuda_device,
            "candidate_period_max_records": run.candidate_period_max_records,
        },
    )
    return CompressionBenchmarkConfig(
        output_dir=run.output_dir,
        background_modes=("ce4",),
        ce4_input_path=run.input_path,
        wavelet=cwt.wavelet,
        cwt_method=cwt.cwt_method,
        cwt_backend=cwt.cwt_backend,
        cuda_device=cwt.cuda_device,
        period_min_records=cwt.period_min_records,
        period_max_records=cwt.period_max_records,
        period_count=cwt.period_count,
        period_spacing=cwt.period_spacing,
        candidate_period_min_records=cwt.candidate_period_min_records,
        candidate_period_max_records=cwt.candidate_period_max_records,
        noise_floor_fraction=cwt.noise_floor_fraction,
        excess_eps_fraction=cwt.excess_eps_fraction,
        structure_baseline_quantile=cwt.structure_baseline_quantile,
        structure_scale_quantile=cwt.structure_scale_quantile,
        structure_z_threshold=cwt.structure_z_threshold,
        structure_time_support_records=cwt.structure_time_support_records,
        structure_period_support_bins=cwt.structure_period_support_bins,
        structure_min_support_fraction=cwt.structure_min_support_fraction,
        pelt_penalty=cwt.pelt_penalty,
        pelt_min_size_records=cwt.pelt_min_size_records,
        pelt_jump_records=cwt.pelt_jump_records,
        window_min_duration_records=cwt.window_min_duration_records,
        window_min_activity_mean=cwt.window_min_activity_mean,
        window_merge_gap_records=cwt.window_merge_gap_records,
        algorithms=run.algorithms,
        progress_every=run.progress_every,
    )


def prepare_configured_component(
    *,
    reader: Any,
    spec: InjectionSpec,
    config: CompressionBenchmarkConfig,
    periods: np.ndarray,
) -> tuple[_PreparedCase, np.ndarray, np.ndarray]:
    channel_index = min(max(int(round(float(spec.channel_center))), 0), int(reader.n_channels) - 1)
    local_start, local_stop = local_record_window(spec, int(reader.n_records))
    block = reader.read_block(slice(local_start, local_stop), slice(channel_index, channel_index + 1))
    baseline = np.asarray(block.data[:, 0], dtype=np.float32)
    local_spec = replace(
        spec,
        record_start=int(spec.record_start) - local_start,
        channel_center=0.0,
        bandwidth_channels=1.0,
        drift_channels=0.0,
    )
    injected, truth = inject_periodic_signal(baseline[:, None], local_spec)
    records = int(injected.shape[0])
    preprocess_start = perf_counter()
    power = cwt_power_cube(
        injected,
        periods,
        wavelet=config.wavelet,
        normalize_channels=True,
        method=config.cwt_method,
        backend=config.cwt_backend,
        cuda_device=config.cuda_device,
    )[:, :, 0]
    valid_power, valid_periods, _mask = crop_valid_periods(
        power,
        periods,
        config.candidate_period_min_records,
        config.candidate_period_max_records,
    )
    noise_floor = low_fraction_noise_floor(valid_power, fraction=config.noise_floor_fraction)
    excess = relative_excess(valid_power, noise_floor, eps_fraction=config.excess_eps_fraction)
    structured = coherent_structure_map(
        excess,
        baseline_quantile=config.structure_baseline_quantile,
        scale_quantile=config.structure_scale_quantile,
        z_threshold=config.structure_z_threshold,
        time_support_records=min(config.structure_time_support_records, max(1, records // 8)),
        period_support_bins=config.structure_period_support_bins,
        min_support_fraction=config.structure_min_support_fraction,
    )
    preprocess_seconds = perf_counter() - preprocess_start

    best_contrast = np.full(records, -np.inf, dtype=np.float32)
    best_ratio = np.zeros(records, dtype=np.float32)
    best_periods = np.full(records, np.nan, dtype=np.float32)
    for width in config.band_widths:
        contrast, ratio, period_values = _best_band_timeseries(structured, valid_periods, width)
        update = contrast > best_contrast
        best_contrast = np.where(update, contrast, best_contrast)
        best_ratio = np.where(update, ratio, best_ratio)
        best_periods = np.where(update, period_values, best_periods)

    truth = dict(truth)
    truth["channel_index"] = 0
    prepared = _PreparedCase(
        case_id=spec.injection_id,
        records=records,
        channels=1,
        noise_std=_physical_noise_std(baseline),
        background_mode="ce4",
        background_source=str(reader.filename),
        background_record_start=local_start,
        background_record_stop=local_stop,
        background_channel_start=channel_index,
        background_channel_stop=channel_index + 1,
        preprocess_seconds=preprocess_seconds,
        injection=local_spec,
        truth=truth,
        valid_periods=valid_periods,
        structured=structured,
        time_band_contrast=best_contrast.astype(np.float32, copy=False),
        time_band_ratio=best_ratio.astype(np.float32, copy=False),
        time_band_periods=best_periods.astype(np.float32, copy=False),
    )
    return prepared, baseline, np.asarray(injected[:, 0], dtype=np.float32)


def _group_rows(
    component_rows: list[dict[str, Any]],
    specs_by_id: dict[str, InjectionSpec],
    freqs_mhz: np.ndarray,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in component_rows:
        buckets[(str(row["group_id"]), str(row["algorithm"]))].append(row)

    grouped: list[dict[str, Any]] = []
    for (group_id, _algorithm), rows in buckets.items():
        best = max(
            rows,
            key=lambda row: (
                int(row["peak_in_truth"]),
                int(row["truth_window_hit"]),
                float(row["time_contrast_z"]),
            ),
        )
        result = dict(best)
        component_ids = [str(row["case_id"]) for row in rows]
        components = [specs_by_id[component_id] for component_id in component_ids]
        result.update(
            {
                "case_id": group_id,
                "group_id": group_id,
                "component_count": len(rows),
                "component_ids": ";".join(component_ids),
                "component_channels": ";".join(
                    str(int(round(float(spec.channel_center)))) for spec in components
                ),
                "component_frequencies_mhz": ";".join(
                    f"{float(freqs_mhz[int(round(float(spec.channel_center))) ]):.9g}"
                    for spec in components
                ),
                "channels": len(rows),
                "peak_in_truth": max(int(row["peak_in_truth"]) for row in rows),
                "truth_window_hit": max(int(row["truth_window_hit"]) for row in rows),
                "truth_window_overlap_fraction": max(
                    float(row["truth_window_overlap_fraction"]) for row in rows
                ),
                "time_contrast_z": max(float(row["time_contrast_z"]) for row in rows),
                "peak_global_band_contrast": max(
                    float(row["peak_global_band_contrast"]) for row in rows
                ),
                "peak_period_concentration": max(
                    float(row["peak_period_concentration"]) for row in rows
                ),
                "peak_band_period_error_fraction": min(
                    float(row["peak_band_period_error_fraction"]) for row in rows
                ),
                "best_window_local_band_contrast": max(
                    float(row["best_window_local_band_contrast"]) for row in rows
                ),
                "truth_window_local_band_contrast": max(
                    float(row["truth_window_local_band_contrast"]) for row in rows
                ),
                "truth_window_period_error_fraction": min(
                    float(row["truth_window_period_error_fraction"]) for row in rows
                    if int(row["truth_window_hit"]) > 0
                )
                if any(int(row["truth_window_hit"]) > 0 for row in rows)
                else 1.0,
                "preprocess_seconds": sum(float(row["preprocess_seconds"]) for row in rows),
                "reduce_seconds": sum(float(row["reduce_seconds"]) for row in rows),
                "window_seconds": sum(float(row["window_seconds"]) for row in rows),
                "algorithm_seconds": sum(float(row["algorithm_seconds"]) for row in rows),
            }
        )
        result["algorithm_over_preprocess_ratio"] = float(
            result["algorithm_seconds"] / max(float(result["preprocess_seconds"]), 1e-12)
        )
        grouped.append(result)
    return grouped


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def _scientific_rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank deterministically; timing remains diagnostic and does not affect order."""
    if not rows:
        return rows
    weight_total = sum(SCIENTIFIC_RANK_WEIGHTS.values())
    scaled: dict[str, dict[str, float]] = {}
    for field in SCIENTIFIC_RANK_WEIGHTS:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
        if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(lo, hi):
            scaled[field] = {str(row["algorithm"]): 0.5 for row in rows}
            continue
        scaled[field] = {
            str(row["algorithm"]): float((value - lo) / (hi - lo))
            for row, value in zip(rows, values, strict=True)
        }
    for row in rows:
        name = str(row["algorithm"])
        row["rank_score"] = sum(
            weight * scaled[field][name]
            for field, weight in SCIENTIFIC_RANK_WEIGHTS.items()
        ) / weight_total
    rows.sort(
        key=lambda row: (
            -float(row["rank_score"]),
            -float(row["peak_in_truth_rate"]),
            str(row["algorithm"]),
        )
    )
    return rows


def run_configured_compression_rank(run: ConfiguredCompressionRun) -> dict[str, Any]:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    config = compression_config_from_cwt(run)
    reader = open_spectrum_reader(run.input_path)
    payload = load_injection_config(run.injection_config)
    specs = make_injections_from_config(
        payload,
        records=reader.n_records,
        channels=reader.n_channels,
        freqs_mhz=reader.freqs_mhz,
    )
    algorithms = _resolve_algorithms(config.algorithms)
    periods = period_grid_records(
        config.period_min_records,
        config.period_max_records,
        config.period_count,
        config.period_spacing,
    )
    specs_by_id = {spec.injection_id: spec for spec in specs}
    component_rows: list[dict[str, Any]] = []
    prepared_index: list[dict[str, Any]] = []
    started = perf_counter()
    for index, spec in enumerate(specs, start=1):
        prepared, _baseline, _injected = prepare_configured_component(
            reader=reader,
            spec=spec,
            config=config,
            periods=periods,
        )
        group_id = injection_group_id(spec.injection_id)
        prepared_index.append(
            {
                "component_id": spec.injection_id,
                "group_id": group_id,
                "channel_index": int(round(float(spec.channel_center))),
                "frequency_mhz": float(
                    reader.freqs_mhz[int(round(float(spec.channel_center)))]
                ),
                "local_record_start": prepared.background_record_start,
                "local_record_stop": prepared.background_record_stop,
            }
        )
        for algorithm in algorithms:
            row = _evaluate_algorithm(prepared, algorithm, config)
            row["group_id"] = group_id
            component_rows.append(row)
        if run.progress_every > 0 and (
            index == 1 or index % run.progress_every == 0 or index == len(specs)
        ):
            print(
                f"[configured-compression] component {index}/{len(specs)} "
                f"elapsed={perf_counter() - started:.1f}s",
                flush=True,
            )

    group_rows = _group_rows(component_rows, specs_by_id, np.asarray(reader.freqs_mhz))
    summary_rows = _scientific_rank(
        _summary_rows(
            group_rows,
            algorithms,
            max_algorithm_over_preprocess_ratio=config.max_algorithm_over_preprocess_ratio,
        )
    )
    component_fields = CASE_FIELDNAMES + ["group_id"]
    group_fields = CASE_FIELDNAMES + GROUP_EXTRA_FIELDS
    _write_csv(run.output_dir / "component_cases.csv", component_fields, component_rows)
    _write_csv(run.output_dir / "group_cases.csv", group_fields, group_rows)
    _write_csv(run.output_dir / "compression_summary.csv", SUMMARY_FIELDNAMES, summary_rows)
    _write_csv(
        run.output_dir / "component_index.csv",
        list(prepared_index[0]),
        prepared_index,
    )
    shutil.copy2(run.injection_config, run.output_dir / "injection_config.json")
    shutil.copy2(run.cwt_config, run.output_dir / "cwt_config.json")
    all_algorithms = compression_algorithm_map()
    selected_names = {algorithm.name for algorithm in algorithms}
    algorithm_map = {
        name: {
            **configuration,
            "selected": name in selected_names,
        }
        for name, configuration in all_algorithms.items()
    }
    (run.output_dir / "compression_algorithm_map.json").write_text(
        json.dumps(algorithm_map, indent=2, ensure_ascii=True)
    )

    grouped_specs: dict[str, list[InjectionSpec]] = defaultdict(list)
    for spec in specs:
        grouped_specs[injection_group_id(spec.injection_id)].append(spec)
    copy_distribution: dict[str, int] = defaultdict(int)
    for components in grouped_specs.values():
        copy_distribution[str(len(components))] += 1

    result = {
        "input_path": str(run.input_path),
        "injection_config": str(run.injection_config),
        "cwt_config": str(run.cwt_config),
        "component_count": len(specs),
        "group_count": len(grouped_specs),
        "copy_distribution": dict(sorted(copy_distribution.items())),
        "algorithm_count": len(algorithms),
        "available_algorithm_count": len(all_algorithms),
        "algorithm_map_file": "compression_algorithm_map.json",
        "reproducibility": {
            "input_size_bytes": run.input_path.stat().st_size,
            "input_sha256": file_sha256(run.input_path),
            "injection_config_sha256": file_sha256(run.injection_config),
            "cwt_config_sha256": file_sha256(run.cwt_config),
            "retained_injection_config": "injection_config.json",
            "retained_cwt_config": "cwt_config.json",
            "pytest_node": (
                "tests/perf/test_compression_benchmark.py::"
                "test_configured_real_ce4_compression_rank"
            ),
            "environment": {
                "CWIPSS_RUN_PERF": "1",
                "CWIPSS_CONFIG_COMPRESSION_OUTPUT": str(run.output_dir),
                "CWIPSS_CONFIG_COMPRESSION_INPUT": str(run.input_path),
                "CWIPSS_CONFIG_COMPRESSION_INJECTIONS": str(run.injection_config),
                "CWIPSS_CONFIG_COMPRESSION_CWT": str(run.cwt_config),
                "CWIPSS_CONFIG_COMPRESSION_ALGORITHMS": ",".join(run.algorithms),
                "CWIPSS_CONFIG_COMPRESSION_BACKEND": run.cwt_backend,
                "CWIPSS_CONFIG_COMPRESSION_CANDIDATE_MAX": str(
                    run.candidate_period_max_records
                ),
            },
            "command": [
                "python",
                "-m",
                "pytest",
                (
                    "tests/perf/test_compression_benchmark.py::"
                    "test_configured_real_ce4_compression_rank"
                ),
                "-q",
                "-s",
            ],
        },
        "group_aggregation": {
            "recovery_metrics": "maximum across same-signal frequency copies",
            "period_error": "minimum across same-signal frequency copies",
            "runtime": "sum across same-signal frequency copies",
        },
        "rank_score_basis": {
            "mode": "deterministic_scientific_metrics_only",
            "weights": SCIENTIFIC_RANK_WEIGHTS,
            "timing_note": "Runtime is recorded but does not affect configured benchmark rank.",
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "best_algorithm": str(summary_rows[0]["algorithm"]) if summary_rows else "",
        "metric_leaders": {
            "rank_score": str(summary_rows[0]["algorithm"]) if summary_rows else "",
            "peak_in_truth_rate": max(
                summary_rows, key=lambda row: float(row["peak_in_truth_rate"])
            )["algorithm"]
            if summary_rows
            else "",
            "truth_window_hit_rate": max(
                summary_rows, key=lambda row: float(row["truth_window_hit_rate"])
            )["algorithm"]
            if summary_rows
            else "",
            "mean_time_contrast_z": max(
                summary_rows, key=lambda row: float(row["mean_time_contrast_z"])
            )["algorithm"]
            if summary_rows
            else "",
            "mean_peak_global_band_contrast": max(
                summary_rows,
                key=lambda row: float(row["mean_peak_global_band_contrast"]),
            )["algorithm"]
            if summary_rows
            else "",
            "mean_peak_period_concentration": max(
                summary_rows,
                key=lambda row: float(row["mean_peak_period_concentration"]),
            )["algorithm"]
            if summary_rows
            else "",
        },
        "summary_rows": summary_rows,
    }
    (run.output_dir / "compression_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True)
    )
    return result
