"""Rank independent post-CWT activity algorithms on configured CE4 injections."""

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
    SUMMARY_FIELDNAMES,
    CompressionBenchmarkConfig,
    _activity_truth_metrics,
    _best_band_timeseries,
    _period_error_fraction,
    _summary_rows,
    _window_rows,
)
from compression_config_rank import (  # noqa: E402
    GROUP_EXTRA_FIELDS,
    SCIENTIFIC_RANK_WEIGHTS,
    _group_rows,
    injection_group_id,
    largest_complete_2c,
    local_record_window,
)
from cwt_activity_algorithms import (  # noqa: E402
    DEFAULT_CWT_ACTIVITY_ALGORITHMS,
    CWTActivityAlgorithm,
    compute_cwt_activity,
    cwt_activity_algorithm_map,
    resolve_cwt_activity_algorithms,
)
from cwipss.analysis.injection_config import (  # noqa: E402
    load_injection_config,
    make_injections_from_config,
)
from cwipss.analysis.simulation import InjectionSpec, inject_periodic_signal  # noqa: E402
from cwipss.config import load_cwt_config  # noqa: E402
from cwipss.data.readers import open_spectrum_reader  # noqa: E402
from cwipss.signal.cpro import cpro_period_mask, difference_noise_std, impulse_cwt_noise_gain  # noqa: E402
from cwipss.signal.activity import crop_valid_periods, robust_standardize  # noqa: E402
from cwipss.signal.cwt import cwt_power_cube, period_grid_records, robust_zscore_channels  # noqa: E402
from stage_boundaries import (  # noqa: E402
    pelt_parameters_from_config,
    segment_activity_batch_with_pelt,
    segment_activity_with_pelt,
    standardize_activity_for_pelt,
)


NEGATIVE_CONTROL_FIELDNAMES = [
    "algorithm",
    "channel_index",
    "frequency_mhz",
    "records",
    "window_count",
    "active_fraction",
    "peak_activity_z",
    "p95_activity_z",
    "mean_activity_z",
    "activity_p99_native",
    "activity_p999_native",
    "activity_max_native",
    "score_p99_native",
    "score_p999_native",
    "score_max_native",
    "score_positive_fraction",
    "cwt_seconds",
    "algorithm_seconds",
]

NEGATIVE_CONTROL_SUMMARY_FIELDNAMES = [
    "algorithm",
    "false_window_count",
    "false_windows_per_channel",
    "false_channels_with_windows_rate",
    "false_active_fraction",
    "false_peak_z_p95",
    "false_peak_z_max",
    "negative_activity_p999_p95",
    "negative_activity_max",
    "negative_score_p999_p95",
    "negative_score_max",
    "negative_score_positive_fraction",
    "negative_mean_algorithm_seconds",
]

DENOISING_CASE_FIELDNAMES = [
    "truth_activity_p95_native",
    "baseline_activity_p999_native",
    "paired_activity_gain_mean",
    "paired_activity_exceedance_fraction",
    "paired_activity_detected",
    "truth_band_score_p95_native",
    "baseline_score_p999_native",
    "paired_band_gain_mean",
    "paired_band_exceedance_fraction",
    "paired_band_detected",
]

DENOISING_SUMMARY_FIELDNAMES = [
    "paired_activity_detection_rate",
    "paired_band_detection_rate",
    "mean_paired_activity_gain",
    "mean_paired_band_gain",
    "mean_paired_activity_exceedance_fraction",
    "mean_paired_band_exceedance_fraction",
    "raw_safe_activity_recall",
    "raw_safe_band_recall",
    "median_truth_to_raw_activity_ratio",
    "median_truth_to_raw_band_ratio",
    "negative_window_control",
    "negative_map_sparsity",
    "negative_map_clean_rate",
]

NEGATIVE_MAP_TARGET_FRACTION = 0.001

CWT_ACTIVITY_SUMMARY_FIELDNAMES = (
    SUMMARY_FIELDNAMES
    + DENOISING_SUMMARY_FIELDNAMES
    + [
        "injection_rank_score",
        "false_window_count",
        "false_windows_per_channel",
        "false_channels_with_windows_rate",
        "false_active_fraction",
        "false_peak_z_p95",
        "false_peak_z_max",
        "negative_mean_algorithm_seconds",
    ]
)

COMBINED_RANK_WEIGHTS = {
    "raw_safe_band_recall": 0.15,
    "raw_safe_activity_recall": 0.12,
    "paired_band_detection_rate": 0.10,
    "paired_activity_detection_rate": 0.08,
    "mean_paired_band_exceedance_fraction": 0.05,
    "mean_paired_activity_exceedance_fraction": 0.03,
    "peak_in_truth_rate": 0.03,
    "truth_window_hit_rate": 0.05,
    "negative_window_control": 0.09,
    "negative_map_sparsity": 0.05,
    "negative_map_clean_rate": 0.25,
}


@dataclass(frozen=True)
class CWTActivityRun:
    output_dir: Path
    input_path: Path
    injection_config: Path
    cwt_config: Path
    algorithms: tuple[str, ...] = DEFAULT_CWT_ACTIVITY_ALGORITHMS
    cwt_backend: str = "cpu"
    cuda_device: int = 0
    pelt_threads: int = 0
    candidate_period_max_records: float = 1000.0
    progress_every: int = 10
    negative_control: bool = True
    negative_f_start_mhz: float = 0.15
    negative_f_stop_mhz: float = 1.90
    negative_max_channels: int = 0
    negative_channel_indices: tuple[int, ...] = ()
    negative_window_method: str = "pelt"
    strict_single_map: bool = False
    max_groups_per_family: int = 0


@dataclass(frozen=True)
class _PreparedActivityCase:
    case_id: str
    records: int
    channels: int
    noise_std: float
    background_mode: str
    background_source: str
    background_record_start: int
    background_record_stop: int
    background_channel_start: int
    background_channel_stop: int
    preprocess_seconds: float
    injection: InjectionSpec
    truth: dict[str, Any]
    valid_periods: np.ndarray
    noise_gain: np.ndarray
    baseline_cwt_power: dict[str, np.ndarray]
    cwt_power: dict[str, np.ndarray]
    baseline_reference_cwt_power: dict[str, np.ndarray]
    reference_cwt_power: dict[str, np.ndarray]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_input_denoisers(algorithms: list[CWTActivityAlgorithm]) -> tuple[str, ...]:
    modes = {str(algorithm.input_denoiser) for algorithm in algorithms}
    modes.add("none")
    return tuple(sorted(modes))


def _single_map_family(injection_id: str) -> str:
    value = str(injection_id)
    if "weak_family_b_multifreq" in value:
        return "weak_family_b_multifreq"
    if "weak_family_c_ultraweak" in value:
        return "weak_family_c_ultraweak"
    if "weak_family_a" in value:
        return "weak_family_a"
    return "other"


def _limit_specs_by_family(specs: list[InjectionSpec], max_groups_per_family: int) -> list[InjectionSpec]:
    limit = max(0, int(max_groups_per_family))
    if limit == 0:
        return specs
    selected_groups: dict[str, list[str]] = defaultdict(list)
    for spec in specs:
        family = _single_map_family(spec.injection_id)
        group = injection_group_id(spec.injection_id)
        if group not in selected_groups[family] and len(selected_groups[family]) < limit:
            selected_groups[family].append(group)
    allowed = {group for groups in selected_groups.values() for group in groups}
    return [spec for spec in specs if injection_group_id(spec.injection_id) in allowed]


def _post_cwt_reference_spec(mode: str) -> tuple[int, int, int] | None:
    """Return (reference count, guard width, read radius) for a post-CWT mode."""
    match = re.fullmatch(r"post_cwt_neighbor(\d+)(?:_guard(\d+))?", str(mode))
    if match is None:
        return None
    count = max(1, int(match.group(1)))
    guard = max(0, int(match.group(2) or 0))
    radius = guard + int(math.ceil(count / 2.0))
    return count, guard, radius


def _post_cwt_reference_indices(
    *,
    target_offset: int,
    channel_count: int,
    mode: str,
) -> list[int]:
    spec = _post_cwt_reference_spec(mode)
    if spec is None:
        return []
    count, guard, _radius = spec
    eligible = [
        index
        for index in range(int(channel_count))
        if index != int(target_offset) and abs(index - int(target_offset)) > guard
    ]
    eligible.sort(key=lambda index: (abs(index - int(target_offset)), index))
    if len(eligible) < count:
        raise ValueError(
            f"{mode} requires {count} reference channels outside a {guard}-channel guard; "
            f"only {len(eligible)} are available"
        )
    return eligible[:count]


def _input_denoiser_radius(modes: tuple[str, ...]) -> int:
    radius = 0
    if "neighbor_linear2" in modes:
        radius = max(radius, 1)
    if "neighbor_median4" in modes:
        radius = max(radius, 2)
    if "neighbor_median8" in modes:
        radius = max(radius, 4)
    for mode in modes:
        spec = _post_cwt_reference_spec(mode)
        if spec is not None:
            radius = max(radius, spec[2])
    unknown = set(modes) - {
        "none",
        "absolute",
        "neighbor_linear2",
        "neighbor_median4",
        "neighbor_median8",
        *(mode for mode in modes if _post_cwt_reference_spec(mode) is not None),
    }
    if unknown:
        raise ValueError(f"Unknown input denoisers: {', '.join(sorted(unknown))}")
    return radius


def _frequency_neighborhood_slice(channel_index: int, channel_count: int, radius: int) -> slice:
    width = min(int(channel_count), 2 * max(0, int(radius)) + 1)
    start = max(0, min(int(channel_index) - radius, int(channel_count) - width))
    return slice(start, start + width)


def _frequency_denoised_inputs(
    baseline: np.ndarray,
    injected: np.ndarray,
    *,
    target_offset: int,
    modes: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    base = np.asarray(baseline, dtype=np.float32)
    signal = np.asarray(injected, dtype=np.float32)
    if base.ndim != 2 or signal.shape != base.shape:
        raise ValueError("baseline and injected frequency neighborhoods must have matching (records, channels) shape")
    target = int(target_offset)
    if not 0 <= target < base.shape[1]:
        raise ValueError("target frequency channel is outside neighborhood")

    center = np.nanmedian(base, axis=0, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(base - center), axis=0, keepdims=True)
    fallback = np.nanstd(base, axis=0, keepdims=True)
    scale = np.where(np.isfinite(mad) & (mad > 1e-12), mad, fallback)
    scale = np.maximum(np.where(np.isfinite(scale), scale, 1.0), 1e-12)
    base_z = (base - center) / scale
    signal_z = (signal - center) / scale
    neighbor_order = sorted(
        (index for index in range(base.shape[1]) if index != target),
        key=lambda index: (abs(index - target), index),
    )

    base_outputs: dict[str, np.ndarray] = {}
    signal_outputs: dict[str, np.ndarray] = {}
    for mode in modes:
        if mode in {"none", "absolute"} or _post_cwt_reference_spec(mode) is not None:
            base_outputs[mode] = base[:, target]
            signal_outputs[mode] = signal[:, target]
            continue
        if mode == "neighbor_linear2":
            left = max(0, target - 1)
            right = min(base.shape[1] - 1, target + 1)
            refs = [index for index in (left, right) if index != target]
            if len(refs) < 2:
                refs = neighbor_order[:2]
            base_reference = np.nanmean(base_z[:, refs], axis=1)
            signal_reference = np.nanmean(signal_z[:, refs], axis=1)
        else:
            count = 4 if mode == "neighbor_median4" else 8
            refs = neighbor_order[:count]
            base_reference = np.nanmedian(base_z[:, refs], axis=1)
            signal_reference = np.nanmedian(signal_z[:, refs], axis=1)
        base_outputs[mode] = (base_z[:, target] - base_reference).astype(np.float32, copy=False)
        signal_outputs[mode] = (signal_z[:, target] - signal_reference).astype(np.float32, copy=False)
    return base_outputs, signal_outputs


def activity_config_from_cwt(run: CWTActivityRun) -> CompressionBenchmarkConfig:
    overrides: dict[str, Any] = {
        "cwt_backend": run.cwt_backend,
        "cuda_device": run.cuda_device,
        "candidate_period_max_records": run.candidate_period_max_records,
    }
    if int(run.pelt_threads) > 0:
        overrides["pelt_threads"] = int(run.pelt_threads)
    cwt = load_cwt_config(
        run.cwt_config,
        overrides=overrides,
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
        pelt_penalty=cwt.pelt_penalty,
        pelt_min_size_records=cwt.pelt_min_size_records,
        pelt_jump_records=cwt.pelt_jump_records,
        pelt_threads=cwt.pelt_threads,
        window_min_duration_records=cwt.window_min_duration_records,
        window_min_activity_mean=cwt.window_min_activity_mean,
        window_merge_gap_records=cwt.window_merge_gap_records,
        algorithms=run.algorithms,
        progress_every=run.progress_every,
    )


def prepare_activity_component(
    *,
    reader: Any,
    spec: InjectionSpec,
    config: CompressionBenchmarkConfig,
    periods: np.ndarray,
    input_denoisers: tuple[str, ...] = ("none",),
    noise_gain: np.ndarray | None = None,
) -> tuple[_PreparedActivityCase, np.ndarray, np.ndarray]:
    channel_index = min(max(int(round(float(spec.channel_center))), 0), int(reader.n_channels) - 1)
    local_start, local_stop = local_record_window(spec, int(reader.n_records))
    radius = _input_denoiser_radius(input_denoisers)
    channel_slice = _frequency_neighborhood_slice(channel_index, int(reader.n_channels), radius)
    block = reader.read_block(slice(local_start, local_stop), channel_slice)
    baseline_block = np.asarray(block.data, dtype=np.float32)
    target_offset = channel_index - int(channel_slice.start)
    baseline = np.asarray(baseline_block[:, target_offset], dtype=np.float32)
    local_spec = replace(
        spec,
        record_start=int(spec.record_start) - local_start,
        channel_center=float(target_offset),
        bandwidth_channels=1.0,
        drift_channels=0.0,
    )
    injected_block, truth = inject_periodic_signal(baseline_block, local_spec)
    base_inputs, signal_inputs = _frequency_denoised_inputs(
        baseline_block,
        injected_block,
        target_offset=target_offset,
        modes=input_denoisers,
    )
    records = int(injected_block.shape[0])
    preprocess_start = perf_counter()
    paired_columns = (
        [base_inputs[mode] for mode in input_denoisers]
        + [signal_inputs[mode] for mode in input_denoisers]
    )
    post_cwt_modes = tuple(
        mode for mode in input_denoisers if _post_cwt_reference_spec(mode) is not None
    )
    post_cwt_requested = bool(post_cwt_modes)
    if post_cwt_requested:
        paired_columns += [baseline_block[:, index] for index in range(baseline_block.shape[1])]
        paired_columns += [injected_block[:, index] for index in range(injected_block.shape[1])]
    paired = np.column_stack(paired_columns)
    absolute_indices = [index for index, mode in enumerate(input_denoisers) if mode == "absolute"]
    absolute_indices += [
        index + len(input_denoisers)
        for index, mode in enumerate(input_denoisers)
        if mode == "absolute"
    ]
    if absolute_indices:
        normalized_indices = [index for index in range(paired.shape[1]) if index not in set(absolute_indices)]
        if normalized_indices:
            paired[:, normalized_indices] = robust_zscore_channels(paired[:, normalized_indices])
    power = cwt_power_cube(
        paired,
        periods,
        wavelet=config.wavelet,
        normalize_channels=not absolute_indices,
        method=config.cwt_method,
        backend=config.cwt_backend,
        cuda_device=config.cuda_device,
    )
    valid_power, valid_periods, _mask = crop_valid_periods(
        power,
        periods,
        config.candidate_period_min_records,
        config.candidate_period_max_records,
    )
    preprocess_seconds = perf_counter() - preprocess_start
    resolved_noise_gain = (
        impulse_cwt_noise_gain(valid_periods, wavelet=config.wavelet, method=config.cwt_method)
        if noise_gain is None
        else np.asarray(noise_gain, dtype=np.float32)
    )
    if resolved_noise_gain.shape != (valid_periods.size,):
        raise ValueError("noise_gain must match the candidate period grid")

    truth = dict(truth)
    truth["channel_index"] = 0
    baseline_references: dict[str, np.ndarray] = {}
    signal_references: dict[str, np.ndarray] = {}
    if post_cwt_requested:
        neighborhood_start = 2 * len(input_denoisers)
        neighborhood_stop = neighborhood_start + baseline_block.shape[1]
        baseline_neighborhood = valid_power[:, :, neighborhood_start:neighborhood_stop]
        signal_neighborhood = valid_power[:, :, neighborhood_stop : neighborhood_stop + baseline_block.shape[1]]
        for mode in post_cwt_modes:
            reference_indices = _post_cwt_reference_indices(
                target_offset=target_offset,
                channel_count=baseline_block.shape[1],
                mode=mode,
            )
            baseline_references[mode] = np.asarray(
                baseline_neighborhood[:, :, reference_indices], dtype=np.float32
            )
            signal_references[mode] = np.asarray(
                signal_neighborhood[:, :, reference_indices], dtype=np.float32
            )

    prepared = _PreparedActivityCase(
        case_id=spec.injection_id,
        records=records,
        channels=1,
        noise_std=difference_noise_std(baseline),
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
        noise_gain=resolved_noise_gain,
        baseline_cwt_power={
            mode: np.asarray(valid_power[:, :, index], dtype=np.float32)
            for index, mode in enumerate(input_denoisers)
        },
        cwt_power={
            mode: np.asarray(valid_power[:, :, index + len(input_denoisers)], dtype=np.float32)
            for index, mode in enumerate(input_denoisers)
        },
        baseline_reference_cwt_power=baseline_references,
        reference_cwt_power=signal_references,
    )
    return prepared, baseline, np.asarray(injected_block[:, target_offset], dtype=np.float32)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def _scientific_rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _scale_rank_field(rows: list[dict[str, Any]], field: str, *, inverse: bool = False) -> dict[str, float]:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(lo, hi):
        return {str(row["algorithm"]): 0.5 for row in rows}
    scaled = (values - lo) / (hi - lo)
    if inverse:
        scaled = 1.0 - scaled
    return {str(row["algorithm"]): float(value) for row, value in zip(rows, scaled, strict=True)}


def _combined_rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    weight_total = sum(COMBINED_RANK_WEIGHTS.values())
    for row in rows:
        row["rank_score"] = sum(
            weight * min(max(float(row[field]), 0.0), 1.0)
            for field, weight in COMBINED_RANK_WEIGHTS.items()
        ) / weight_total
    rows.sort(
        key=lambda row: (
            -float(row["rank_score"]),
            float(row["false_windows_per_channel"]),
            -float(row["peak_in_truth_rate"]),
            str(row["algorithm"]),
        )
    )
    return rows


def _safe_quantile(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(np.nanquantile(finite, quantile))


def _activity_windows(
    activity_z: np.ndarray,
    config: CompressionBenchmarkConfig,
) -> list[dict[str, float | int]]:
    result = segment_activity_with_pelt(
        activity_z,
        pelt_parameters_from_config(config),
        activity_z=activity_z,
    )
    return [dict(window) for window in result.windows]


def _negative_windows(
    activity_z: np.ndarray,
    config: CompressionBenchmarkConfig,
    run: CWTActivityRun,
) -> list[dict[str, float | int]]:
    if str(run.negative_window_method).strip().lower() != "pelt":
        raise ValueError("The three-stage rank requires native PELT for every negative control")
    return _activity_windows(activity_z, config)


def _windows_active_fraction(windows: list[dict[str, float | int]], records: int) -> float:
    if records <= 0:
        return 0.0
    duration = sum(
        max(0, int(window["record_stop"]) - int(window["record_start"]))
        for window in windows
    )
    return float(duration) / float(records)


def _negative_control_summary(
    rows: list[dict[str, Any]],
    algorithms: list[CWTActivityAlgorithm],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for algorithm in algorithms:
        selected = [row for row in rows if row["algorithm"] == algorithm.name]
        if not selected:
            summary.append(
                {
                    "algorithm": algorithm.name,
                    "false_window_count": 0,
                    "false_windows_per_channel": 0.0,
                    "false_channels_with_windows_rate": 0.0,
                    "false_active_fraction": 0.0,
                    "false_peak_z_p95": 0.0,
                    "false_peak_z_max": 0.0,
                    "negative_activity_p999_p95": 0.0,
                    "negative_activity_max": 0.0,
                    "negative_score_p999_p95": 0.0,
                    "negative_score_max": 0.0,
                    "negative_score_positive_fraction": 0.0,
                    "negative_mean_algorithm_seconds": 0.0,
                }
            )
            continue
        window_counts = np.asarray([int(row["window_count"]) for row in selected], dtype=np.float64)
        active = np.asarray([float(row["active_fraction"]) for row in selected], dtype=np.float64)
        peaks = np.asarray([float(row["peak_activity_z"]) for row in selected], dtype=np.float64)
        activity_p999 = np.asarray([float(row["activity_p999_native"]) for row in selected], dtype=np.float64)
        activity_max = np.asarray([float(row["activity_max_native"]) for row in selected], dtype=np.float64)
        score_p999 = np.asarray([float(row["score_p999_native"]) for row in selected], dtype=np.float64)
        score_max = np.asarray([float(row["score_max_native"]) for row in selected], dtype=np.float64)
        score_positive = np.asarray([float(row["score_positive_fraction"]) for row in selected], dtype=np.float64)
        seconds = np.asarray([float(row["algorithm_seconds"]) for row in selected], dtype=np.float64)
        summary.append(
            {
                "algorithm": algorithm.name,
                "false_window_count": int(np.nansum(window_counts)),
                "false_windows_per_channel": float(np.nanmean(window_counts)),
                "false_channels_with_windows_rate": float(np.nanmean(window_counts > 0)),
                "false_active_fraction": float(np.nanmean(active)),
                "false_peak_z_p95": float(np.nanquantile(peaks, 0.95)) if peaks.size else 0.0,
                "false_peak_z_max": float(np.nanmax(peaks)) if peaks.size else 0.0,
                "negative_activity_p999_p95": _safe_quantile(activity_p999, 0.95),
                "negative_activity_max": float(np.nanmax(activity_max)) if activity_max.size else 0.0,
                "negative_score_p999_p95": _safe_quantile(score_p999, 0.95),
                "negative_score_max": float(np.nanmax(score_max)) if score_max.size else 0.0,
                "negative_score_positive_fraction": float(np.nanmean(score_positive)) if score_positive.size else 0.0,
                "negative_mean_algorithm_seconds": float(np.nanmean(seconds)) if seconds.size else 0.0,
            }
        )
    return summary


def _window_standardize(activity: np.ndarray, algorithm: CWTActivityAlgorithm) -> np.ndarray:
    return standardize_activity_for_pelt(
        activity,
        absolute_calibrated=algorithm.method == "single_map_cpro_activity",
        native_absolute=(
            str(algorithm.input_denoiser) == "absolute"
            and algorithm.method != "single_map_cpro_activity"
        ),
    )


def _negative_result_row(
    *,
    run: CWTActivityRun,
    config: CompressionBenchmarkConfig,
    reader: Any,
    algorithm: CWTActivityAlgorithm,
    channel_index: int,
    result: Any,
    cwt_seconds: float,
    algorithm_seconds: float,
) -> dict[str, Any]:
    activity_native = np.asarray(result.activity, dtype=np.float32)
    score_native = np.asarray(result.score_map, dtype=np.float32)
    activity_z = _window_standardize(activity_native, algorithm)
    windows = _negative_windows(activity_z, config, run)
    return {
        "algorithm": algorithm.name,
        "channel_index": int(channel_index),
        "frequency_mhz": float(reader.freqs_mhz[channel_index]),
        "records": int(reader.n_records),
        "window_count": int(len(windows)),
        "active_fraction": _windows_active_fraction(windows, int(reader.n_records)),
        "peak_activity_z": float(np.nanmax(activity_z)) if activity_z.size else 0.0,
        "p95_activity_z": _safe_quantile(activity_z, 0.95),
        "mean_activity_z": float(np.nanmean(activity_z)) if activity_z.size else 0.0,
        "activity_p99_native": _safe_quantile(activity_native, 0.99),
        "activity_p999_native": _safe_quantile(activity_native, 0.999),
        "activity_max_native": float(np.nanmax(activity_native)) if activity_native.size else 0.0,
        "score_p99_native": _safe_quantile(score_native, 0.99),
        "score_p999_native": _safe_quantile(score_native, 0.999),
        "score_max_native": float(np.nanmax(score_native)) if score_native.size else 0.0,
        "score_positive_fraction": float(np.nanmean(score_native > 0.0)) if score_native.size else 0.0,
        "cwt_seconds": float(cwt_seconds),
        "algorithm_seconds": float(algorithm_seconds),
    }


def _blocked_post_cwt_negative_rows(
    *,
    run: CWTActivityRun,
    reader: Any,
    config: CompressionBenchmarkConfig,
    algorithms: list[CWTActivityAlgorithm],
    periods: np.ndarray,
    channels: list[int],
    radius: int,
    started: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    block_targets = 4
    processed = 0
    for block_start in range(0, len(channels), block_targets):
        target_channels = channels[block_start : block_start + block_targets]
        width = min(int(reader.n_channels), len(target_channels) + 2 * radius)
        read_start = max(0, min(target_channels[0] - radius, int(reader.n_channels) - width))
        read_slice = slice(read_start, read_start + width)
        block = reader.read_block(slice(0, int(reader.n_records)), read_slice)
        data = np.asarray(block.data, dtype=np.float32)
        cwt_start = perf_counter()
        power = cwt_power_cube(
            data,
            periods,
            wavelet=config.wavelet,
            normalize_channels=True,
            method=config.cwt_method,
            backend=config.cwt_backend,
            cuda_device=config.cuda_device,
        )
        valid_power, valid_periods, _mask = crop_valid_periods(
            power,
            periods,
            config.candidate_period_min_records,
            config.candidate_period_max_records,
        )
        cwt_seconds = (perf_counter() - cwt_start) / float(len(target_channels))
        for channel_index in target_channels:
            target_offset = channel_index - read_start
            target_power = valid_power[:, :, target_offset]
            for algorithm in algorithms:
                algorithm_start = perf_counter()
                mode = str(algorithm.input_denoiser)
                reference_indices = _post_cwt_reference_indices(
                    target_offset=target_offset,
                    channel_count=data.shape[1],
                    mode=mode,
                )
                reference_power = (
                    valid_power[:, :, reference_indices]
                    if reference_indices
                    else None
                )
                result = compute_cwt_activity(
                    target_power,
                    valid_periods,
                    algorithm,
                    reference_power=reference_power,
                )
                rows.append(
                    _negative_result_row(
                        run=run,
                        config=config,
                        reader=reader,
                        algorithm=algorithm,
                        channel_index=channel_index,
                        result=result,
                        cwt_seconds=cwt_seconds,
                        algorithm_seconds=perf_counter() - algorithm_start,
                    )
                )
            processed += 1
            if run.progress_every > 0 and (
                processed == 1 or processed % run.progress_every == 0 or processed == len(channels)
            ):
                print(
                    f"[cwt-activity-negative] channel {processed}/{len(channels)} "
                    f"elapsed={perf_counter() - started:.1f}s",
                    flush=True,
                )
    return rows


def _run_negative_control(
    *,
    run: CWTActivityRun,
    reader: Any,
    config: CompressionBenchmarkConfig,
    algorithms: list[CWTActivityAlgorithm],
    periods: np.ndarray,
    noise_gain: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not run.negative_control:
        return [], _negative_control_summary([], algorithms), {"enabled": False}
    freq_slice = reader.freq_slice(float(run.negative_f_start_mhz), float(run.negative_f_stop_mhz))
    channels = list(range(int(freq_slice.start), int(freq_slice.stop)))
    if run.negative_channel_indices:
        allowed = set(channels)
        channels = sorted({int(channel) for channel in run.negative_channel_indices if int(channel) in allowed})
        if not channels:
            raise ValueError("negative_channel_indices do not intersect the configured frequency range")
    elif int(run.negative_max_channels) > 0:
        channels = channels[: int(run.negative_max_channels)]
    input_denoisers = _required_input_denoisers(algorithms)
    radius = _input_denoiser_radius(input_denoisers)
    rows: list[dict[str, Any]] = []
    started = perf_counter()
    post_cwt_modes = {
        mode for mode in input_denoisers if _post_cwt_reference_spec(mode) is not None
    }
    blocked_compatible = not run.negative_channel_indices
    if blocked_compatible and (
        input_denoisers == ("none",)
        or (set(input_denoisers).issubset({"none", *post_cwt_modes}) and post_cwt_modes)
    ):
        rows = _blocked_post_cwt_negative_rows(
            run=run,
            reader=reader,
            config=config,
            algorithms=algorithms,
            periods=periods,
            channels=channels,
            radius=radius,
            started=started,
        )
        summary_rows = _negative_control_summary(rows, algorithms)
        return rows, summary_rows, {
            "enabled": True,
            "frequency_mhz": [float(run.negative_f_start_mhz), float(run.negative_f_stop_mhz)],
            "window_method": str(run.negative_window_method),
            "window_boundary": "native_cpp_pelt",
            "channel_start": int(freq_slice.start),
            "channel_stop": int(freq_slice.stop),
            "evaluated_channel_count": len(channels),
            "channel_indices": channels,
            "execution": f"four-target CWT blocks with {radius}-channel frequency radius",
            "elapsed_seconds": float(perf_counter() - started),
            "summary_file": "negative_control_summary.csv",
            "cases_file": "negative_control_cases.csv",
        }
    for channel_offset, channel_index in enumerate(channels, start=1):
        channel_slice = _frequency_neighborhood_slice(channel_index, int(reader.n_channels), radius)
        block = reader.read_block(slice(0, int(reader.n_records)), channel_slice)
        neighborhood = np.asarray(block.data, dtype=np.float32)
        target_offset = channel_index - int(channel_slice.start)
        denoised, _duplicate = _frequency_denoised_inputs(
            neighborhood,
            neighborhood,
            target_offset=target_offset,
            modes=input_denoisers,
        )
        data_columns = [denoised[mode] for mode in input_denoisers]
        post_cwt_requested = any(
            _post_cwt_reference_spec(mode) is not None for mode in input_denoisers
        )
        if post_cwt_requested:
            data_columns += [neighborhood[:, index] for index in range(neighborhood.shape[1])]
        data = np.column_stack(data_columns)
        cwt_start = perf_counter()
        absolute_indices = [index for index, mode in enumerate(input_denoisers) if mode == "absolute"]
        if absolute_indices:
            normalized_indices = [index for index in range(data.shape[1]) if index not in set(absolute_indices)]
            if normalized_indices:
                data[:, normalized_indices] = robust_zscore_channels(data[:, normalized_indices])
        power = cwt_power_cube(
            data,
            periods,
            wavelet=config.wavelet,
            normalize_channels=not absolute_indices,
            method=config.cwt_method,
            backend=config.cwt_backend,
            cuda_device=config.cuda_device,
        )
        valid_power, valid_periods, _mask = crop_valid_periods(
            power,
            periods,
            config.candidate_period_min_records,
            config.candidate_period_max_records,
        )
        cwt_seconds = perf_counter() - cwt_start
        evaluated: list[tuple[CWTActivityAlgorithm, np.ndarray, np.ndarray, np.ndarray, float]] = []
        for algorithm in algorithms:
            algorithm_start = perf_counter()
            input_index = input_denoisers.index(str(algorithm.input_denoiser))
            reference_power = None
            if _post_cwt_reference_spec(str(algorithm.input_denoiser)) is not None:
                neighborhood_start = len(input_denoisers)
                reference_indices = _post_cwt_reference_indices(
                    target_offset=target_offset,
                    channel_count=neighborhood.shape[1],
                    mode=str(algorithm.input_denoiser),
                )
                reference_power = valid_power[
                    :,
                    :,
                    [neighborhood_start + index for index in reference_indices],
                ]
            result = compute_cwt_activity(
                valid_power[:, :, input_index],
                valid_periods,
                algorithm,
                reference_power=reference_power,
                noise_std=difference_noise_std(neighborhood[:, target_offset]),
                noise_gain=noise_gain,
            )
            activity_native = np.asarray(result.activity, dtype=np.float32)
            score_native = np.asarray(result.score_map, dtype=np.float32)
            activity_z = _window_standardize(activity_native, algorithm)
            evaluated.append(
                (
                    algorithm,
                    activity_native,
                    score_native,
                    activity_z,
                    perf_counter() - algorithm_start,
                )
            )
        pelt_start = perf_counter()
        pelt_results = segment_activity_batch_with_pelt(
            np.stack([item[1] for item in evaluated]),
            pelt_parameters_from_config(config),
            activities_z=np.stack([item[3] for item in evaluated]),
            threads=int(config.pelt_threads),
        )
        pelt_seconds_each = (perf_counter() - pelt_start) / float(max(1, len(evaluated)))
        for evaluated_row, pelt_result in zip(evaluated, pelt_results, strict=True):
            algorithm, activity_native, score_native, activity_z, algorithm_seconds = evaluated_row
            windows = [dict(window) for window in pelt_result.windows]
            rows.append(
                {
                    "algorithm": algorithm.name,
                    "channel_index": int(channel_index),
                    "frequency_mhz": float(reader.freqs_mhz[channel_index]),
                    "records": int(reader.n_records),
                    "window_count": int(len(windows)),
                    "active_fraction": _windows_active_fraction(windows, int(reader.n_records)),
                    "peak_activity_z": float(np.nanmax(activity_z)) if activity_z.size else 0.0,
                    "p95_activity_z": float(np.nanquantile(activity_z, 0.95)) if activity_z.size else 0.0,
                    "mean_activity_z": float(np.nanmean(activity_z)) if activity_z.size else 0.0,
                    "activity_p99_native": _safe_quantile(activity_native, 0.99),
                    "activity_p999_native": _safe_quantile(activity_native, 0.999),
                    "activity_max_native": float(np.nanmax(activity_native)) if activity_native.size else 0.0,
                    "score_p99_native": _safe_quantile(score_native, 0.99),
                    "score_p999_native": _safe_quantile(score_native, 0.999),
                    "score_max_native": float(np.nanmax(score_native)) if score_native.size else 0.0,
                    "score_positive_fraction": float(np.nanmean(score_native > 0.0)) if score_native.size else 0.0,
                    "cwt_seconds": float(cwt_seconds),
                    "algorithm_seconds": float(algorithm_seconds + pelt_seconds_each),
                }
            )
        if run.progress_every > 0 and (
            channel_offset == 1
            or channel_offset % run.progress_every == 0
            or channel_offset == len(channels)
        ):
            print(
                f"[cwt-activity-negative] channel {channel_offset}/{len(channels)} "
                f"elapsed={perf_counter() - started:.1f}s",
                flush=True,
            )
    summary_rows = _negative_control_summary(rows, algorithms)
    payload = {
        "enabled": True,
        "frequency_mhz": [float(run.negative_f_start_mhz), float(run.negative_f_stop_mhz)],
        "window_method": str(run.negative_window_method),
        "window_boundary": "native_cpp_pelt",
        "pelt_threads": int(config.pelt_threads),
        "channel_start": int(freq_slice.start),
        "channel_stop": int(freq_slice.stop),
        "evaluated_channel_count": len(channels),
        "channel_indices": channels,
        "elapsed_seconds": float(perf_counter() - started),
        "summary_file": "negative_control_summary.csv",
        "cases_file": "negative_control_cases.csv",
    }
    return rows, summary_rows, payload


def _best_score_map_timeseries(
    score_map: np.ndarray,
    periods: np.ndarray,
    widths: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = int(score_map.shape[1])
    best_contrast = np.full(records, -np.inf, dtype=np.float32)
    best_ratio = np.zeros(records, dtype=np.float32)
    best_periods = np.full(records, np.nan, dtype=np.float32)
    for width in widths:
        contrast, ratio, period_values = _best_band_timeseries(score_map, periods, width)
        update = contrast > best_contrast
        best_contrast = np.where(update, contrast, best_contrast)
        best_ratio = np.where(update, ratio, best_ratio)
        best_periods = np.where(update, period_values, best_periods)
    best_contrast[~np.isfinite(best_contrast)] = 0.0
    return (
        best_contrast.astype(np.float32, copy=False),
        best_ratio.astype(np.float32, copy=False),
        best_periods.astype(np.float32, copy=False),
    )


def _evaluate_activity_algorithm(
    prepared: _PreparedActivityCase,
    algorithm: CWTActivityAlgorithm,
    config: CompressionBenchmarkConfig,
) -> dict[str, Any]:
    reduce_start = perf_counter()
    input_mode = str(algorithm.input_denoiser)
    result = compute_cwt_activity(
        prepared.cwt_power[input_mode],
        prepared.valid_periods,
        algorithm,
        reference_power=prepared.reference_cwt_power.get(input_mode),
        noise_std=prepared.noise_std,
        noise_gain=prepared.noise_gain,
    )
    baseline_result = compute_cwt_activity(
        prepared.baseline_cwt_power[input_mode],
        prepared.valid_periods,
        algorithm,
        reference_power=prepared.baseline_reference_cwt_power.get(input_mode),
        noise_std=prepared.noise_std,
        noise_gain=prepared.noise_gain,
    )
    activity = np.asarray(result.activity, dtype=np.float32)
    baseline_activity = np.asarray(baseline_result.activity, dtype=np.float32)
    activity_z = _window_standardize(activity, algorithm)
    score_map = np.asarray(result.score_map, dtype=np.float32)
    baseline_score_map = np.asarray(baseline_result.score_map, dtype=np.float32)
    score_map[~np.isfinite(score_map)] = 0.0
    baseline_score_map[~np.isfinite(baseline_score_map)] = 0.0

    truth_start = max(0, min(int(prepared.truth["record_start"]), activity.size))
    truth_stop = max(truth_start + 1, min(int(prepared.truth["record_stop"]), activity.size))
    truth_slice = slice(truth_start, truth_stop)
    truth_activity = activity[truth_slice]
    paired_baseline_activity = baseline_activity[truth_slice]
    baseline_activity_p999 = _safe_quantile(baseline_activity, 0.999)
    truth_activity_p95 = _safe_quantile(truth_activity, 0.95)

    truth_period = float(prepared.truth["period_records"])
    log_error = np.abs(
        np.log(np.maximum(np.asarray(prepared.valid_periods, dtype=np.float64), 1e-12) / max(truth_period, 1e-12))
    )
    period_center = int(np.nanargmin(log_error))
    period_slice = slice(max(0, period_center - 1), min(score_map.shape[0], period_center + 2))
    truth_band_score = score_map[period_slice, truth_slice]
    paired_baseline_band = baseline_score_map[period_slice, truth_slice]
    baseline_score_p999 = _safe_quantile(baseline_score_map, 0.999)
    truth_band_score_p95 = _safe_quantile(truth_band_score, 0.95)
    time_band_contrast, time_band_ratio, time_band_periods = _best_score_map_timeseries(
        score_map,
        prepared.valid_periods,
        config.band_widths,
    )
    reduce_seconds = perf_counter() - reduce_start
    window_start = perf_counter()
    truth_metrics = _activity_truth_metrics(activity_z, prepared.truth)
    window_metrics = _window_rows(activity_z, prepared.truth, score_map, prepared.valid_periods, config)
    window_seconds = perf_counter() - window_start
    algorithm_seconds = reduce_seconds + window_seconds
    peak_record = int(truth_metrics["peak_record"])
    peak_period = float(time_band_periods[peak_record]) if time_band_periods.size else math.nan
    return {
        "case_id": prepared.case_id,
        "algorithm": algorithm.name,
        "algorithm_family": algorithm.family,
        "algorithm_description": algorithm.description,
        "records": prepared.records,
        "channels": prepared.channels,
        "background_mode": prepared.background_mode,
        "background_source": prepared.background_source,
        "background_record_start": prepared.background_record_start,
        "background_record_stop": prepared.background_record_stop,
        "background_channel_start": prepared.background_channel_start,
        "background_channel_stop": prepared.background_channel_stop,
        "noise_std": prepared.noise_std,
        "signal_model": prepared.injection.signal_model,
        "amplitude": float(prepared.injection.amplitude),
        "amplitude_factor": float(prepared.injection.amplitude / max(prepared.noise_std, 1e-12)),
        "period_records": truth_period,
        "duration_records": int(prepared.truth["duration_records"]),
        "duration_fraction": float(prepared.truth["duration_records"]) / float(max(1, prepared.records)),
        "record_start": int(prepared.truth["record_start"]),
        "record_stop": int(prepared.truth["record_stop"]),
        "channel_index": int(prepared.truth["channel_index"]),
        "bandwidth_channels": float(prepared.injection.bandwidth_channels),
        "drift_channels": float(prepared.injection.drift_channels),
        "duty_cycle": float(prepared.injection.duty_cycle),
        "peak_record": peak_record,
        "peak_in_truth": int(truth_metrics["peak_in_truth"]),
        "peak_activity_z": float(truth_metrics["peak_activity_z"]),
        "truth_peak_z": float(truth_metrics["truth_peak_z"]),
        "truth_mean_z": float(truth_metrics["truth_mean_z"]),
        "outside_p95_z": float(truth_metrics["outside_p95_z"]),
        "time_contrast_z": float(truth_metrics["truth_peak_z"]) - float(truth_metrics["outside_p95_z"]),
        "peak_global_band_contrast": float(time_band_contrast[peak_record]) if time_band_contrast.size else 0.0,
        "peak_period_concentration": float(time_band_ratio[peak_record]) if time_band_ratio.size else 0.0,
        "peak_band_period_records": peak_period,
        "peak_band_period_error_fraction": _period_error_fraction(peak_period, truth_period),
        "window_count": int(window_metrics["window_count"]),
        "truth_window_hit": int(window_metrics["truth_window_hit"]),
        "truth_window_overlap_fraction": float(window_metrics["truth_window_overlap_fraction"]),
        "best_window_local_band_contrast": float(window_metrics["best_window_local_band_contrast"]),
        "truth_window_local_band_contrast": float(window_metrics["truth_window_local_band_contrast"]),
        "truth_window_period_records": float(window_metrics["truth_window_period_records"]),
        "truth_window_period_error_fraction": _period_error_fraction(
            float(window_metrics["truth_window_period_records"]),
            truth_period,
        ),
        "truth_activity_p95_native": truth_activity_p95,
        "baseline_activity_p999_native": baseline_activity_p999,
        "paired_activity_gain_mean": float(np.nanmean(truth_activity - paired_baseline_activity)),
        "paired_activity_exceedance_fraction": float(np.nanmean(truth_activity > baseline_activity_p999)),
        "paired_activity_detected": int(truth_activity_p95 > baseline_activity_p999),
        "truth_band_score_p95_native": truth_band_score_p95,
        "baseline_score_p999_native": baseline_score_p999,
        "paired_band_gain_mean": float(np.nanmean(truth_band_score - paired_baseline_band)),
        "paired_band_exceedance_fraction": float(np.nanmean(truth_band_score > baseline_score_p999)),
        "paired_band_detected": int(truth_band_score_p95 > baseline_score_p999),
        "preprocess_seconds": float(prepared.preprocess_seconds),
        "reduce_seconds": float(reduce_seconds),
        "window_seconds": float(window_seconds),
        "algorithm_seconds": float(algorithm_seconds),
        "algorithm_over_preprocess_ratio": float(algorithm_seconds / max(prepared.preprocess_seconds, 1e-12)),
    }


def _add_group_denoising_metrics(
    group_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
) -> None:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in component_rows:
        buckets[(str(row["group_id"]), str(row["algorithm"]))].append(row)
    for row in group_rows:
        selected = buckets[(str(row["group_id"]), str(row["algorithm"]))]
        for field in DENOISING_CASE_FIELDNAMES:
            row[field] = max(float(component[field]) for component in selected)
        row["paired_activity_detected"] = max(int(component["paired_activity_detected"]) for component in selected)
        row["paired_band_detected"] = max(int(component["paired_band_detected"]) for component in selected)


def _add_denoising_summary_metrics(
    summary_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
) -> None:
    for summary in summary_rows:
        algorithm = str(summary["algorithm"])
        selected = [row for row in group_rows if str(row["algorithm"]) == algorithm]
        if not selected:
            for field in DENOISING_SUMMARY_FIELDNAMES:
                summary[field] = 0.0
            continue

        def values(field: str) -> np.ndarray:
            return np.asarray([float(row[field]) for row in selected], dtype=np.float64)

        truth_activity = values("truth_activity_p95_native")
        truth_band = values("truth_band_score_p95_native")
        raw_activity_threshold = max(float(summary["negative_activity_p999_p95"]), 1e-12)
        raw_band_threshold = max(float(summary["negative_score_p999_p95"]), 1e-12)
        summary.update(
            {
                "paired_activity_detection_rate": float(np.nanmean(values("paired_activity_detected"))),
                "paired_band_detection_rate": float(np.nanmean(values("paired_band_detected"))),
                "mean_paired_activity_gain": float(np.nanmean(values("paired_activity_gain_mean"))),
                "mean_paired_band_gain": float(np.nanmean(values("paired_band_gain_mean"))),
                "mean_paired_activity_exceedance_fraction": float(
                    np.nanmean(values("paired_activity_exceedance_fraction"))
                ),
                "mean_paired_band_exceedance_fraction": float(
                    np.nanmean(values("paired_band_exceedance_fraction"))
                ),
                "raw_safe_activity_recall": float(np.nanmean(truth_activity > raw_activity_threshold)),
                "raw_safe_band_recall": float(np.nanmean(truth_band > raw_band_threshold)),
                "median_truth_to_raw_activity_ratio": float(np.nanmedian(truth_activity / raw_activity_threshold)),
                "median_truth_to_raw_band_ratio": float(np.nanmedian(truth_band / raw_band_threshold)),
                "negative_window_control": 1.0 / (1.0 + float(summary["false_windows_per_channel"])),
                "negative_map_sparsity": 1.0 - min(
                    max(float(summary["negative_score_positive_fraction"]), 0.0),
                    1.0,
                ),
                "negative_map_clean_rate": max(
                    0.0,
                    1.0
                    - float(summary["negative_score_positive_fraction"])
                    / NEGATIVE_MAP_TARGET_FRACTION,
                ),
            }
        )


def _compat_algorithms(algorithms: list[CWTActivityAlgorithm]) -> list[Any]:
    """Use the existing summary reducer without coupling to old algorithm math."""
    return algorithms


def run_cwt_activity_rank(run: CWTActivityRun) -> dict[str, Any]:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    if str(run.negative_window_method).strip().lower() != "pelt":
        raise ValueError("negative_window_method must be 'pelt'; threshold windows are not scientific stage 2")
    config = activity_config_from_cwt(run)
    reader = open_spectrum_reader(run.input_path)
    payload = load_injection_config(run.injection_config)
    specs = make_injections_from_config(
        payload,
        records=reader.n_records,
        channels=reader.n_channels,
        freqs_mhz=reader.freqs_mhz,
    )
    specs = _limit_specs_by_family(specs, run.max_groups_per_family)
    algorithms = resolve_cwt_activity_algorithms(run.algorithms)
    if run.strict_single_map:
        invalid = [
            algorithm.name
            for algorithm in algorithms
            if algorithm.input_denoiser not in {"none", "absolute"} or algorithm.complexity != "O(P*T)"
        ]
        if invalid:
            raise ValueError(
                "strict single-map rank rejects non-unary or non-linear algorithms: "
                + ", ".join(invalid)
            )
    input_denoisers = _required_input_denoisers(algorithms)
    if run.strict_single_map and not set(input_denoisers).issubset({"none", "absolute"}):
        raise AssertionError("strict single-map preparation may read only the target channel")
    periods = period_grid_records(
        config.period_min_records,
        config.period_max_records,
        config.period_count,
        config.period_spacing,
    )
    valid_periods = periods[
        cpro_period_mask(
            periods,
            config.candidate_period_min_records,
            config.candidate_period_max_records,
        )
    ]
    noise_gain = impulse_cwt_noise_gain(
        valid_periods,
        wavelet=config.wavelet,
        method=config.cwt_method,
    )
    specs_by_id = {spec.injection_id: spec for spec in specs}
    component_rows: list[dict[str, Any]] = []
    prepared_index: list[dict[str, Any]] = []
    started = perf_counter()
    for index, spec in enumerate(specs, start=1):
        prepared, _baseline, _injected = prepare_activity_component(
            reader=reader,
            spec=spec,
            config=config,
            periods=periods,
            input_denoisers=input_denoisers,
            noise_gain=noise_gain,
        )
        group_id = injection_group_id(spec.injection_id)
        prepared_index.append(
            {
                "component_id": spec.injection_id,
                "group_id": group_id,
                "channel_index": int(round(float(spec.channel_center))),
                "frequency_mhz": float(reader.freqs_mhz[int(round(float(spec.channel_center)))]),
                "local_record_start": prepared.background_record_start,
                "local_record_stop": prepared.background_record_stop,
            }
        )
        for algorithm in algorithms:
            row = _evaluate_activity_algorithm(prepared, algorithm, config)
            row["group_id"] = group_id
            component_rows.append(row)
        if run.progress_every > 0 and (
            index == 1 or index % run.progress_every == 0 or index == len(specs)
        ):
            print(
                f"[cwt-activity-rank] component {index}/{len(specs)} "
                f"elapsed={perf_counter() - started:.1f}s",
                flush=True,
            )

    group_rows = _group_rows(component_rows, specs_by_id, np.asarray(reader.freqs_mhz))
    _add_group_denoising_metrics(group_rows, component_rows)
    summary_rows = _scientific_rank(
        _summary_rows(
            group_rows,
            _compat_algorithms(algorithms),
            max_algorithm_over_preprocess_ratio=config.max_algorithm_over_preprocess_ratio,
        )
    )
    for row in summary_rows:
        row["injection_rank_score"] = float(row["rank_score"])

    negative_rows, negative_summary_rows, negative_payload = _run_negative_control(
        run=run,
        reader=reader,
        config=config,
        algorithms=algorithms,
        periods=periods,
        noise_gain=noise_gain,
    )
    negative_by_algorithm = {
        str(row["algorithm"]): row
        for row in negative_summary_rows
    }
    for row in summary_rows:
        negative = negative_by_algorithm.get(str(row["algorithm"]), {})
        for field in NEGATIVE_CONTROL_SUMMARY_FIELDNAMES:
            if field == "algorithm":
                continue
            row[field] = negative.get(field, 0.0)
    _add_denoising_summary_metrics(summary_rows, group_rows)
    summary_rows = _combined_rank(summary_rows)

    component_fields = CASE_FIELDNAMES + DENOISING_CASE_FIELDNAMES + ["group_id"]
    group_fields = CASE_FIELDNAMES + DENOISING_CASE_FIELDNAMES + GROUP_EXTRA_FIELDS
    _write_csv(run.output_dir / "component_cases.csv", component_fields, component_rows)
    _write_csv(run.output_dir / "group_cases.csv", group_fields, group_rows)
    _write_csv(run.output_dir / "negative_control_cases.csv", NEGATIVE_CONTROL_FIELDNAMES, negative_rows)
    _write_csv(
        run.output_dir / "negative_control_summary.csv",
        NEGATIVE_CONTROL_SUMMARY_FIELDNAMES,
        negative_summary_rows,
    )
    _write_csv(run.output_dir / "cwt_activity_summary.csv", CWT_ACTIVITY_SUMMARY_FIELDNAMES, summary_rows)
    if prepared_index:
        _write_csv(run.output_dir / "component_index.csv", list(prepared_index[0]), prepared_index)
    shutil.copy2(run.injection_config, run.output_dir / "injection_config.json")
    shutil.copy2(run.cwt_config, run.output_dir / "cwt_config.json")

    all_algorithms = cwt_activity_algorithm_map()
    selected_names = {algorithm.name for algorithm in algorithms}
    algorithm_map = {
        name: {
            **configuration,
            "selected": name in selected_names,
        }
        for name, configuration in all_algorithms.items()
    }
    (run.output_dir / "cwt_activity_algorithm_map.json").write_text(
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
        "algorithm_map_file": "cwt_activity_algorithm_map.json",
        "paradigm": {
            "input": "raw cropped CWT power map with shape (periods, records)",
            "constraint": (
                "strict unary P(period,time)->activity(time); target channel only; O(P*T)"
                if run.strict_single_map
                else "each candidate owns its denoising front-end; compression is evaluated only after denoising"
            ),
            "candidate_output": "each algorithm returns its own 1D activity and diagnostic 2D score map",
            "stage2": "all positive and real-negative activity axes use the same native C++ PELT boundary",
            "shared_code": "paired baseline/injected CWT computation, PELT windowing, truth scoring, raw-negative scoring, same-signal group aggregation",
        },
        "reproducibility": {
            "input_size_bytes": run.input_path.stat().st_size,
            "input_sha256": file_sha256(run.input_path),
            "injection_config_sha256": file_sha256(run.injection_config),
            "cwt_config_sha256": file_sha256(run.cwt_config),
            "retained_injection_config": "injection_config.json",
            "retained_cwt_config": "cwt_config.json",
            "pytest_node": (
                "tests/perf/test_compression_benchmark.py::"
                "test_cwt_activity_real_ce4_rank"
            ),
            "environment": {
                "CWIPSS_RUN_PERF": "1",
                "CWIPSS_ACTIVITY_OUTPUT": str(run.output_dir),
                "CWIPSS_ACTIVITY_INPUT": str(run.input_path),
                "CWIPSS_ACTIVITY_INJECTIONS": str(run.injection_config),
                "CWIPSS_ACTIVITY_CWT": str(run.cwt_config),
                "CWIPSS_ACTIVITY_ALGORITHMS": ",".join(run.algorithms),
                "CWIPSS_ACTIVITY_BACKEND": run.cwt_backend,
                "CWIPSS_ACTIVITY_PELT_THREADS": str(run.pelt_threads),
                "CWIPSS_ACTIVITY_CANDIDATE_MAX": str(run.candidate_period_max_records),
                "CWIPSS_ACTIVITY_NEGATIVE_CONTROL": str(int(run.negative_control)),
                "CWIPSS_ACTIVITY_NEGATIVE_F_START": str(run.negative_f_start_mhz),
                "CWIPSS_ACTIVITY_NEGATIVE_F_STOP": str(run.negative_f_stop_mhz),
                "CWIPSS_ACTIVITY_NEGATIVE_MAX_CHANNELS": str(run.negative_max_channels),
                "CWIPSS_ACTIVITY_NEGATIVE_WINDOW_METHOD": str(run.negative_window_method),
                "CWIPSS_ACTIVITY_STRICT_SINGLE_MAP": str(int(run.strict_single_map)),
                "CWIPSS_ACTIVITY_MAX_GROUPS_PER_FAMILY": str(run.max_groups_per_family),
            },
            "command": [
                "python",
                "-m",
                "pytest",
                (
                    "tests/perf/test_compression_benchmark.py::"
                    "test_cwt_activity_real_ce4_rank"
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
            "mode": "paired_bright_band_retention_plus_raw_lowfreq_2d_suppression",
            "injection_only_weights": SCIENTIFIC_RANK_WEIGHTS,
            "combined_weights": COMBINED_RANK_WEIGHTS,
            "raw_safe_threshold": "95th percentile across channels of each channel's native 99.9th-percentile response",
            "negative_map_target_fraction": NEGATIVE_MAP_TARGET_FRACTION,
            "timing_note": "Runtime is recorded but does not affect rank; all selected denoisers must remain O(P*T).",
        },
        "negative_control": negative_payload,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "best_algorithm": str(summary_rows[0]["algorithm"]) if summary_rows else "",
        "metric_leaders": {
            "rank_score": str(summary_rows[0]["algorithm"]) if summary_rows else "",
            "peak_in_truth_rate": max(
                summary_rows,
                key=lambda row: float(row["peak_in_truth_rate"]),
            )["algorithm"] if summary_rows else "",
            "truth_window_hit_rate": max(
                summary_rows,
                key=lambda row: float(row["truth_window_hit_rate"]),
            )["algorithm"] if summary_rows else "",
            "mean_time_contrast_z": max(
                summary_rows,
                key=lambda row: float(row["mean_time_contrast_z"]),
            )["algorithm"] if summary_rows else "",
            "mean_peak_global_band_contrast": max(
                summary_rows,
                key=lambda row: float(row["mean_peak_global_band_contrast"]),
            )["algorithm"] if summary_rows else "",
            "mean_peak_period_concentration": max(
                summary_rows,
                key=lambda row: float(row["mean_peak_period_concentration"]),
            )["algorithm"] if summary_rows else "",
        },
        "summary_rows": summary_rows,
    }
    (run.output_dir / "cwt_activity_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True)
    )
    return result


def rerank_cwt_activity_output(output_dir: str | Path) -> dict[str, Any]:
    """Reapply rank weights to retained per-group and negative-control metrics."""
    root = Path(output_dir)
    summary_json = root / "cwt_activity_summary.json"
    payload = json.loads(summary_json.read_text())
    with (root / "group_cases.csv").open(newline="") as fp:
        group_rows = list(csv.DictReader(fp))
    with (root / "negative_control_summary.csv").open(newline="") as fp:
        negative_rows = list(csv.DictReader(fp))

    algorithm_names = tuple(str(name) for name in payload["config"]["algorithms"])
    algorithms = resolve_cwt_activity_algorithms(algorithm_names)
    summary_rows = _scientific_rank(
        _summary_rows(
            group_rows,
            _compat_algorithms(algorithms),
            max_algorithm_over_preprocess_ratio=float(
                payload["config"].get("max_algorithm_over_preprocess_ratio", 1.0)
            ),
        )
    )
    for row in summary_rows:
        row["injection_rank_score"] = float(row["rank_score"])
    negative_by_algorithm = {str(row["algorithm"]): row for row in negative_rows}
    for row in summary_rows:
        negative = negative_by_algorithm[str(row["algorithm"])]
        for field in NEGATIVE_CONTROL_SUMMARY_FIELDNAMES:
            if field != "algorithm":
                row[field] = float(negative[field])
    _add_denoising_summary_metrics(summary_rows, group_rows)
    summary_rows = _combined_rank(summary_rows)

    _write_csv(root / "cwt_activity_summary.csv", CWT_ACTIVITY_SUMMARY_FIELDNAMES, summary_rows)
    payload["rank_score_basis"]["combined_weights"] = COMBINED_RANK_WEIGHTS
    payload["rank_score_basis"]["negative_map_target_fraction"] = NEGATIVE_MAP_TARGET_FRACTION
    payload["rank_score_basis"]["reranked_from_saved_cases"] = True
    payload["best_algorithm"] = str(summary_rows[0]["algorithm"]) if summary_rows else ""
    payload["metric_leaders"] = {
        "rank_score": str(summary_rows[0]["algorithm"]) if summary_rows else "",
        "peak_in_truth_rate": max(summary_rows, key=lambda row: float(row["peak_in_truth_rate"]))["algorithm"] if summary_rows else "",
        "truth_window_hit_rate": max(summary_rows, key=lambda row: float(row["truth_window_hit_rate"]))["algorithm"] if summary_rows else "",
        "raw_safe_band_recall": max(summary_rows, key=lambda row: float(row["raw_safe_band_recall"]))["algorithm"] if summary_rows else "",
        "raw_safe_activity_recall": max(summary_rows, key=lambda row: float(row["raw_safe_activity_recall"]))["algorithm"] if summary_rows else "",
        "negative_map_sparsity": max(summary_rows, key=lambda row: float(row["negative_map_sparsity"]))["algorithm"] if summary_rows else "",
    }
    payload["summary_rows"] = summary_rows
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    return payload


def run_strict_top_negative_verification(
    run: CWTActivityRun,
    algorithm_name: str | None = None,
) -> dict[str, Any]:
    """Run full PELT negative control only for the selected top candidate."""
    summary_path = run.output_dir / "cwt_activity_summary.json"
    payload = json.loads(summary_path.read_text())
    selected_name = str(algorithm_name or payload["best_algorithm"])
    algorithms = resolve_cwt_activity_algorithms((selected_name,))
    strict_run = replace(
        run,
        algorithms=(selected_name,),
        negative_control=True,
        negative_window_method="pelt",
    )
    config = activity_config_from_cwt(strict_run)
    reader = open_spectrum_reader(strict_run.input_path)
    periods = period_grid_records(
        config.period_min_records,
        config.period_max_records,
        config.period_count,
        config.period_spacing,
    )
    valid_periods = periods[
        cpro_period_mask(
            periods,
            config.candidate_period_min_records,
            config.candidate_period_max_records,
        )
    ]
    noise_gain = impulse_cwt_noise_gain(
        valid_periods,
        wavelet=config.wavelet,
        method=config.cwt_method,
    )
    rows, summary_rows, metadata = _run_negative_control(
        run=strict_run,
        reader=reader,
        config=config,
        algorithms=algorithms,
        periods=periods,
        noise_gain=noise_gain,
    )
    cases_name = "strict_pelt_negative_cases.csv"
    summary_name = "strict_pelt_negative_summary.csv"
    _write_csv(run.output_dir / cases_name, NEGATIVE_CONTROL_FIELDNAMES, rows)
    _write_csv(run.output_dir / summary_name, NEGATIVE_CONTROL_SUMMARY_FIELDNAMES, summary_rows)
    metadata = {
        **metadata,
        "cases_file": cases_name,
        "summary_file": summary_name,
    }
    result = {
        "algorithm": selected_name,
        "window_method": "pelt",
        "cases_file": cases_name,
        "summary_file": summary_name,
        "metadata": metadata,
        "summary": summary_rows[0],
    }
    payload["strict_top_negative_control"] = result
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    return result


__all__ = [
    "CWTActivityRun",
    "activity_config_from_cwt",
    "prepare_activity_component",
    "rerank_cwt_activity_output",
    "run_strict_top_negative_verification",
    "run_cwt_activity_rank",
    "largest_complete_2c",
]
