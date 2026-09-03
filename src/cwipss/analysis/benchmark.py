"""Injection benchmark orchestration and recovery metrics."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..signal.cpro import CPRO_DETECTOR, CPROParameters, cpro_period_mask, impulse_cwt_noise_gain
from ..signal.cprf import CPRFParameters
from ..signal.cwt import cwt_power_cube, period_grid_records
from ..signal.detection import add_candidate_ids, detect_block_periods
from ..signal.windows import require_native_pelt
from .injection import BackgroundData, ce4_background, inject_many, synthetic_background
from ..data.schemas import (
    INJECTION_PERFORMANCE_FIELDNAMES,
    INJECTION_RESULT_FIELDNAMES,
    INJECTION_TRUTH_FIELDNAMES,
    RAW_CANDIDATE_FIELDNAMES,
    RAW_CANDIDATE_SCHEMA_VERSION,
    REVIEWED_CANDIDATE_FIELDNAMES,
    TIME_WINDOW_FIELDNAMES,
    VALIDATION_FIELDNAMES,
    VALIDATION_REVIEWED_FIELDNAMES,
    normalize_candidate_row,
)
from ..runtime import runtime_info
from .simulation import InjectionSpec
from .statistics import review_validation_rows
from .validation import (
    ValidationConfig,
    aggregate_frequency_series,
    best_acf_peak,
    best_fold_period,
    fft_periodogram_peak,
    period_grid,
    refined_period_from_metrics,
    shuffle_null_pvalue,
    validation_period_bounds,
)
from .veto import VetoConfig, VetoContext, review_candidates


@dataclass(frozen=True)
class CWTBenchmarkConfig:
    wavelet: str = "cmor1.5-1.0"
    cwt_method: str = "fft"
    cwt_backend: str = "cpu"
    cuda_device: int = 0
    period_min_records: float = 2.0
    period_max_records: float = 512.0
    period_count: int = 96
    period_spacing: str = "log"
    block_channels: int = 128
    time_aggregation: str = "p95"
    aggregation_percentile: float = 95.0
    detector: str = CPRO_DETECTOR
    candidate_period_min_records: float = 10.0
    candidate_period_max_records: float = 200.0
    cpro_threshold_snr: float = 32.0
    cpro_texture_quantile: float = 0.9375
    cpro_period_center_bins: int = 3
    cpro_period_context_bins: int = 15
    cpro_min_period_contrast: float = 1.5
    cpro_period_support_bins: int = 3
    cpro_shape_power_softness: float = 1.0
    cpro_shape_contrast_softness: float = 0.10
    cpro_continuity_decay: float = 0.995
    cpro_continuity_power: float = 2.0
    cpro_min_continuity_mean: float = 0.47
    cpro_min_ridge_lock: float = 0.94
    pelt_penalty: float = 16.0
    pelt_min_size_records: int = 64
    pelt_jump_records: int = 8
    pelt_threads: int = 1
    window_min_activity_mean: float = 0.05
    window_merge_gap_records: int = 0
    cprf_threshold_snr: float = 32.0
    cprf_texture_quantile: float = 0.9375
    cprf_smooth_bins: int = 3
    cprf_peak_band_fraction: float = 0.50
    cprf_min_width_bins: int = 3
    cprf_min_peak_strength: float = 1.25
    cprf_min_integrated_strength: float = 0.0
    cprf_min_band_persistence: float = 0.40
    cprf_min_band_concentration: float = 0.50
    cprf_min_local_contrast: float = 1.20
    cprf_harmonic_weight: float = 0.20
    cprf_harmonic_min_relative: float = 0.12
    cprf_harmonic_window_scale: float = 1.25
    cprf_max_peak_hypotheses: int = 8
    max_candidates_per_channel: int | str = "auto"
    max_candidates_per_record: float = 3.0 / 4096.0
    progress_enabled: bool = True
    progress_leave: bool = False


@dataclass(frozen=True)
class MatchConfig:
    min_time_overlap: float = 0.30
    min_freq_overlap: float = 0.30
    max_period_error_fraction: float = 0.50


def write_rows_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _span_overlap(start_a: float, stop_a: float, start_b: float, stop_b: float) -> float:
    lo_a, hi_a = sorted([float(start_a), float(stop_a)])
    lo_b, hi_b = sorted([float(start_b), float(stop_b)])
    span_a = abs(hi_a - lo_a)
    span_b = abs(hi_b - lo_b)
    if span_a <= 1e-12 and span_b <= 1e-12:
        return 1.0 if abs(lo_a - lo_b) <= 1e-12 else 0.0
    if span_a <= 1e-12:
        return 1.0 if lo_b <= lo_a <= hi_b else 0.0
    if span_b <= 1e-12:
        return 1.0 if lo_a <= lo_b <= hi_a else 0.0
    overlap = max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))
    denom = max(1e-12, min(span_a, span_b))
    return overlap / denom


def _channel_progress(total: int, run_id: str, enabled: bool, leave: bool):
    if not enabled:
        return None
    from tqdm.auto import tqdm

    return tqdm(
        total=max(0, int(total)),
        desc=f"CWT channels {run_id}",
        unit="ch",
        leave=bool(leave),
        dynamic_ncols=True,
    )


def _use_cuda_block_backend(backend: str, method: str, cuda_device: int) -> bool:
    backend_name = str(backend or "cpu").lower()
    if backend_name == "cuda":
        return True
    if backend_name != "auto" or method != "fft":
        return False
    try:
        from ..signal.cwt_cuda import cuda_available
    except ImportError:
        return False
    return cuda_available(device=int(cuda_device))


def run_cwt_candidate_search(
    data: np.ndarray,
    freqs_mhz: np.ndarray,
    source_name: str,
    tsamp_seconds: float,
    run_id: str,
    search_config: CWTBenchmarkConfig,
    veto_config: VetoConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    require_native_pelt()
    if search_config.detector != CPRO_DETECTOR:
        raise ValueError(f"Unsupported detector {search_config.detector!r}; required detector is {CPRO_DETECTOR!r}")
    CPROParameters(
        threshold_snr=search_config.cpro_threshold_snr,
        texture_quantile=search_config.cpro_texture_quantile,
        period_center_bins=search_config.cpro_period_center_bins,
        period_context_bins=search_config.cpro_period_context_bins,
        min_period_contrast=search_config.cpro_min_period_contrast,
        period_support_bins=search_config.cpro_period_support_bins,
        shape_power_softness=search_config.cpro_shape_power_softness,
        shape_contrast_softness=search_config.cpro_shape_contrast_softness,
    ).validate()
    cprf_params = CPRFParameters(
        threshold_snr=search_config.cprf_threshold_snr,
        texture_quantile=search_config.cprf_texture_quantile,
        smooth_bins=search_config.cprf_smooth_bins,
        peak_band_fraction=search_config.cprf_peak_band_fraction,
        min_width_bins=search_config.cprf_min_width_bins,
        min_peak_strength=search_config.cprf_min_peak_strength,
        min_integrated_strength=search_config.cprf_min_integrated_strength,
        min_band_persistence=search_config.cprf_min_band_persistence,
        min_band_concentration=search_config.cprf_min_band_concentration,
        min_local_contrast=search_config.cprf_min_local_contrast,
        harmonic_weight=search_config.cprf_harmonic_weight,
        harmonic_min_relative=search_config.cprf_harmonic_min_relative,
        harmonic_window_scale=search_config.cprf_harmonic_window_scale,
        max_peak_hypotheses=search_config.cprf_max_peak_hypotheses,
    )
    cprf_params.validate()
    detector = detect_block_periods
    matrix = np.asarray(data, dtype=np.float32)
    periods = period_grid_records(
        search_config.period_min_records,
        search_config.period_max_records,
        search_config.period_count,
        search_config.period_spacing,
    )
    candidate_period_mask = cpro_period_mask(
        periods,
        search_config.candidate_period_min_records,
        search_config.candidate_period_max_records,
    )
    noise_gain = impulse_cwt_noise_gain(
        periods[candidate_period_mask],
        wavelet=search_config.wavelet,
        method=search_config.cwt_method,
    )
    all_candidates: list[dict[str, Any]] = []
    all_windows: list[dict[str, Any]] = []
    progress = _channel_progress(
        total=matrix.shape[1],
        run_id=run_id,
        enabled=search_config.progress_enabled,
        leave=search_config.progress_leave,
    )
    use_cuda_block_backend = _use_cuda_block_backend(
        search_config.cwt_backend,
        search_config.cwt_method,
        search_config.cuda_device,
    )
    if use_cuda_block_backend:
        from ..signal.cwt_cuda import _cupy, cwt_power_cube_cuda_gpu
        from ..signal.detection_cuda import detect_block_periods_cuda_power
    try:
        for block_index, block_start in enumerate(range(0, matrix.shape[1], search_config.block_channels), start=1):
            block_stop = min(block_start + int(search_config.block_channels), matrix.shape[1])
            halo = slice(block_start, block_stop)
            block_data = matrix[:, halo]
            block_freqs = freqs_mhz[halo]
            if use_cuda_block_backend:
                cp = _cupy()
                cp.cuda.Device(int(search_config.cuda_device)).use()
                raw_device = cp.asarray(block_data, dtype=cp.float32)
                power = cwt_power_cube_cuda_gpu(
                    raw_device,
                    wavelet=search_config.wavelet,
                    periods=periods,
                    method=search_config.cwt_method,
                    device=search_config.cuda_device,
                    normalize_channels=False,
                )
                detector = detect_block_periods_cuda_power
            else:
                raw_device = block_data
                power = cwt_power_cube(
                    block_data,
                    wavelet=search_config.wavelet,
                    periods=periods,
                    method=search_config.cwt_method,
                    backend=search_config.cwt_backend,
                    cuda_device=search_config.cuda_device,
                    normalize_channels=False,
                )
                detector = detect_block_periods
            candidates, windows = detector(
                power_cube=power,
                raw_data=raw_device,
                periods=periods,
                freqs_mhz=block_freqs,
                noise_gain=noise_gain,
                record_start=0,
                target_channel_start=block_start - halo.start,
                target_channel_stop=block_stop - halo.start,
                candidate_period_min_records=search_config.candidate_period_min_records,
                candidate_period_max_records=search_config.candidate_period_max_records,
                cpro_threshold_snr=search_config.cpro_threshold_snr,
                cpro_texture_quantile=search_config.cpro_texture_quantile,
                cpro_period_center_bins=search_config.cpro_period_center_bins,
                cpro_period_context_bins=search_config.cpro_period_context_bins,
                cpro_min_period_contrast=search_config.cpro_min_period_contrast,
                cpro_period_support_bins=search_config.cpro_period_support_bins,
                cpro_shape_power_softness=search_config.cpro_shape_power_softness,
                cpro_shape_contrast_softness=search_config.cpro_shape_contrast_softness,
                cpro_continuity_decay=search_config.cpro_continuity_decay,
                cpro_continuity_power=search_config.cpro_continuity_power,
                cpro_min_continuity_mean=search_config.cpro_min_continuity_mean,
                cpro_min_ridge_lock=search_config.cpro_min_ridge_lock,
                pelt_penalty=search_config.pelt_penalty,
                pelt_min_size_records=search_config.pelt_min_size_records,
                pelt_jump_records=search_config.pelt_jump_records,
                pelt_threads=search_config.pelt_threads,
                window_min_activity_mean=search_config.window_min_activity_mean,
                window_merge_gap_records=search_config.window_merge_gap_records,
                cuda_device=search_config.cuda_device,
                cprf_params=cprf_params,
                max_candidates_per_channel=search_config.max_candidates_per_channel,
                max_candidates_per_record=search_config.max_candidates_per_record,
            )
            del power, raw_device
            for row in windows:
                row["channel"] = block_start + int(row["channel"])
                row["schema_version"] = RAW_CANDIDATE_SCHEMA_VERSION
                row["run_id"] = run_id
                row["source_file"] = source_name
                row["block_id"] = f"block_{block_index:04d}"
                row["block_ch0"] = block_start
                row["block_ch1"] = block_stop
                all_windows.append(row)
            for row in candidates:
                row["channel"] = block_start + int(row["channel"])
                row["wavelet"] = search_config.wavelet
                row["time_agg"] = search_config.time_aggregation
                row["block_ch0"] = block_start
                row["block_ch1"] = block_stop
                all_candidates.append(
                    normalize_candidate_row(
                        row,
                        run_id=run_id,
                        source_file=source_name,
                        block_id=f"block_{block_index:04d}",
                        tsamp_seconds=tsamp_seconds,
                    )
                )
            if progress is not None:
                progress.update(int(block_stop - block_start))
    finally:
        if progress is not None:
            progress.close()
    raw = add_candidate_ids(all_candidates)
    context = VetoContext(
        record_start=0,
        record_stop=matrix.shape[0],
        freq_start_mhz=float(np.nanmin(freqs_mhz)),
        freq_stop_mhz=float(np.nanmax(freqs_mhz)),
    )
    return raw, review_candidates(raw, context=context, config=veto_config), all_windows


def _freq_slice_for_row(freqs_mhz: np.ndarray, row: dict[str, Any]) -> slice:
    freq = _float(row, "freq_mhz", np.nan)
    if not np.isfinite(freq):
        raise ValueError("candidate freq_mhz must be finite")
    idx = int(np.nanargmin(np.abs(freqs_mhz - freq)))
    return slice(idx, idx + 1)


def _record_window_for_row(row: dict[str, Any], records: int, config: ValidationConfig) -> slice:
    peak = int(_float(row, "t_peak_rec", _float(row, "t0_rec", 0)))
    approx = max(float(config.min_period_records), _float(row, "period_rec", 2.0))
    target = int(np.ceil(approx * max(1, config.window_periods)))
    target = max(int(config.min_window_records), min(int(config.max_window_records), target, int(records)))
    start = max(0, peak - target // 2)
    stop = min(records, start + target)
    start = max(0, stop - target)
    return slice(start, max(start + 1, stop))


def validate_cwt_candidates(
    data: np.ndarray,
    freqs_mhz: np.ndarray,
    reviewed_candidates: list[dict[str, Any]],
    validation_config: ValidationConfig,
    tsamp_seconds: float,
) -> list[dict[str, Any]]:
    rows = [
        row for row in reviewed_candidates
        if validation_config.include_vetoed or row.get("candidate_status") != "vetoed"
    ][: max(0, validation_config.max_candidates)]
    results: list[dict[str, Any]] = []
    for row in rows:
        candidate_id = int(_float(row, "candidate_id", len(results) + 1))
        rng = np.random.default_rng(int(validation_config.random_seed) + candidate_id)
        record_slice = _record_window_for_row(row, data.shape[0], validation_config)
        freq_slice = _freq_slice_for_row(freqs_mhz, row)
        series = aggregate_frequency_series(data[record_slice, freq_slice])
        approx_period = max(float(validation_config.min_period_records), _float(row, "period_rec", 2.0))
        min_period, max_period = validation_period_bounds(approx_period, series.size, validation_config)
        periods = period_grid(min_period, max_period)
        if series.size < max(8, 3 * min_period) or periods.size == 0:
            results.append(
                {
                    "schema_version": 1,
                    "run_id": row.get("run_id", ""),
                    "source_file": row.get("source_file", ""),
                    "candidate_id": row.get("candidate_id", ""),
                    "candidate_status": row.get("candidate_status", ""),
                    "validation_status": "insufficient_data",
                    "validation_notes": "validation window is too short for the requested period search",
                    "approx_period_records": approx_period,
                    "period_min_records": min_period,
                    "period_max_records": max_period,
                }
            )
            continue
        acf_metrics = best_acf_peak(series, min_period, max_period)
        periodogram_metrics = fft_periodogram_peak(series, min_period, max_period)
        fold_metrics = best_fold_period(series, periods, validation_config.fold_bins)
        null_metrics = shuffle_null_pvalue(
            series,
            periods,
            validation_config.fold_bins,
            validation_config.shuffle_trials,
            rng,
        )
        refined_period = refined_period_from_metrics(approx_period, acf_metrics, periodogram_metrics, fold_metrics)
        results.append(
            {
                "schema_version": 1,
                "run_id": row.get("run_id", ""),
                "source_file": row.get("source_file", ""),
                "candidate_id": row.get("candidate_id", ""),
                "candidate_status": row.get("candidate_status", ""),
                "validation_status": "evaluated",
                "validation_notes": "injection benchmark evidence; not a signal claim",
                "validation_record_start": int(record_slice.start),
                "validation_record_stop": int(record_slice.stop),
                "validation_duration_records": int(record_slice.stop - record_slice.start),
                "validation_freq_start_mhz": float(freqs_mhz[freq_slice.start]),
                "validation_freq_stop_mhz": float(freqs_mhz[freq_slice.stop - 1]),
                "validation_channel_count": int(freq_slice.stop - freq_slice.start),
                "approx_period_records": approx_period,
                "period_min_records": min_period,
                "period_max_records": max_period,
                "refined_period_records": refined_period,
                "refined_period_seconds": refined_period * float(tsamp_seconds) if math.isfinite(refined_period) else "",
                **acf_metrics,
                **periodogram_metrics,
                **fold_metrics,
                **null_metrics,
            }
        )
    return results


def _candidate_overlap(candidate: dict[str, Any], truth: dict[str, Any]) -> tuple[float, float]:
    time_overlap = _span_overlap(
        _float(candidate, "t0_rec"),
        _float(candidate, "t1_rec"),
        _float(truth, "record_start"),
        _float(truth, "record_stop"),
    )
    candidate_freq = _float(candidate, "freq_mhz")
    freq_overlap = _span_overlap(
        candidate_freq,
        candidate_freq,
        _float(truth, "freq_start_mhz"),
        _float(truth, "freq_stop_mhz"),
    )
    return time_overlap, freq_overlap


def best_truth_match(candidates: list[dict[str, Any]], truth: dict[str, Any], match_config: MatchConfig) -> dict[str, Any] | None:
    matches: list[tuple[float, dict[str, Any], float, float]] = []
    for candidate in candidates:
        time_overlap, freq_overlap = _candidate_overlap(candidate, truth)
        if time_overlap >= match_config.min_time_overlap and freq_overlap >= match_config.min_freq_overlap:
            candidate_score = _float(candidate, "score", 0.0)
            score = time_overlap + freq_overlap + candidate_score * 0.01
            matches.append((score, candidate, time_overlap, freq_overlap))
    if not matches:
        return None
    _score, candidate, time_overlap, freq_overlap = max(matches, key=lambda item: item[0])
    matched = dict(candidate)
    matched["_time_overlap_fraction"] = time_overlap
    matched["_freq_overlap_fraction"] = freq_overlap
    return matched


def evaluate_injections(
    truths: list[dict[str, Any]],
    raw_candidates: list[dict[str, Any]],
    reviewed_candidates: list[dict[str, Any]],
    validation_reviewed: list[dict[str, Any]],
    match_config: MatchConfig,
) -> list[dict[str, Any]]:
    validation_by_id = {str(row.get("candidate_id")): row for row in validation_reviewed}
    results: list[dict[str, Any]] = []
    after_veto_candidates = [
        row for row in reviewed_candidates
        if row.get("candidate_status") != "vetoed"
    ]
    for truth in truths:
        raw_match = best_truth_match(raw_candidates, truth, match_config)
        reviewed_match = best_truth_match(after_veto_candidates, truth, match_config)
        detected_raw = raw_match is not None
        detected_after_veto = reviewed_match is not None
        validation_row = validation_by_id.get(str(reviewed_match.get("candidate_id"))) if reviewed_match else None
        refined_period = _float(validation_row or {}, "refined_period_records")
        period = _float(truth, "period_records")
        period_error = abs(refined_period - period) / max(period, 1e-12) if math.isfinite(refined_period) else math.nan
        validated = bool(
            detected_after_veto
            and validation_row
            and validation_row.get("validation_status") == "evaluated"
            and math.isfinite(period_error)
            and period_error <= match_config.max_period_error_fraction
        )
        if not detected_raw:
            failure_stage = "missed_detection"
        elif not detected_after_veto:
            failure_stage = "vetoed"
        elif validation_row is None:
            failure_stage = "not_validated"
        elif not validated:
            failure_stage = "period_mismatch"
        else:
            failure_stage = "validated"
        match = reviewed_match or raw_match or {}
        results.append(
            {
                "injection_id": truth.get("injection_id", ""),
                "signal_model": truth.get("signal_model", ""),
                "period_records": truth.get("period_records", ""),
                "amplitude": truth.get("amplitude", ""),
                "detected_raw": detected_raw,
                "detected_after_veto": detected_after_veto,
                "validated": validated,
                "matched_candidate_id": match.get("candidate_id", ""),
                "failure_stage": failure_stage,
                "time_overlap_fraction": match.get("_time_overlap_fraction", ""),
                "freq_overlap_fraction": match.get("_freq_overlap_fraction", ""),
                "period_error_fraction": period_error if math.isfinite(period_error) else "",
                "score": match.get("score", ""),
                "candidate_status": match.get("candidate_status", ""),
                "veto_flags": match.get("veto_flags", ""),
                "p_value": (validation_row or {}).get("p_value", ""),
                "q_value": (validation_row or {}).get("q_value", ""),
                "global_q_value": (validation_row or {}).get("global_q_value", ""),
                "evidence_rank": (validation_row or {}).get("evidence_rank", ""),
                "refined_period_records": (validation_row or {}).get("refined_period_records", ""),
            }
        )
    return results


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def aggregate_injection_performance(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in results:
        key = (
            str(row.get("signal_model", "")),
            str(row.get("period_records", "")),
            str(row.get("amplitude", "")),
        )
        groups.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    for (signal_model, period_records, amplitude), group in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            _float({"value": item[0][1]}, "value", math.inf),
            _float({"value": item[0][2]}, "value", math.inf),
        ),
    ):
        count = max(1, len(group))
        detected_raw = sum(1 for row in group if _bool_value(row.get("detected_raw")))
        detected_after_veto = sum(1 for row in group if _bool_value(row.get("detected_after_veto")))
        validated = sum(1 for row in group if _bool_value(row.get("validated")))
        failure_counts: dict[str, int] = {}
        for row in group:
            stage = str(row.get("failure_stage", ""))
            failure_counts[stage] = failure_counts.get(stage, 0) + 1
        rows.append(
            {
                "signal_model": signal_model,
                "period_records": period_records,
                "amplitude": amplitude,
                "injection_count": len(group),
                "detected_raw_count": detected_raw,
                "detected_after_veto_count": detected_after_veto,
                "validated_count": validated,
                "detected_raw_rate": detected_raw / count,
                "detected_after_veto_rate": detected_after_veto / count,
                "validated_rate": validated / count,
                "failure_stage_counts_json": json.dumps(failure_counts, sort_keys=True, ensure_ascii=True),
            }
        )
    return rows


def run_injection_benchmark(
    background: BackgroundData,
    injections: list[InjectionSpec],
    output_dir: str | Path,
    run_id: str,
    search_config: CWTBenchmarkConfig,
    veto_config: VetoConfig,
    validation_config: ValidationConfig,
    match_config: MatchConfig,
    visualization_config: Any | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    injected_background, truths = inject_many(background, injections)
    raw, reviewed, time_windows = run_cwt_candidate_search(
        injected_background.data,
        injected_background.freqs_mhz,
        source_name=injected_background.source_name,
        tsamp_seconds=injected_background.tsamp_seconds,
        run_id=run_id,
        search_config=search_config,
        veto_config=veto_config,
    )
    validation_rows = validate_cwt_candidates(
        injected_background.data,
        injected_background.freqs_mhz,
        reviewed,
        validation_config,
        tsamp_seconds=injected_background.tsamp_seconds,
    )
    validation_reviewed = review_validation_rows(validation_rows)
    results = evaluate_injections(truths, raw, reviewed, validation_reviewed, match_config)
    performance_rows = aggregate_injection_performance(results)
    write_rows_csv(output_dir / "injection_truth.csv", truths, INJECTION_TRUTH_FIELDNAMES)
    write_rows_csv(output_dir / "time_windows.csv", time_windows, TIME_WINDOW_FIELDNAMES)
    write_rows_csv(output_dir / "candidates_raw.csv", raw, RAW_CANDIDATE_FIELDNAMES)
    write_rows_csv(output_dir / "candidates_reviewed.csv", reviewed, REVIEWED_CANDIDATE_FIELDNAMES)
    write_rows_csv(output_dir / "validation_summary.csv", validation_rows, VALIDATION_FIELDNAMES)
    write_rows_csv(output_dir / "validation_reviewed.csv", validation_reviewed, VALIDATION_REVIEWED_FIELDNAMES)
    write_rows_csv(output_dir / "injection_results.csv", results, INJECTION_RESULT_FIELDNAMES)
    write_rows_csv(output_dir / "injection_performance.csv", performance_rows, INJECTION_PERFORMANCE_FIELDNAMES)
    failure_stage_counts: dict[str, int] = {}
    for row in results:
        stage = str(row.get("failure_stage", ""))
        failure_stage_counts[stage] = failure_stage_counts.get(stage, 0) + 1
    summary = {
        "schema_version": RAW_CANDIDATE_SCHEMA_VERSION,
        "run_id": run_id,
        "source": injected_background.source_name,
        "injection_count": len(truths),
        "candidate_count": len(raw),
        "reviewed_candidate_count": len(reviewed),
        "vetoed_candidate_count": sum(1 for row in reviewed if row.get("candidate_status") == "vetoed"),
        "time_window_count": len(time_windows),
        "validation_count": len(validation_rows),
        "detected_raw_count": sum(1 for row in results if row["detected_raw"]),
        "detected_after_veto_count": sum(1 for row in results if row["detected_after_veto"]),
        "validated_count": sum(1 for row in results if row["validated"]),
        "detected_raw_rate": (
            sum(1 for row in results if row["detected_raw"]) / len(results) if results else 0.0
        ),
        "detected_after_veto_rate": (
            sum(1 for row in results if row["detected_after_veto"]) / len(results) if results else 0.0
        ),
        "validated_rate": (
            sum(1 for row in results if row["validated"]) / len(results) if results else 0.0
        ),
        "failure_stage_counts": failure_stage_counts,
        "runtime": runtime_info(),
        "search_config": asdict(search_config),
        "veto_config": asdict(veto_config),
        "validation_config": asdict(validation_config),
        "match_config": asdict(match_config),
    }
    (output_dir / "injection_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True))
    if visualization_config is not None and getattr(visualization_config, "enabled", False):
        from ..reporting.visualization import (
            CWTVisualizationConfig,
            SearchVisualizationConfig,
            visualize_cwt_stages,
        )

        periods = period_grid_records(
            search_config.period_min_records,
            search_config.period_max_records,
            search_config.period_count,
            search_config.period_spacing,
        )

        visualize_cwt_stages(
            injected_background.data,
            injected_background.freqs_mhz,
            output_dir / "visualization",
            SearchVisualizationConfig(
                wavelet=search_config.wavelet,
                cwt_method=search_config.cwt_method,
                cwt_backend=search_config.cwt_backend,
                cuda_device=search_config.cuda_device,
                periods=periods,
                block_channels=search_config.block_channels,
                candidate_period_min_records=search_config.candidate_period_min_records,
                candidate_period_max_records=search_config.candidate_period_max_records,
                time_aggregation=search_config.time_aggregation,
                aggregation_percentile=search_config.aggregation_percentile,
                cpro_threshold_snr=search_config.cpro_threshold_snr,
                cpro_texture_quantile=search_config.cpro_texture_quantile,
                cpro_period_center_bins=search_config.cpro_period_center_bins,
                cpro_period_context_bins=search_config.cpro_period_context_bins,
                cpro_min_period_contrast=search_config.cpro_min_period_contrast,
                cpro_period_support_bins=search_config.cpro_period_support_bins,
                cpro_shape_power_softness=search_config.cpro_shape_power_softness,
                cpro_shape_contrast_softness=search_config.cpro_shape_contrast_softness,
                cpro_continuity_decay=search_config.cpro_continuity_decay,
                cpro_continuity_power=search_config.cpro_continuity_power,
                cpro_min_continuity_mean=search_config.cpro_min_continuity_mean,
                cpro_min_ridge_lock=search_config.cpro_min_ridge_lock,
                cprf_threshold_snr=search_config.cprf_threshold_snr,
                cprf_texture_quantile=search_config.cprf_texture_quantile,
                cprf_smooth_bins=search_config.cprf_smooth_bins,
                cprf_peak_band_fraction=search_config.cprf_peak_band_fraction,
                cprf_min_width_bins=search_config.cprf_min_width_bins,
                cprf_min_peak_strength=search_config.cprf_min_peak_strength,
                cprf_min_integrated_strength=search_config.cprf_min_integrated_strength,
                cprf_min_band_persistence=search_config.cprf_min_band_persistence,
                cprf_min_band_concentration=search_config.cprf_min_band_concentration,
                cprf_min_local_contrast=search_config.cprf_min_local_contrast,
                cprf_harmonic_weight=search_config.cprf_harmonic_weight,
                cprf_harmonic_min_relative=search_config.cprf_harmonic_min_relative,
                cprf_harmonic_window_scale=search_config.cprf_harmonic_window_scale,
                cprf_max_peak_hypotheses=search_config.cprf_max_peak_hypotheses,
            ),
            raw_candidates=raw,
            reviewed_candidates=reviewed,
            time_windows=time_windows,
            truths=truths,
            validation_rows=validation_reviewed,
            injection_results=results,
            run_id=run_id,
            source_name=f"{injected_background.source_name} + {len(truths)} injections",
            record_offset=0,
            config=visualization_config if isinstance(visualization_config, CWTVisualizationConfig) else CWTVisualizationConfig(
                enabled=True,
                max_blocks=getattr(visualization_config, "max_blocks", 2),
                max_channels=getattr(visualization_config, "max_channels", 4),
                top_candidates=getattr(visualization_config, "top_candidates", 50),
                dpi=getattr(visualization_config, "dpi", 140),
            ),
        )
    return output_dir

def make_background_from_args(
    mode: str,
    input_path: str | None,
    records: int,
    channels: int,
    seed: int,
    noise_std: float,
    f_start: float | None,
    f_stop: float | None,
    t_start: int | None,
    t_stop: int | None,
) -> BackgroundData:
    if mode == "synthetic":
        return synthetic_background(
            records=records,
            channels=channels,
            noise_std=noise_std,
            seed=seed,
            f_start_mhz=0.0 if f_start is None else f_start,
            f_stop_mhz=None if f_stop is None else f_stop,
        )
    if mode == "ce4":
        if not input_path:
            raise ValueError("--input is required for CE4-format background mode")
        return ce4_background(input_path, f_start=f_start, f_stop=f_stop, t_start=t_start, t_stop=t_stop)
    raise ValueError(f"Unknown benchmark background mode: {mode}")
