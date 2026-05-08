from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .detection import add_candidate_ids, robust_score_2d, summarize_components
from .injection import BackgroundData, ce4_background, inject_many, synthetic_background
from .models import (
    INJECTION_PERFORMANCE_FIELDNAMES,
    INJECTION_RESULT_FIELDNAMES,
    INJECTION_TRUTH_FIELDNAMES,
    RAW_CANDIDATE_FIELDNAMES,
    REVIEWED_CANDIDATE_FIELDNAMES,
    VALIDATION_FIELDNAMES,
    VALIDATION_REVIEWED_FIELDNAMES,
    normalize_candidate_row,
)
from .runtime import runtime_info
from .simulation import InjectionSpec
from .stats import review_validation_rows
from .swt import approximate_scale_records, swt_detail_power_matrix
from .validation import (
    ValidationConfig,
    aggregate_frequency_series,
    best_acf_peak,
    best_fold_period,
    fft_periodogram_peak,
    period_grid,
    shuffle_null_pvalue,
    validation_period_bounds,
)
from .veto import VetoConfig, VetoContext, review_candidates


@dataclass(frozen=True)
class MatrixSearchConfig:
    wavelet: str = "db4"
    levels: int = 5
    block_channels: int = 128
    threshold: float = 5.0
    min_pixels: int = 12
    local_time: int = 513
    local_freq: int = 9
    max_candidates_per_block: int = 200


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


def run_matrix_candidate_search(
    data: np.ndarray,
    freqs_mhz: np.ndarray,
    source_name: str,
    tsamp_seconds: float,
    run_id: str,
    search_config: MatrixSearchConfig,
    veto_config: VetoConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matrix = np.asarray(data, dtype=np.float32)
    all_candidates: list[dict[str, Any]] = []
    for block_index, block_start in enumerate(range(0, matrix.shape[1], search_config.block_channels), start=1):
        block_stop = min(block_start + int(search_config.block_channels), matrix.shape[1])
        block_data = matrix[:, block_start:block_stop]
        block_freqs = freqs_mhz[block_start:block_stop]
        powers, level_numbers = swt_detail_power_matrix(
            block_data,
            wavelet=search_config.wavelet,
            levels=search_config.levels,
            normalize_channels=True,
        )
        for level_idx, level_number in enumerate(level_numbers):
            log_power = np.log10(powers[level_idx] + 1e-12)
            score = robust_score_2d(
                log_power,
                local_time=search_config.local_time,
                local_freq=min(search_config.local_freq, max(3, block_freqs.size | 1)),
            )
            candidates = summarize_components(
                score=score,
                freqs_mhz=block_freqs,
                record_start=0,
                level_number=int(level_number),
                threshold=search_config.threshold,
                min_pixels=search_config.min_pixels,
                max_components=search_config.max_candidates_per_block,
            )
            for row in candidates:
                row["approx_scale_records"] = approximate_scale_records(int(level_number))
                row["block_channel_start"] = block_start
                row["block_channel_stop"] = block_stop
                all_candidates.append(
                    normalize_candidate_row(
                        row,
                        run_id=run_id,
                        source_file=source_name,
                        block_id=f"block_{block_index:04d}",
                        tsamp_seconds=tsamp_seconds,
                    )
                )
    raw = add_candidate_ids(all_candidates)
    context = VetoContext(
        record_start=0,
        record_stop=matrix.shape[0],
        freq_start_mhz=float(np.nanmin(freqs_mhz)),
        freq_stop_mhz=float(np.nanmax(freqs_mhz)),
    )
    return raw, review_candidates(raw, context=context, config=veto_config)


def _freq_slice_for_row(freqs_mhz: np.ndarray, row: dict[str, Any]) -> slice:
    lo = _float(row, "freq_start_mhz", _float(row, "peak_freq_mhz", 0.0))
    hi = _float(row, "freq_stop_mhz", lo)
    lo, hi = sorted([lo, hi])
    if hi == lo:
        idx = int(np.nanargmin(np.abs(freqs_mhz - lo)))
        return slice(idx, idx + 1)
    mask = (freqs_mhz >= lo) & (freqs_mhz <= hi)
    if not np.any(mask):
        idx = int(np.nanargmin(np.abs(freqs_mhz - _float(row, "peak_freq_mhz", 0.5 * (lo + hi)))))
        return slice(idx, idx + 1)
    indices = np.where(mask)[0]
    return slice(int(indices[0]), int(indices[-1]) + 1)


def _record_window_for_row(row: dict[str, Any], records: int, config: ValidationConfig) -> slice:
    peak = int(_float(row, "peak_record", _float(row, "record_start", 0)))
    approx = max(float(config.min_period_records), _float(row, "approx_scale_records", 2.0))
    target = int(np.ceil(approx * max(1, config.window_periods)))
    target = max(int(config.min_window_records), min(int(config.max_window_records), target, int(records)))
    start = max(0, peak - target // 2)
    stop = min(records, start + target)
    start = max(0, stop - target)
    return slice(start, max(start + 1, stop))


def validate_matrix_candidates(
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
        approx_period = max(float(validation_config.min_period_records), _float(row, "approx_scale_records", 2.0))
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
        refined_period = _float(fold_metrics, "folding_best_period_records")
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
        _float(candidate, "record_start"),
        _float(candidate, "record_stop"),
        _float(truth, "record_start"),
        _float(truth, "record_stop"),
    )
    freq_overlap = _span_overlap(
        _float(candidate, "freq_start_mhz"),
        _float(candidate, "freq_stop_mhz"),
        _float(truth, "freq_start_mhz"),
        _float(truth, "freq_stop_mhz"),
    )
    return time_overlap, freq_overlap


def best_truth_match(candidates: list[dict[str, Any]], truth: dict[str, Any], match_config: MatchConfig) -> dict[str, Any] | None:
    matches: list[tuple[float, dict[str, Any], float, float]] = []
    for candidate in candidates:
        time_overlap, freq_overlap = _candidate_overlap(candidate, truth)
        if time_overlap >= match_config.min_time_overlap and freq_overlap >= match_config.min_freq_overlap:
            score = time_overlap + freq_overlap + _float(candidate, "peak_score", 0.0) * 0.01
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
                "peak_score": match.get("peak_score", ""),
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
    search_config: MatrixSearchConfig,
    veto_config: VetoConfig,
    validation_config: ValidationConfig,
    match_config: MatchConfig,
    visualization_config: Any | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    injected_background, truths = inject_many(background, injections)
    raw, reviewed = run_matrix_candidate_search(
        injected_background.data,
        injected_background.freqs_mhz,
        source_name=injected_background.source_name,
        tsamp_seconds=injected_background.tsamp_seconds,
        run_id=run_id,
        search_config=search_config,
        veto_config=veto_config,
    )
    validation_rows = validate_matrix_candidates(
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
        "run_id": run_id,
        "source": injected_background.source_name,
        "injection_count": len(truths),
        "candidate_count": len(raw),
        "reviewed_candidate_count": len(reviewed),
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
        from .visualization import SearchVisualizationConfig, visualize_matrix_stages

        visualize_matrix_stages(
            injected_background.data,
            injected_background.freqs_mhz,
            output_dir / "visualization",
            SearchVisualizationConfig(
                wavelet=search_config.wavelet,
                levels=search_config.levels,
                block_channels=search_config.block_channels,
                threshold=search_config.threshold,
                local_time=search_config.local_time,
                local_freq=search_config.local_freq,
            ),
            raw_candidates=raw,
            reviewed_candidates=reviewed,
            truths=truths,
            validation_rows=validation_reviewed,
            injection_results=results,
            run_id=run_id,
            source_name=injected_background.source_name,
            record_offset=0,
            config=visualization_config,
        )
    return output_dir


def make_default_injections(
    periods: list[float],
    amplitudes: list[float],
    records: int,
    channels: int,
    model: str = "pulsed_periodic",
    grid: bool = False,
    repeats: int = 1,
) -> list[InjectionSpec]:
    specs: list[InjectionSpec] = []
    if not periods:
        periods = [16.0]
    if not amplitudes:
        amplitudes = [5.0]
    repeats = max(1, int(repeats))
    combos: list[tuple[float, float, int]] = []
    if grid:
        for period in periods:
            for amplitude in amplitudes:
                for repeat_idx in range(repeats):
                    combos.append((float(period), float(amplitude), repeat_idx))
    else:
        for idx, period in enumerate(periods):
            amplitude = amplitudes[min(idx, len(amplitudes) - 1)]
            for repeat_idx in range(repeats):
                combos.append((float(period), float(amplitude), repeat_idx))

    for idx, (period, amplitude, repeat_idx) in enumerate(combos):
        specs.append(
            InjectionSpec(
                injection_id=f"inj_{idx + 1:04d}",
                signal_model=model,
                period_records=period,
                amplitude=amplitude,
                record_start=max(0, records // 4),
                duration_records=max(1, records // 2),
                channel_center=max(0.0, (channels - 1) * (idx + 1) / (len(combos) + 1)),
                bandwidth_channels=max(3.0, channels * 0.08),
                phase=(repeat_idx / repeats) % 1.0,
            )
        )
    return specs


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
            raise ValueError("--input is required for CE-4 background mode")
        return ce4_background(input_path, f_start=f_start, f_stop=f_stop, t_start=t_start, t_stop=t_stop)
    raise ValueError(f"Unknown benchmark background mode: {mode}")
