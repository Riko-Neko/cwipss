from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import CE4Reader
from .models import VALIDATION_FIELDNAMES
from .swt import robust_zscore


VALIDATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ValidationConfig:
    include_vetoed: bool = False
    max_candidates: int = 50
    window_periods: int = 128
    min_window_records: int = 256
    max_window_records: int = 4096
    period_search_radius: float = 2.0
    min_period_records: int = 2
    max_period_records: int = 2048
    fold_bins: int = 16
    shuffle_trials: int = 100
    random_seed: int = 12345


def validation_config_from_scan_config(config: object) -> ValidationConfig:
    return ValidationConfig(
        include_vetoed=bool(getattr(config, "validation_include_vetoed")),
        max_candidates=int(getattr(config, "validation_max_candidates")),
        window_periods=int(getattr(config, "validation_window_periods")),
        min_window_records=int(getattr(config, "validation_min_window_records")),
        max_window_records=int(getattr(config, "validation_max_window_records")),
        period_search_radius=float(getattr(config, "validation_period_search_radius")),
        min_period_records=int(getattr(config, "validation_min_period_records")),
        max_period_records=int(getattr(config, "validation_max_period_records")),
        fold_bins=int(getattr(config, "validation_fold_bins")),
        shuffle_trials=int(getattr(config, "validation_shuffle_trials")),
        random_seed=int(getattr(config, "validation_random_seed")),
    )


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv_rows(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    with Path(path).open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_validation_outputs(
    output_path: str | Path,
    detail_dir: str | Path,
    rows: list[dict[str, Any]],
    config: ValidationConfig,
) -> None:
    output_path = Path(output_path)
    detail_dir = Path(detail_dir)
    detail_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_path, rows, VALIDATION_FIELDNAMES)
    (detail_dir / "validation_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=True)
    )
    for row in rows:
        candidate_id = int(_float(row, "candidate_id", 0))
        detail_path = detail_dir / f"candidate_{candidate_id:06d}.json"
        detail_path.write_text(json.dumps(row, indent=2, ensure_ascii=True))


def select_candidates(rows: Iterable[Mapping[str, Any]], config: ValidationConfig) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    limit = max(0, int(config.max_candidates))
    if limit == 0:
        return selected
    for row in rows:
        status = str(row.get("candidate_status", "needs_validation"))
        if not config.include_vetoed and status == "vetoed":
            continue
        selected.append(dict(row))
        if len(selected) >= limit:
            break
    return selected


def autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    series = robust_zscore(values)
    series = np.nan_to_num(series, nan=0.0)
    if series.size == 0:
        return np.zeros(0, dtype=np.float64)
    max_lag = max(0, min(int(max_lag), series.size - 1))
    corr = np.correlate(series, series, mode="full")[series.size - 1 : series.size + max_lag]
    norm = corr[0] if corr[0] > 0 else 1.0
    return (corr / norm).astype(np.float64)


def best_acf_peak(values: np.ndarray, min_lag: int, max_lag: int) -> dict[str, float]:
    acf = autocorrelation(values, max_lag)
    if acf.size <= min_lag:
        return {"acf_best_lag_records": np.nan, "acf_peak": np.nan, "acf_prominence": np.nan}
    start = max(1, int(min_lag))
    stop = min(int(max_lag), acf.size - 1)
    if stop < start:
        return {"acf_best_lag_records": np.nan, "acf_peak": np.nan, "acf_prominence": np.nan}
    window = acf[start : stop + 1]
    local_idx = int(np.nanargmax(window))
    peak = float(window[local_idx])
    baseline = float(np.nanmedian(window))
    return {
        "acf_best_lag_records": float(start + local_idx),
        "acf_peak": peak,
        "acf_prominence": peak - baseline,
    }


def fft_periodogram_peak(values: np.ndarray, min_period: int, max_period: int) -> dict[str, float]:
    series = robust_zscore(values)
    series = np.nan_to_num(series, nan=0.0)
    if series.size < 4:
        return {"periodogram_best_period_records": np.nan, "periodogram_peak_power": np.nan}
    freqs = np.fft.rfftfreq(series.size, d=1.0)
    power = np.abs(np.fft.rfft(series)) ** 2 / float(series.size)
    valid = freqs > 0
    periods = np.full(freqs.shape, np.inf, dtype=np.float64)
    periods[valid] = 1.0 / freqs[valid]
    mask = (periods >= float(min_period)) & (periods <= float(max_period))
    if not np.any(mask):
        return {"periodogram_best_period_records": np.nan, "periodogram_peak_power": np.nan}
    masked_power = power[mask]
    masked_periods = periods[mask]
    idx = int(np.nanargmax(masked_power))
    baseline = float(np.nanmedian(masked_power))
    scale = baseline if baseline > 1e-12 else 1.0
    return {
        "periodogram_best_period_records": float(masked_periods[idx]),
        "periodogram_peak_power": float(masked_power[idx] / scale),
    }


def fold_profile_snr(values: np.ndarray, period_records: float, fold_bins: int = 16) -> dict[str, float]:
    series = robust_zscore(values)
    series = np.nan_to_num(series, nan=0.0)
    period = max(2.0, float(period_records))
    n_bins = max(2, min(int(fold_bins), int(np.floor(period))))
    phases = (np.arange(series.size, dtype=np.float64) / period) % 1.0
    indices = np.minimum((phases * n_bins).astype(np.int64), n_bins - 1)
    sums = np.bincount(indices, weights=series, minlength=n_bins).astype(np.float64)
    counts = np.bincount(indices, minlength=n_bins).astype(np.float64)
    profile = sums / np.where(counts > 0, counts, 1.0)
    valid = counts > 0
    if not np.any(valid):
        return {"fold_profile_snr": np.nan, "fold_bin_count": float(n_bins)}
    profile = profile[valid]
    median = float(np.nanmedian(profile))
    mad = float(np.nanmedian(np.abs(profile - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.nanstd(profile))
    if not np.isfinite(scale) or scale <= 1e-12:
        return {"fold_profile_snr": 0.0, "fold_bin_count": float(n_bins)}
    return {
        "fold_profile_snr": float((np.nanmax(profile) - median) / scale),
        "fold_bin_count": float(n_bins),
    }


def period_grid(min_period: int, max_period: int) -> np.ndarray:
    if max_period < min_period:
        return np.zeros(0, dtype=np.float64)
    return np.arange(int(min_period), int(max_period) + 1, dtype=np.float64)


def best_fold_period(values: np.ndarray, periods: np.ndarray, fold_bins: int) -> dict[str, float]:
    if periods.size == 0:
        return {
            "folding_best_period_records": np.nan,
            "fold_profile_snr": np.nan,
            "fold_bin_count": np.nan,
        }
    best_period = float(periods[0])
    best_score = -np.inf
    best_bins = np.nan
    for period in periods:
        metrics = fold_profile_snr(values, period, fold_bins)
        score = float(metrics["fold_profile_snr"])
        if np.isfinite(score) and score > best_score:
            best_period = float(period)
            best_score = score
            best_bins = float(metrics["fold_bin_count"])
    if not np.isfinite(best_score):
        best_score = np.nan
    return {
        "folding_best_period_records": best_period,
        "fold_profile_snr": float(best_score),
        "fold_bin_count": best_bins,
    }


def shuffle_null_pvalue(
    values: np.ndarray,
    periods: np.ndarray,
    fold_bins: int,
    shuffle_trials: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    observed = best_fold_period(values, periods, fold_bins)
    observed_metric = float(observed["fold_profile_snr"])
    if shuffle_trials <= 0 or not np.isfinite(observed_metric):
        return {
            "observed_metric": observed_metric,
            "null_max_metric": np.nan,
            "shuffle_trials": float(max(0, shuffle_trials)),
            "shuffle_pvalue": np.nan,
        }
    null_metrics: list[float] = []
    count_ge = 0
    for _ in range(int(shuffle_trials)):
        shuffled = rng.permutation(values)
        metric = float(best_fold_period(shuffled, periods, fold_bins)["fold_profile_snr"])
        if np.isfinite(metric):
            null_metrics.append(metric)
            if metric >= observed_metric:
                count_ge += 1
    pvalue = (count_ge + 1.0) / (len(null_metrics) + 1.0)
    return {
        "observed_metric": observed_metric,
        "null_max_metric": float(np.nanmax(null_metrics)) if null_metrics else np.nan,
        "shuffle_trials": float(len(null_metrics)),
        "shuffle_pvalue": float(pvalue),
    }


def validation_period_bounds(
    approx_period_records: float,
    series_size: int,
    config: ValidationConfig,
) -> tuple[int, int]:
    approx = max(float(config.min_period_records), float(approx_period_records))
    radius = max(1.0, float(config.period_search_radius))
    min_period = max(int(config.min_period_records), int(np.floor(approx / radius)))
    max_period = min(
        int(config.max_period_records),
        max(min_period, int(np.ceil(approx * radius))),
        max(min_period, series_size // 2),
    )
    return min_period, max_period


def candidate_record_window(
    row: Mapping[str, Any],
    reader: CE4Reader,
    config: ValidationConfig,
) -> slice:
    peak = int(_float(row, "peak_record", _float(row, "record_start", 0)))
    approx_period = max(float(config.min_period_records), _float(row, "approx_scale_records", 2.0))
    target = int(np.ceil(approx_period * max(1, config.window_periods)))
    target = max(int(config.min_window_records), min(int(config.max_window_records), target))
    target = min(target, reader.n_records)
    start = peak - target // 2
    stop = start + target
    if start < 0:
        stop -= start
        start = 0
    if stop > reader.n_records:
        start -= stop - reader.n_records
        stop = reader.n_records
    start = max(0, start)
    stop = max(start + 1, stop)
    return slice(int(start), int(stop))


def candidate_freq_slice(row: Mapping[str, Any], reader: CE4Reader) -> slice:
    freqs = reader.freqs_mhz
    lo = _float(row, "freq_start_mhz", _float(row, "peak_freq_mhz", 0.0))
    hi = _float(row, "freq_stop_mhz", lo)
    lo, hi = sorted([lo, hi])
    if hi == lo:
        idx = int(np.nanargmin(np.abs(freqs - lo)))
        return slice(idx, idx + 1)
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        peak = _float(row, "peak_freq_mhz", 0.5 * (lo + hi))
        idx = int(np.nanargmin(np.abs(freqs - peak)))
        return slice(idx, idx + 1)
    indices = np.where(mask)[0]
    return slice(int(indices[0]), int(indices[-1]) + 1)


def aggregate_frequency_series(data: np.ndarray) -> np.ndarray:
    matrix = np.asarray(data, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("data must have shape (records, channels)")
    if matrix.shape[1] == 1:
        return robust_zscore(matrix[:, 0]).astype(np.float64)
    normalized = np.empty(matrix.shape, dtype=np.float32)
    for channel_idx in range(matrix.shape[1]):
        normalized[:, channel_idx] = robust_zscore(matrix[:, channel_idx])
    return np.nanmean(normalized, axis=1).astype(np.float64)


def extract_candidate_series(
    reader: CE4Reader,
    row: Mapping[str, Any],
    config: ValidationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    record_slice = candidate_record_window(row, reader, config)
    freq_slice = candidate_freq_slice(row, reader)
    block = reader.read_block(record_slice, freq_slice)
    series = aggregate_frequency_series(block.data)
    meta = {
        "validation_record_start": int(block.record_range[0]),
        "validation_record_stop": int(block.record_range[1]),
        "validation_duration_records": int(block.record_range[1] - block.record_range[0]),
        "validation_freq_start_mhz": float(np.nanmin(block.freqs_mhz)),
        "validation_freq_stop_mhz": float(np.nanmax(block.freqs_mhz)),
        "validation_channel_count": int(block.freqs_mhz.size),
    }
    return series, meta


def validate_candidate(
    reader: CE4Reader,
    row: Mapping[str, Any],
    config: ValidationConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    base = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "run_id": row.get("run_id", ""),
        "source_file": row.get("source_file", str(reader.filename)),
        "candidate_id": row.get("candidate_id", ""),
        "candidate_status": row.get("candidate_status", ""),
    }
    try:
        series, meta = extract_candidate_series(reader, row, config)
        approx_period = max(float(config.min_period_records), _float(row, "approx_scale_records", 2.0))
        min_period, max_period = validation_period_bounds(approx_period, series.size, config)
        periods = period_grid(min_period, max_period)
        min_required = max(8, 3 * min_period)
        if series.size < min_required or periods.size == 0:
            return {
                **base,
                **meta,
                "validation_status": "insufficient_data",
                "validation_notes": "validation window is too short for the requested period search",
                "approx_period_records": approx_period,
                "period_min_records": min_period,
                "period_max_records": max_period,
            }

        acf_metrics = best_acf_peak(series, min_period, max_period)
        periodogram_metrics = fft_periodogram_peak(series, min_period, max_period)
        fold_metrics = best_fold_period(series, periods, config.fold_bins)
        null_metrics = shuffle_null_pvalue(
            series,
            periods,
            config.fold_bins,
            config.shuffle_trials,
            rng,
        )
        refined_period = float(fold_metrics["folding_best_period_records"])
        refined_seconds = refined_period * float(reader.tsamp_seconds) if np.isfinite(refined_period) else np.nan
        return {
            **base,
            **meta,
            "validation_status": "evaluated",
            "validation_notes": "candidate remains unclaimed; metrics are evidence for later review",
            "approx_period_records": approx_period,
            "period_min_records": min_period,
            "period_max_records": max_period,
            "refined_period_records": refined_period,
            "refined_period_seconds": refined_seconds,
            **acf_metrics,
            **periodogram_metrics,
            **fold_metrics,
            **null_metrics,
        }
    except Exception as exc:
        return {
            **base,
            "validation_status": "error",
            "validation_notes": str(exc),
        }


def _resolve_source_path(path_text: str, project_dir: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path(project_dir) / path


def validate_candidate_rows(
    rows: Iterable[Mapping[str, Any]],
    config: ValidationConfig,
    project_dir: str | Path,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    selected = select_candidates(rows, config)
    for row in selected:
        source_file = str(row.get("source_file", ""))
        grouped[source_file].append(row)

    results: list[dict[str, Any]] = []
    for source_file, group_rows in grouped.items():
        reader = CE4Reader(_resolve_source_path(source_file, project_dir))
        for row in group_rows:
            candidate_id = int(_float(row, "candidate_id", len(results) + 1))
            rng = np.random.default_rng(int(config.random_seed) + candidate_id)
            results.append(validate_candidate(reader, row, config, rng))
    return results


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)
