from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import ndimage as ndi


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(matrix)
    if not np.any(finite):
        return np.zeros_like(matrix, dtype=np.float32)
    median = float(np.nanmedian(matrix[finite]))
    centered = matrix - median
    mad = float(np.nanmedian(np.abs(centered[finite])))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = float(np.nanstd(centered[finite]))
    if not np.isfinite(scale) or scale <= 1e-6:
        return np.zeros_like(matrix, dtype=np.float32)
    score = centered / scale
    score[~finite] = 0.0
    return score.astype(np.float32, copy=False)


def scalogram_region_score(
    log_power: np.ndarray,
    sigma_period_peak: float = 1.0,
    sigma_period_background: float = 10.0,
    sigma_time: float = 1.0,
) -> np.ndarray:
    """Score one channel's `(period, time)` CWT scalogram for short period bands."""
    values = np.asarray(log_power, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("log_power must have shape (periods, records)")
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float32)
    filled = values.copy()
    filled[~finite] = float(np.nanmedian(filled[finite]))

    sigma_period_peak = max(0.0, float(sigma_period_peak))
    sigma_period_background = max(sigma_period_peak + 1e-6, float(sigma_period_background))
    sigma_time = max(0.0, float(sigma_time))
    narrow = (
        ndi.gaussian_filter1d(filled, sigma=sigma_period_peak, axis=0, mode="nearest")
        if sigma_period_peak > 0
        else filled
    )
    broad = ndi.gaussian_filter1d(filled, sigma=sigma_period_background, axis=0, mode="nearest")
    bandpass = narrow - broad
    if sigma_time > 0:
        bandpass = ndi.gaussian_filter1d(bandpass, sigma=sigma_time, axis=1, mode="nearest")
    return _robust_zscore(bandpass)


def _period_width(periods: np.ndarray, p0: int, p1: int) -> float:
    return float(abs(float(periods[p1]) - float(periods[p0])))


def _period_domain_mask(
    periods: np.ndarray,
    min_period_records: float | None,
    max_period_records: float | None,
) -> np.ndarray:
    values = np.asarray(periods, dtype=np.float64)
    lo = -np.inf if min_period_records is None else float(min_period_records)
    hi = np.inf if max_period_records is None else float(max_period_records)
    if hi < lo:
        lo, hi = hi, lo
    return np.asarray((values >= lo) & (values <= hi), dtype=bool)


def _region_time_integral(score: np.ndarray, p0: int, p1: int, t0: int, t1: int) -> tuple[float, float]:
    window = np.asarray(score[p0:p1 + 1, t0:t1 + 1], dtype=np.float32)
    if window.size == 0:
        return 0.0, 0.0
    time_profile = np.nanmax(window, axis=0)
    positive = np.clip(time_profile, 0.0, None)
    duration = max(1, int(time_profile.size))
    integrated = float(np.nansum(positive) / np.sqrt(duration))
    mean_score = float(np.nanmean(window))
    return integrated, mean_score


def _region_rows_for_channel(
    score: np.ndarray,
    power: np.ndarray,
    periods: np.ndarray,
    freq_mhz: float,
    channel_idx: int,
    record_start: int,
    threshold: float,
    min_duration_records: int,
    min_width_bins: float,
    max_width_bins: float,
    max_candidates_per_channel: int,
    candidate_period_min_records: float | None,
    candidate_period_max_records: float | None,
) -> list[dict]:
    mask = np.asarray(score >= float(threshold), dtype=bool)
    period_mask = _period_domain_mask(periods, candidate_period_min_records, candidate_period_max_records)
    mask &= period_mask[:, np.newaxis]
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    slices = ndi.find_objects(labels)
    rows: list[dict] = []
    width_min = max(1.0, float(min_width_bins))
    width_max = max(width_min, float(max_width_bins))
    min_duration = max(1, int(min_duration_records))

    for label_id, region_slice in enumerate(slices, start=1):
        if region_slice is None:
            continue
        local_p, local_t = np.nonzero(labels[region_slice] == label_id)
        if local_p.size == 0:
            continue
        p_offset = int(region_slice[0].start or 0)
        t_offset = int(region_slice[1].start or 0)
        p_idx = local_p + p_offset
        t_idx = local_t + t_offset
        p0, p1 = int(p_idx.min()), int(p_idx.max())
        t0, t1 = int(t_idx.min()), int(t_idx.max())
        period_width_bins = float(p1 - p0 + 1)
        duration = int(t1 - t0 + 1)
        if duration < min_duration or period_width_bins < width_min or period_width_bins > width_max:
            continue

        region_scores = score[p_idx, t_idx]
        peak_local = int(np.nanargmax(region_scores))
        peak_period_idx = int(p_idx[peak_local])
        peak_time_idx = int(t_idx[peak_local])
        integrated_score, mean_score = _region_time_integral(score, p0, p1, t0, t1)
        rows.append(
            {
                "detection_method": "per_channel_scalogram_region",
                "channel_index": int(channel_idx),
                "region_pixels": int(local_p.size),
                "record_start": int(record_start + t0),
                "record_stop": int(record_start + t1 + 1),
                "duration_records": duration,
                "period_start_records": float(periods[p0]),
                "period_stop_records": float(periods[p1]),
                "period_width_records": _period_width(periods, p0, p1),
                "period_width_bins": period_width_bins,
                "peak_period_records": float(periods[peak_period_idx]),
                "freq_start_mhz": float(freq_mhz),
                "freq_stop_mhz": float(freq_mhz),
                "bandwidth_mhz": 0.0,
                "peak_record": int(record_start + peak_time_idx),
                "peak_freq_mhz": float(freq_mhz),
                "peak_score": float(region_scores[peak_local]),
                "mean_score": mean_score,
                "integrated_score": integrated_score,
            }
        )

    rows.sort(key=lambda row: (row["integrated_score"], row["peak_score"]), reverse=True)
    return rows[: max(1, int(max_candidates_per_channel))]


def summarize_scalogram_regions(
    power_cube: np.ndarray,
    periods: np.ndarray,
    freqs_mhz: np.ndarray,
    record_start: int,
    threshold: float,
    sigma_period_peak: float,
    sigma_period_background: float,
    sigma_time: float,
    min_duration_records: int,
    min_width_bins: float,
    max_width_bins: float,
    max_candidates_per_channel: int,
    candidate_period_min_records: float | None = None,
    candidate_period_max_records: float | None = None,
    max_candidates: int | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Detect candidate regions in each channel's first-hand `(period, time)` scalogram."""
    power = np.asarray(power_cube, dtype=np.float32)
    period_values = np.asarray(periods, dtype=np.float64)
    freqs = np.asarray(freqs_mhz, dtype=np.float64)
    if power.ndim != 3:
        raise ValueError("power_cube must have shape (periods, records, channels)")
    if power.shape[0] != period_values.size or power.shape[2] != freqs.size:
        raise ValueError("power_cube shape must match periods and freqs_mhz")

    score_cube = np.zeros_like(power, dtype=np.float32)
    rows: list[dict] = []
    for channel_idx in range(power.shape[2]):
        score = scalogram_region_score(
            np.log10(power[:, :, channel_idx] + 1e-12),
            sigma_period_peak=sigma_period_peak,
            sigma_period_background=sigma_period_background,
            sigma_time=sigma_time,
        )
        score_cube[:, :, channel_idx] = score
        rows.extend(
            _region_rows_for_channel(
                score,
                power[:, :, channel_idx],
                period_values,
                float(freqs[channel_idx]),
                channel_idx,
                record_start,
                threshold,
                min_duration_records,
                min_width_bins,
                max_width_bins,
                max_candidates_per_channel,
                candidate_period_min_records,
                candidate_period_max_records,
            )
        )

    rows.sort(key=lambda row: (row["integrated_score"], row["peak_score"]), reverse=True)
    if max_candidates is not None:
        rows = rows[: max(0, int(max_candidates))]
    return rows, score_cube


def project_scalogram_score(score_cube: np.ndarray, method: str = "max") -> np.ndarray:
    """Project `(period, time, channel)` scalogram scores to `(period, channel)` for overview plots."""
    scores = np.asarray(score_cube, dtype=np.float32)
    if scores.ndim != 3:
        raise ValueError("score_cube must have shape (periods, records, channels)")
    if method == "mean":
        return np.nanmean(scores, axis=1).astype(np.float32)
    if method == "p95":
        return np.nanpercentile(scores, 95.0, axis=1).astype(np.float32)
    return np.nanmax(scores, axis=1).astype(np.float32)


def add_candidate_ids(candidates: Iterable[dict]) -> list[dict]:
    rows = sorted(
        candidates,
        key=lambda row: float(row.get("integrated_score", row.get("peak_score", 0.0)) or 0.0),
        reverse=True,
    )
    for idx, row in enumerate(rows, start=1):
        row["candidate_id"] = idx
    return rows
