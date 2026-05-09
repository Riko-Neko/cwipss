from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks


def _robust_zscore_1d(values: np.ndarray) -> np.ndarray:
    profile = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(profile)
    if not np.any(finite):
        return np.zeros_like(profile, dtype=np.float32)
    median = float(np.nanmedian(profile[finite]))
    centered = profile - median
    mad = float(np.nanmedian(np.abs(centered[finite])))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = float(np.nanstd(centered[finite]))
    if not np.isfinite(scale) or scale <= 1e-6:
        return np.zeros_like(profile, dtype=np.float32)
    score = centered / scale
    score[~finite] = 0.0
    return score.astype(np.float32, copy=False)


def channel_period_peak_score(
    image: np.ndarray,
    sigma_peak: float = 1.0,
    sigma_background: float = 10.0,
) -> np.ndarray:
    """Enhance short period-profile peaks independently in each channel.

    The input is a log-power map with shape `(periods, channels)`. Each
    frequency channel is treated as a 1D period profile. A narrow Gaussian keeps
    short bright bands, a broad Gaussian estimates that channel's slow period
    background, and a robust per-channel z-score makes the result comparable
    across channels.
    """
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("image must have shape (periods, channels)")
    periods, channels = values.shape
    score = np.zeros((periods, channels), dtype=np.float32)
    sigma_peak = max(0.0, float(sigma_peak))
    sigma_background = max(sigma_peak + 1e-6, float(sigma_background))
    for channel in range(channels):
        profile = values[:, channel].astype(np.float32, copy=True)
        finite = np.isfinite(profile)
        if not np.any(finite):
            continue
        fill = float(np.nanmedian(profile[finite]))
        profile[~finite] = fill
        narrow = (
            ndi.gaussian_filter1d(profile, sigma=sigma_peak, mode="nearest")
            if sigma_peak > 0
            else profile
        )
        broad = ndi.gaussian_filter1d(profile, sigma=sigma_background, mode="nearest")
        score[:, channel] = _robust_zscore_1d(narrow - broad)
    return score


def _interp_period(periods: np.ndarray, index_value: float) -> float:
    if periods.size == 0:
        return float("nan")
    idx = np.arange(periods.size, dtype=np.float64)
    return float(np.interp(float(index_value), idx, periods))


def _window_mean(profile: np.ndarray, left_ip: float, right_ip: float) -> float:
    left = max(0, int(np.floor(left_ip)))
    right = min(profile.size, int(np.ceil(right_ip)) + 1)
    if right <= left:
        return float(profile[min(max(left, 0), profile.size - 1)])
    return float(np.nanmean(profile[left:right]))


def summarize_channel_period_peaks(
    score: np.ndarray,
    power_cube: np.ndarray,
    periods: np.ndarray,
    freqs_mhz: np.ndarray,
    record_start: int,
    record_stop: int,
    threshold: float,
    min_prominence: float,
    min_width_bins: float,
    max_width_bins: float,
    min_distance_bins: int,
    max_candidates_per_channel: int,
    max_candidates: int | None = None,
) -> list[dict]:
    """Summarize 1D period-profile peaks independently for each channel."""
    score_map = np.asarray(score, dtype=np.float32)
    power = np.asarray(power_cube, dtype=np.float32)
    period_values = np.asarray(periods, dtype=np.float64)
    freqs = np.asarray(freqs_mhz, dtype=np.float64)
    if score_map.ndim != 2:
        raise ValueError("score must have shape (periods, channels)")
    if power.ndim != 3:
        raise ValueError("power_cube must have shape (periods, records, channels)")
    if score_map.shape[0] != period_values.size or score_map.shape[1] != freqs.size:
        raise ValueError("score shape must match periods and freqs_mhz")

    width_min = max(0.0, float(min_width_bins))
    width_max = max(width_min, float(max_width_bins))
    distance = max(1, int(min_distance_bins))
    per_channel_limit = max(1, int(max_candidates_per_channel))
    rows: list[dict] = []

    for channel_idx in range(score_map.shape[1]):
        profile = np.asarray(score_map[:, channel_idx], dtype=np.float32)
        finite_profile = np.where(np.isfinite(profile), profile, -np.inf)
        peaks, props = find_peaks(
            finite_profile,
            height=float(threshold),
            prominence=float(min_prominence),
            width=(width_min, width_max),
            distance=distance,
        )
        if peaks.size == 0:
            continue
        order = np.argsort(
            np.asarray(props.get("prominences", np.zeros_like(peaks, dtype=np.float64)))
            + np.asarray(props.get("peak_heights", finite_profile[peaks]), dtype=np.float64)
        )[::-1]
        for peak_order in order[:per_channel_limit]:
            peak_idx = int(peaks[int(peak_order)])
            left_ip = float(props["left_ips"][int(peak_order)])
            right_ip = float(props["right_ips"][int(peak_order)])
            width_bins = float(props["widths"][int(peak_order)])
            period_start = _interp_period(period_values, left_ip)
            period_stop = _interp_period(period_values, right_ip)
            peak_period = float(period_values[peak_idx])
            time_power = power[peak_idx, :, channel_idx]
            peak_record = int(record_start + int(np.nanargmax(time_power)))
            rows.append(
                {
                    "detection_method": "channel_period_peak_dog",
                    "channel_index": int(channel_idx),
                    "record_start": int(record_start),
                    "record_stop": int(record_stop),
                    "duration_records": int(record_stop - record_start),
                    "period_start_records": period_start,
                    "period_stop_records": period_stop,
                    "period_width_records": float(abs(period_stop - period_start)),
                    "peak_period_records": peak_period,
                    "freq_start_mhz": float(freqs[channel_idx]),
                    "freq_stop_mhz": float(freqs[channel_idx]),
                    "bandwidth_mhz": 0.0,
                    "peak_record": peak_record,
                    "peak_freq_mhz": float(freqs[channel_idx]),
                    "peak_score": float(finite_profile[peak_idx]),
                    "mean_score": _window_mean(profile, left_ip, right_ip),
                    "peak_prominence": float(props["prominences"][int(peak_order)]),
                    "peak_width_bins": width_bins,
                    "peak_width_records": float(abs(period_stop - period_start)),
                }
            )

    rows.sort(key=lambda row: (row["peak_score"], row["peak_prominence"]), reverse=True)
    if max_candidates is not None:
        rows = rows[: max(0, int(max_candidates))]
    return rows


def add_candidate_ids(candidates: Iterable[dict]) -> list[dict]:
    rows = sorted(candidates, key=lambda row: row["peak_score"], reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["candidate_id"] = idx
    return rows
