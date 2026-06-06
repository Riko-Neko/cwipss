"""Windowed period profiles and peak extraction."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_widths


def windowed_period_profile(excess: np.ndarray, start: int, stop: int) -> np.ndarray:
    values = np.asarray(excess, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("excess must have shape (periods, records)")
    start = max(0, min(int(start), values.shape[1]))
    stop = max(start + 1, min(int(stop), values.shape[1]))
    window = values[:, start:stop]
    duration = max(1, int(stop - start))
    profile = np.nansum(window, axis=1) / np.sqrt(duration)
    profile[~np.isfinite(profile)] = 0.0
    return profile.astype(np.float32, copy=False)


def find_period_profile_peaks(
    profile: np.ndarray,
    periods: np.ndarray,
    *,
    min_prominence: float = 0.5,
    max_peaks: int = 3,
) -> list[dict]:
    values = np.asarray(profile, dtype=np.float32)
    period_values = np.asarray(periods, dtype=np.float64)
    if values.ndim != 1 or period_values.ndim != 1 or values.size != period_values.size:
        raise ValueError("profile and periods must be matching 1D arrays")
    if values.size == 0:
        return []

    peaks, props = find_peaks(values, prominence=max(0.0, float(min_prominence)))
    if peaks.size == 0:
        peaks = np.array([int(np.nanargmax(values))], dtype=np.int64)
        props = {"prominences": np.array([0.0], dtype=np.float64)}
    widths = peak_widths(values, peaks, rel_height=0.5)
    prominences = np.asarray(props.get("prominences", np.zeros(peaks.size)), dtype=np.float64)

    rows: list[dict] = []
    for idx, peak_idx in enumerate(peaks):
        left = int(max(0, np.floor(widths[2][idx])))
        right = int(min(values.size - 1, np.ceil(widths[3][idx])))
        rows.append(
            {
                "peak_period_index": int(peak_idx),
                "period_start_index": left,
                "period_stop_index": right,
                "peak_period_records": float(period_values[peak_idx]),
                "period_start_records": float(period_values[left]),
                "period_stop_records": float(period_values[right]),
                "period_width_bins": float(max(1, right - left + 1)),
                "period_width_records": float(abs(period_values[right] - period_values[left])),
                "profile_score": float(values[peak_idx]),
                "period_peak_prominence": float(prominences[idx]) if idx < prominences.size else 0.0,
            }
        )

    rows.sort(key=lambda row: (row["profile_score"], row["period_peak_prominence"]), reverse=True)
    return rows[: max(1, int(max_peaks))]
