from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


def robust_standardize(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)
    median = float(np.nanmedian(x[finite]))
    centered = x - median
    mad = float(np.nanmedian(np.abs(centered[finite])))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = float(np.nanstd(centered[finite]))
    if not np.isfinite(scale) or scale <= 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    z = centered / scale
    z[~finite] = 0.0
    return z.astype(np.float32, copy=False)


def valid_period_mask(
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


def crop_valid_periods(
    power: np.ndarray,
    periods: np.ndarray,
    min_period_records: float | None,
    max_period_records: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(power, dtype=np.float32)
    period_values = np.asarray(periods, dtype=np.float64)
    mask = valid_period_mask(period_values, min_period_records, max_period_records)
    if values.shape[0] != period_values.size:
        raise ValueError("power period axis must match periods")
    if not np.any(mask):
        raise ValueError("No CWT periods remain after candidate period filtering.")
    return values[mask, ...], period_values[mask], mask


def low_fraction_noise_floor(power: np.ndarray, fraction: float = 0.20) -> float:
    values = np.asarray(power, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    fraction = min(max(float(fraction), 1.0 / finite.size), 1.0)
    k = max(1, int(np.ceil(fraction * finite.size)))
    low = np.partition(finite, k - 1)[:k]
    floor = float(np.nanmean(low))
    if not np.isfinite(floor) or floor <= 0.0:
        positive = finite[finite > 0.0]
        if positive.size == 0:
            return 1.0
        floor = float(np.nanmedian(positive))
    return max(floor, 1e-12)


def relative_excess(power: np.ndarray, noise_floor: float, eps_fraction: float = 1e-6) -> np.ndarray:
    values = np.asarray(power, dtype=np.float32)
    floor = max(float(noise_floor), 1e-12)
    eps = max(1e-12, abs(floor) * float(eps_fraction))
    excess = values / (floor + eps) - 1.0
    excess[~np.isfinite(excess)] = 0.0
    return excess.astype(np.float32, copy=False)


def signed_trimmed_period_activity(
    excess: np.ndarray,
    trim_low: float = 0.05,
    trim_high: float = 0.95,
) -> np.ndarray:
    values = np.asarray(excess, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("excess must have shape (periods, records)")
    lo = min(max(float(trim_low), 0.0), 0.49)
    hi = min(max(float(trim_high), lo + 1e-6), 1.0)
    period_count = values.shape[0]
    if period_count == 0:
        return np.zeros(values.shape[1], dtype=np.float32)
    start = int(np.floor(lo * period_count))
    stop = int(np.ceil(hi * period_count))
    stop = min(max(start + 1, stop), period_count)
    sorted_values = np.sort(values, axis=0)
    activity = np.nanmean(sorted_values[start:stop, :], axis=0)
    activity[~np.isfinite(activity)] = 0.0
    return activity.astype(np.float32, copy=False)


def smooth_activity(activity: np.ndarray, smooth_records: int = 1) -> np.ndarray:
    values = np.asarray(activity, dtype=np.float32)
    width = max(1, int(smooth_records))
    if width <= 1 or values.size == 0:
        return values.astype(np.float32, copy=False)
    return uniform_filter1d(values, size=width, mode="nearest").astype(np.float32, copy=False)
