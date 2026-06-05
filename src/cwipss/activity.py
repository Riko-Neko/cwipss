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


def period_robust_zscore(
    excess: np.ndarray,
    *,
    baseline_quantile: float = 0.10,
    scale_quantile: float = 0.20,
) -> np.ndarray:
    """Center and scale each CWT period row over time.

    Low-fraction floor division makes pure-noise power maps positively biased.
    This step removes each period row's low-time-quantile background before any
    period-axis compression. A low quantile is used instead of the row median so
    long-duration signals are not treated as their own background.
    """
    values = np.asarray(excess, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("excess must have shape (periods, records)")
    q_bg = min(max(float(baseline_quantile), 0.0), 0.45)
    q_scale = min(max(float(scale_quantile), q_bg + 1e-6), 0.50)
    baseline = np.nanquantile(values, q_bg, axis=1, keepdims=True)
    centered = values - baseline
    low_cut = np.nanquantile(values, q_scale, axis=1, keepdims=True)
    low_centered = np.where(values <= low_cut, centered, np.nan)
    low_median = np.nanmedian(low_centered, axis=1, keepdims=True)
    low_mad = np.nanmedian(np.abs(low_centered - low_median), axis=1, keepdims=True)
    scale = 1.4826 * low_mad
    low_std = np.nanstd(low_centered, axis=1, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, low_std)
    fallback = np.nanstd(centered, axis=1, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    z = centered / scale
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32, copy=False)


def coherent_structure_map(
    excess: np.ndarray,
    *,
    baseline_quantile: float = 0.10,
    scale_quantile: float = 0.20,
    z_threshold: float = 1.0,
    time_support_records: int = 64,
    period_support_bins: int = 3,
    min_support_fraction: float = 0.10,
) -> np.ndarray:
    """Return a 2D map that keeps coherent positive CWT structures.

    Isolated random bright texture can survive floor division, but it usually
    lacks support across nearby times and period bins. The support gate keeps
    only positive standardized excess with local 2D persistence.
    """
    z = period_robust_zscore(
        excess,
        baseline_quantile=baseline_quantile,
        scale_quantile=scale_quantile,
    )
    positive = np.maximum(z, 0.0)
    threshold = float(z_threshold)
    support = (z > threshold).astype(np.float32)
    time_width = max(1, int(time_support_records))
    period_width = max(1, int(period_support_bins))
    if time_width > 1:
        support = uniform_filter1d(support, size=time_width, axis=1, mode="nearest")
    if period_width > 1:
        support = uniform_filter1d(support, size=period_width, axis=0, mode="nearest")
    floor = min(max(float(min_support_fraction), 0.0), 0.95)
    if floor > 0.0:
        weight = np.clip((support - floor) / max(1e-6, 1.0 - floor), 0.0, 1.0)
    else:
        weight = np.clip(support, 0.0, 1.0)
    structured = positive * weight
    structured[~np.isfinite(structured)] = 0.0
    return structured.astype(np.float32, copy=False)


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
