"""Calibrated Persistent Ridge Occupancy (CPRO) detector."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2, sqrt

import numpy as np
from scipy import ndimage


CPRO_DETECTOR = "calibrated_persistent_ridge_occupancy"
MIN_POSITIVE = float(np.finfo(np.float32).tiny)


@dataclass(frozen=True)
class CPROParameters:
    """Scientific parameters of the validated CPRO configuration."""

    threshold_snr: float = 32.0
    texture_quantile: float = 0.9375
    period_center_bins: int = 3
    period_context_bins: int = 15
    min_period_contrast: float = 1.5
    support_records: int = 65
    min_occupancy: float = 0.65
    period_support_bins: int = 3
    window_support_records: int = 769
    min_window_occupancy: float = 0.40

    def validate(self) -> None:
        if self.threshold_snr <= 0.0:
            raise ValueError("cpro_threshold_snr must be positive")
        if not 0.0 <= self.texture_quantile < 1.0:
            raise ValueError("cpro_texture_quantile must be in [0, 1)")
        if self.period_center_bins < 1:
            raise ValueError("cpro_period_center_bins must be positive")
        if self.period_context_bins < self.period_center_bins:
            raise ValueError("cpro_period_context_bins must not be narrower than the center")
        if self.min_period_contrast < 0.0:
            raise ValueError("cpro_min_period_contrast must be non-negative")
        if self.support_records < 1:
            raise ValueError("cpro_support_records must be positive")
        if not 0.0 < self.min_occupancy <= 1.0:
            raise ValueError("cpro_min_occupancy must be in (0, 1]")
        if self.period_support_bins < 1:
            raise ValueError("cpro_period_support_bins must be positive")
        if self.window_support_records < 1:
            raise ValueError("cpro_window_support_records must be positive")
        if not 0.0 <= self.min_window_occupancy <= 1.0:
            raise ValueError("cpro_min_window_occupancy must be in [0, 1]")


@dataclass(frozen=True)
class CPROActivityResult:
    activity: np.ndarray
    score_map: np.ndarray
    occupancy_map: np.ndarray
    ridge_mask: np.ndarray
    ridge_time_mask: np.ndarray
    window_occupancy: np.ndarray
    active_mask: np.ndarray
    threshold: float


def cpro_period_mask(
    periods: np.ndarray,
    minimum: float | None,
    maximum: float | None,
) -> np.ndarray:
    values = np.asarray(periods, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("periods must be a 1D array")
    mask = np.isfinite(values) & (values > 0.0)
    if minimum is not None:
        mask &= values >= float(minimum)
    if maximum is not None:
        mask &= values <= float(maximum)
    if not np.any(mask):
        raise ValueError("candidate period range does not intersect the CWT period grid")
    return mask


def difference_noise_std(values: np.ndarray) -> float:
    """Estimate white-noise sigma from first differences without fallback."""
    series = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(series)
    if series.ndim != 1 or np.count_nonzero(finite) < 3:
        raise ValueError("noise calibration requires a finite 1D series")
    differences = np.diff(series[finite])
    center = float(np.median(differences))
    mad = float(np.median(np.abs(differences - center)))
    sigma = 1.4826 * mad / sqrt(2.0)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("first-difference noise sigma must be finite and positive")
    return sigma


def impulse_cwt_noise_gain(
    periods: np.ndarray,
    *,
    wavelet: str,
    method: str = "fft",
) -> np.ndarray:
    """Return the canonical per-period CWT power gain for unit white noise."""
    from .cwt import cwt_power_cube

    period_values = np.asarray(periods, dtype=np.float64)
    if period_values.ndim != 1 or period_values.size == 0 or np.any(period_values <= 0.0):
        raise ValueError("periods must be a positive 1D array")
    required = max(4096, int(32.0 * float(np.max(period_values))))
    records = 1 << int(ceil(log2(required)))
    impulse = np.zeros((records, 1), dtype=np.float32)
    impulse[records // 2, 0] = 1.0
    power = cwt_power_cube(
        impulse,
        period_values,
        wavelet=wavelet,
        normalize_channels=False,
        method=method,
        backend="cpu",
    )[:, :, 0]
    gain = np.sum(power, axis=1, dtype=np.float64)
    if not np.all(np.isfinite(gain)) or np.any(gain <= 0.0):
        raise ValueError("CWT impulse gains must be finite and positive")
    return gain.astype(np.float32)


def _edge_corrected_time_mean(values: np.ndarray, width: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    records = int(matrix.shape[1])
    size = max(1, min(int(width), records))
    if size <= 1:
        return matrix.astype(np.float32, copy=False)
    summed = ndimage.uniform_filter1d(
        matrix,
        size=size,
        axis=1,
        mode="constant",
        cval=0.0,
    ) * float(size)
    counts = ndimage.uniform_filter1d(
        np.ones(records, dtype=np.float32),
        size=size,
        mode="constant",
        cval=0.0,
    ) * float(size)
    return (summed / np.maximum(counts[None, :], 1.0)).astype(np.float32, copy=False)


def _contiguous_period_support(mask: np.ndarray, bins: int) -> np.ndarray:
    width = max(1, min(int(bins), int(mask.shape[0])))
    if width <= 1:
        return mask.astype(bool, copy=False)
    count = ndimage.uniform_filter1d(
        mask.astype(np.float32),
        size=width,
        axis=0,
        mode="constant",
    )
    return count >= (1.0 - 0.5 / float(width))


def _period_ridge_contrast(
    calibrated_power: np.ndarray,
    *,
    center_bins: int,
    context_bins: int,
) -> np.ndarray:
    center_width = max(1, min(int(center_bins), int(calibrated_power.shape[0])))
    context_width = max(center_width, min(int(context_bins), int(calibrated_power.shape[0])))
    center = ndimage.uniform_filter1d(
        calibrated_power,
        size=center_width,
        axis=0,
        mode="nearest",
    )
    if context_width == center_width:
        return np.ones_like(center, dtype=np.float32)
    context = ndimage.uniform_filter1d(
        calibrated_power,
        size=context_width,
        axis=0,
        mode="nearest",
    )
    side = (
        float(context_width) * context - float(center_width) * center
    ) / float(context_width - center_width)
    return (center / np.maximum(side, MIN_POSITIVE)).astype(np.float32, copy=False)


def cpro_activity(
    power: np.ndarray,
    *,
    noise_std: float,
    noise_gain: np.ndarray,
    params: CPROParameters | None = None,
) -> CPROActivityResult:
    """Stage 1 only: compress one absolute CWT map into time activity."""
    params = params or CPROParameters()
    params.validate()
    values = np.asarray(power, dtype=np.float32)
    gain = np.asarray(noise_gain, dtype=np.float32)
    if values.ndim != 2 or gain.shape != (values.shape[0],):
        raise ValueError("power and noise_gain must match on the period axis")
    if not np.isfinite(noise_std) or noise_std <= 0.0:
        raise ValueError("noise_std must be finite and positive")
    values = np.maximum(np.where(np.isfinite(values), values, 0.0), 0.0)
    denominator = np.maximum(float(noise_std) ** 2 * gain[:, None], MIN_POSITIVE)
    calibrated = values / denominator
    threshold = float(params.threshold_snr)
    if params.texture_quantile > 0.0:
        threshold = max(threshold, float(np.quantile(calibrated, params.texture_quantile)))
    period_contrast = _period_ridge_contrast(
        calibrated,
        center_bins=params.period_center_bins,
        context_bins=params.period_context_bins,
    )
    exceedance = calibrated >= threshold
    if params.min_period_contrast > 0.0:
        exceedance &= period_contrast >= float(params.min_period_contrast)
    occupancy = _edge_corrected_time_mean(exceedance.astype(np.float32), params.support_records)
    persistent = occupancy >= float(params.min_occupancy)
    persistent = _contiguous_period_support(persistent, params.period_support_bins)

    occupied_power = _edge_corrected_time_mean(calibrated * exceedance, params.support_records)
    bright_mean = occupied_power / np.maximum(
        occupancy,
        1.0 / float(max(1, params.support_records)),
    )
    ridge_time_mask = np.any(persistent, axis=0)
    window_occupancy_map = persistent.astype(np.float32)
    if params.window_support_records > 1 and params.min_window_occupancy > 0.0:
        window_occupancy_map = _edge_corrected_time_mean(
            persistent.astype(np.float32),
            params.window_support_records,
        )
        window_ridge_mask = window_occupancy_map >= float(params.min_window_occupancy)
    else:
        window_ridge_mask = persistent
    score_map = np.where(window_ridge_mask, bright_mean, 0.0).astype(np.float32, copy=False)
    activity = np.max(score_map, axis=0).astype(np.float32, copy=False)
    window_occupancy = np.max(window_occupancy_map, axis=0).astype(np.float32, copy=False)
    active_mask = np.any(window_ridge_mask, axis=0)
    return CPROActivityResult(
        activity=activity,
        score_map=score_map,
        occupancy_map=occupancy.astype(np.float32, copy=False),
        ridge_mask=persistent,
        ridge_time_mask=ridge_time_mask,
        window_occupancy=window_occupancy,
        active_mask=active_mask,
        threshold=threshold,
    )
