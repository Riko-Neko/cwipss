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
    shape_power_softness: float = 0.50
    shape_contrast_softness: float = 0.25
    shape_occupancy_softness: float = 0.10
    shape_top_k: int = 3

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
        if self.shape_power_softness <= 0.0:
            raise ValueError("cpro_shape_power_softness must be positive")
        if self.shape_contrast_softness <= 0.0:
            raise ValueError("cpro_shape_contrast_softness must be positive")
        if self.shape_occupancy_softness <= 0.0:
            raise ValueError("cpro_shape_occupancy_softness must be positive")
        if self.shape_top_k < 1:
            raise ValueError("cpro_shape_top_k must be positive")


@dataclass(frozen=True)
class CPROActivityResult:
    shape_activity: np.ndarray
    shape_map: np.ndarray
    occupancy_map: np.ndarray
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


def _period_mean(values: np.ndarray, bins: int) -> np.ndarray:
    width = max(1, min(int(bins), int(values.shape[0])))
    if width <= 1:
        return values.astype(np.float32, copy=False)
    return ndimage.uniform_filter1d(
        values,
        size=width,
        axis=0,
        mode="constant",
        cval=0.0,
    ).astype(np.float32, copy=False)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32, copy=False)


def _top_k_period_mean(values: np.ndarray, top_k: int) -> np.ndarray:
    count = max(1, min(int(top_k), int(values.shape[0])))
    split = int(values.shape[0]) - count
    top = np.partition(values, split, axis=0)[split:, :]
    return np.mean(top, axis=0, dtype=np.float32).astype(np.float32, copy=False)


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
    """Compress one CWT2D map into the continuous time-proposal axis."""
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

    power_log_ratio = np.log(np.maximum(calibrated, MIN_POSITIVE) / threshold)
    shape_power = threshold * float(params.shape_power_softness) * np.logaddexp(
        0.0,
        power_log_ratio / float(params.shape_power_softness),
    )
    if params.min_period_contrast > 0.0:
        contrast_log_ratio = np.log(
            np.maximum(period_contrast, MIN_POSITIVE) / float(params.min_period_contrast)
        )
        contrast_weight = _sigmoid(
            contrast_log_ratio / float(params.shape_contrast_softness)
        )
    else:
        contrast_weight = np.ones_like(shape_power, dtype=np.float32)
    shape_map = _edge_corrected_time_mean(
        shape_power * contrast_weight,
        params.support_records,
    )
    shape_map = _period_mean(shape_map, params.period_support_bins)
    occupancy_weight = _sigmoid(
        (occupancy - float(params.min_occupancy))
        / float(params.shape_occupancy_softness)
    )
    occupancy_weight = _period_mean(occupancy_weight, params.period_support_bins)
    window_support = _edge_corrected_time_mean(
        occupancy_weight,
        params.window_support_records,
    )
    window_weight = _sigmoid(
        (window_support - float(params.min_window_occupancy))
        / float(params.shape_occupancy_softness)
    )
    shape_map *= window_weight
    shape_map = shape_map.astype(np.float32, copy=False)
    shape_activity = _top_k_period_mean(shape_map, params.shape_top_k)
    return CPROActivityResult(
        shape_activity=shape_activity,
        shape_map=shape_map,
        occupancy_map=occupancy.astype(np.float32, copy=False),
        threshold=threshold,
    )
