"""Calibrated Period-Ridge Observation (CPRO) detector."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, log2, sqrt

import numpy as np
from scipy import ndimage, signal


CPRO_DETECTOR = "calibrated_period_ridge_observation"
MIN_POSITIVE = float(np.finfo(np.float32).tiny)


@dataclass(frozen=True)
class CPROParameters:
    """Scientific parameters of the edge-preserving CPRO configuration."""

    threshold_snr: float = 32.0
    texture_quantile: float = 0.9375
    period_center_bins: int = 3
    period_context_bins: int = 15
    min_period_contrast: float = 1.5
    period_support_bins: int = 3
    shape_power_softness: float = 1.0
    shape_contrast_softness: float = 0.10
    continuity_decay: float = 0.995
    continuity_power: float = 2.0
    min_continuity_mean: float = 0.47
    min_ridge_lock: float = 0.94

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
        if self.period_support_bins < 1:
            raise ValueError("cpro_period_support_bins must be positive")
        if self.shape_power_softness <= 0.0:
            raise ValueError("cpro_shape_power_softness must be positive")
        if self.shape_contrast_softness <= 0.0:
            raise ValueError("cpro_shape_contrast_softness must be positive")
        if not 0.0 <= self.continuity_decay < 1.0:
            raise ValueError("cpro_continuity_decay must be in [0, 1)")
        if self.continuity_power <= 0.0:
            raise ValueError("cpro_continuity_power must be positive")
        if self.min_continuity_mean < 0.0:
            raise ValueError("cpro_min_continuity_mean must be non-negative")
        if not 0.0 <= self.min_ridge_lock <= 1.0:
            raise ValueError("cpro_min_ridge_lock must be in [0, 1]")


@dataclass(frozen=True)
class CPROActivityResult:
    shape_activity: np.ndarray
    shape_map: np.ndarray
    threshold: float


def _exponential_mean(values: np.ndarray, decay: float) -> np.ndarray:
    """Apply a boundary-initialized first-order IIR along time."""
    data = np.asarray(values, dtype=np.float32)
    initial = float(decay) * data[:, :1]
    filtered, _state = signal.lfilter(
        [1.0 - float(decay)],
        [1.0, -float(decay)],
        data,
        axis=1,
        zi=initial,
    )
    return filtered.astype(np.float32, copy=False)


def cpro_continuity_map(
    shape_map: np.ndarray,
    *,
    decay: float,
    power: float,
) -> np.ndarray:
    """Retain responses supported from both time directions on one CWT ridge."""
    if not 0.0 <= float(decay) < 1.0:
        raise ValueError("continuity decay must be in [0, 1)")
    if float(power) <= 0.0:
        raise ValueError("continuity power must be positive")
    values = np.maximum(np.asarray(shape_map, dtype=np.float32), 0.0)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("shape_map must have shape (periods, non-empty records)")
    forward = _exponential_mean(values, decay)
    backward = _exponential_mean(values[:, ::-1], decay)[:, ::-1]
    support = np.sqrt(np.maximum(forward * backward, 0.0))
    ratio = np.divide(
        np.minimum(support, values),
        values,
        out=np.zeros_like(values),
        where=values > 0.0,
    )
    return (
        values * np.power(ratio, float(power), dtype=np.float32)
    ).astype(np.float32, copy=False)


def cpro_continuity_features(
    continuity_map: np.ndarray,
    start: int,
    stop: int,
    *,
    threshold: float,
) -> tuple[float, float]:
    """Return normalized continuity strength and single-ridge energy locking."""
    values = np.maximum(np.asarray(continuity_map, dtype=np.float32), 0.0)
    start = int(start)
    stop = int(stop)
    if values.ndim != 2 or not 0 <= start < stop <= values.shape[1]:
        raise ValueError("continuity window must be a non-empty in-range interval")
    if not np.isfinite(threshold) or float(threshold) <= 0.0:
        raise ValueError("CPRO threshold must be finite and positive")
    window = values[:, start:stop]
    time_peak = np.max(window, axis=0)
    continuity_mean = float(np.mean(time_peak, dtype=np.float64)) / float(threshold)
    ridge_energy = np.sum(window, axis=1, dtype=np.float64)
    ridge_lock = float(np.max(ridge_energy)) / max(
        float(np.sum(time_peak, dtype=np.float64)),
        MIN_POSITIVE,
    )
    return continuity_mean, ridge_lock


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


@lru_cache(maxsize=32)
def _cached_impulse_cwt_noise_gain(
    period_key: tuple[float, ...],
    wavelet: str,
    method: str,
) -> np.ndarray:
    from .cwt import cwt_power_cube

    period_values = np.asarray(period_key, dtype=np.float64)
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
    gain = np.sum(power, axis=1, dtype=np.float64).astype(np.float32)
    gain.setflags(write=False)
    return gain


def impulse_cwt_noise_gain(
    periods: np.ndarray,
    *,
    wavelet: str,
    method: str = "fft",
) -> np.ndarray:
    """Return a process-cached canonical CWT power gain for unit white noise."""
    period_values = np.asarray(periods, dtype=np.float64)
    if period_values.ndim != 1 or period_values.size == 0 or np.any(period_values <= 0.0):
        raise ValueError("periods must be a positive 1D array")
    gain = _cached_impulse_cwt_noise_gain(
        tuple(float(value) for value in period_values),
        str(wavelet),
        str(method),
    )
    if not np.all(np.isfinite(gain)) or np.any(gain <= 0.0):
        raise ValueError("CWT impulse gains must be finite and positive")
    return gain


def _period_mean(values: np.ndarray, bins: int) -> np.ndarray:
    width = max(1, min(int(bins), int(values.shape[0])))
    if width <= 1:
        return values.astype(np.float32, copy=False)
    return ndimage.uniform_filter1d(
        values,
        size=width,
        axis=0,
        mode="nearest",
    ).astype(np.float32, copy=False)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32, copy=False)


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
    target_period_mask: np.ndarray | None = None,
    params: CPROParameters | None = None,
) -> CPROActivityResult:
    """Compress one CWT2D map into an edge-preserving ridge-observation axis."""
    params = params or CPROParameters()
    params.validate()
    values = np.asarray(power, dtype=np.float32)
    gain = np.asarray(noise_gain, dtype=np.float32)
    if values.ndim != 2 or gain.shape != (values.shape[0],):
        raise ValueError("power and noise_gain must match on the period axis")
    target = np.ones(values.shape[0], dtype=bool)
    if target_period_mask is not None:
        target = np.asarray(target_period_mask, dtype=bool)
        if target.shape != (values.shape[0],) or not np.any(target):
            raise ValueError("target_period_mask must select at least one period row")
    if not np.isfinite(noise_std) or noise_std <= 0.0:
        raise ValueError("noise_std must be finite and positive")
    values = np.maximum(np.where(np.isfinite(values), values, 0.0), 0.0)
    denominator = np.maximum(float(noise_std) ** 2 * gain[:, None], MIN_POSITIVE)
    calibrated = values / denominator
    threshold = float(params.threshold_snr)
    if params.texture_quantile > 0.0:
        threshold = max(
            threshold,
            float(np.quantile(calibrated[target], params.texture_quantile)),
        )
    period_contrast = _period_ridge_contrast(
        calibrated,
        center_bins=params.period_center_bins,
        context_bins=params.period_context_bins,
    )
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
    shape_map = _period_mean(
        shape_power * contrast_weight,
        params.period_support_bins,
    )
    shape_map = np.where(target[:, None], shape_map, 0.0).astype(np.float32, copy=False)
    shape_activity = np.max(shape_map, axis=0).astype(np.float32, copy=False)
    return CPROActivityResult(
        shape_activity=shape_activity,
        shape_map=shape_map,
        threshold=threshold,
    )
