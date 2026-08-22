"""CUDA implementation of Calibrated Persistent Ridge Occupancy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cpro import CPROParameters, MIN_POSITIVE


@dataclass(frozen=True)
class CPROActivityCudaResult:
    shape_activity: object
    shape_map: object
    occupancy_map: object
    threshold: object


def _cupy_modules():
    try:
        import cupy as cp
        from cupyx.scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("CPRO CUDA requires CuPy with cupyx.scipy") from exc
    return cp, ndimage


def difference_noise_std_cuda(values):
    """Estimate one channel's noise sigma entirely on CUDA."""
    cp, _ndimage = _cupy_modules()
    series = cp.asarray(values, dtype=cp.float64)
    if series.ndim != 1:
        raise ValueError("noise calibration requires a finite 1D series")
    finite_values = series[cp.isfinite(series)]
    if int(finite_values.size) < 3:
        raise ValueError("noise calibration requires a finite 1D series")
    differences = cp.diff(finite_values)
    center = cp.median(differences)
    mad = cp.median(cp.abs(differences - center))
    sigma = 1.4826 * mad / cp.sqrt(cp.asarray(2.0, dtype=cp.float64))
    if not bool(cp.isfinite(sigma) & (sigma > 0.0)):
        raise ValueError("first-difference noise sigma must be finite and positive")
    return sigma


def _edge_corrected_time_mean(values, width: int, cp, ndimage):
    matrix = cp.asarray(values, dtype=cp.float32)
    records = int(matrix.shape[1])
    size = max(1, min(int(width), records))
    if size <= 1:
        return matrix.astype(cp.float32, copy=False)
    summed = ndimage.uniform_filter1d(
        matrix,
        size=size,
        axis=1,
        mode="constant",
        cval=0.0,
    ) * float(size)
    counts = ndimage.uniform_filter1d(
        cp.ones(records, dtype=cp.float32),
        size=size,
        mode="constant",
        cval=0.0,
    ) * float(size)
    return (summed / cp.maximum(counts[None, :], 1.0)).astype(cp.float32, copy=False)


def _period_mean(values, bins: int, cp, ndimage):
    width = max(1, min(int(bins), int(values.shape[0])))
    if width <= 1:
        return values.astype(cp.float32, copy=False)
    return ndimage.uniform_filter1d(
        values,
        size=width,
        axis=0,
        mode="constant",
        cval=0.0,
    ).astype(cp.float32, copy=False)


def _sigmoid(values, cp):
    clipped = cp.clip(values, -40.0, 40.0)
    return (1.0 / (1.0 + cp.exp(-clipped))).astype(cp.float32, copy=False)


def _top_k_period_mean(values, top_k: int, cp):
    count = max(1, min(int(top_k), int(values.shape[0])))
    split = int(values.shape[0]) - count
    top = cp.partition(values, split, axis=0)[split:, :]
    return cp.mean(top, axis=0, dtype=cp.float32).astype(cp.float32, copy=False)


def _period_ridge_contrast(calibrated, params: CPROParameters, cp, ndimage):
    center_width = max(1, min(int(params.period_center_bins), int(calibrated.shape[0])))
    context_width = max(
        center_width,
        min(int(params.period_context_bins), int(calibrated.shape[0])),
    )
    center = ndimage.uniform_filter1d(
        calibrated,
        size=center_width,
        axis=0,
        mode="nearest",
    )
    if context_width == center_width:
        return cp.ones_like(center, dtype=cp.float32)
    context = ndimage.uniform_filter1d(
        calibrated,
        size=context_width,
        axis=0,
        mode="nearest",
    )
    side = (
        float(context_width) * context - float(center_width) * center
    ) / float(context_width - center_width)
    return (center / cp.maximum(side, MIN_POSITIVE)).astype(cp.float32, copy=False)


def cpro_activity_cuda(
    power,
    *,
    noise_std,
    noise_gain: np.ndarray,
    params: CPROParameters | None = None,
) -> CPROActivityCudaResult:
    """Stage 1 only: compress one absolute CWT map entirely on CUDA."""
    params = params or CPROParameters()
    params.validate()
    cp, ndimage = _cupy_modules()
    values = cp.asarray(power, dtype=cp.float32)
    gain = cp.asarray(noise_gain, dtype=cp.float32)
    if values.ndim != 2 or gain.shape != (values.shape[0],):
        raise ValueError("power and noise_gain must match on the period axis")
    sigma = cp.asarray(noise_std, dtype=cp.float64)
    if not bool(cp.isfinite(sigma) & (sigma > 0.0)):
        raise ValueError("noise_std must be finite and positive")
    values = cp.maximum(cp.where(cp.isfinite(values), values, 0.0), 0.0)
    denominator = cp.maximum(sigma**2 * gain[:, None], MIN_POSITIVE)
    calibrated = values / denominator
    threshold = cp.asarray(params.threshold_snr, dtype=cp.float64)
    if params.texture_quantile > 0.0:
        threshold = cp.maximum(threshold, cp.quantile(calibrated, params.texture_quantile))
    period_contrast = _period_ridge_contrast(calibrated, params, cp, ndimage)
    exceedance = calibrated >= threshold
    if params.min_period_contrast > 0.0:
        exceedance &= period_contrast >= float(params.min_period_contrast)
    occupancy = _edge_corrected_time_mean(
        exceedance.astype(cp.float32),
        params.support_records,
        cp,
        ndimage,
    )

    power_log_ratio = cp.log(cp.maximum(calibrated, MIN_POSITIVE) / threshold)
    shape_power = threshold * float(params.shape_power_softness) * cp.logaddexp(
        0.0,
        power_log_ratio / float(params.shape_power_softness),
    )
    if params.min_period_contrast > 0.0:
        contrast_log_ratio = cp.log(
            cp.maximum(period_contrast, MIN_POSITIVE) / float(params.min_period_contrast)
        )
        contrast_weight = _sigmoid(
            contrast_log_ratio / float(params.shape_contrast_softness),
            cp,
        )
    else:
        contrast_weight = cp.ones_like(shape_power, dtype=cp.float32)
    shape_map = _edge_corrected_time_mean(
        shape_power * contrast_weight,
        params.support_records,
        cp,
        ndimage,
    )
    shape_map = _period_mean(shape_map, params.period_support_bins, cp, ndimage)
    occupancy_weight = _sigmoid(
        (occupancy - float(params.min_occupancy))
        / float(params.shape_occupancy_softness),
        cp,
    )
    occupancy_weight = _period_mean(
        occupancy_weight,
        params.period_support_bins,
        cp,
        ndimage,
    )
    window_support = _edge_corrected_time_mean(
        occupancy_weight,
        params.window_support_records,
        cp,
        ndimage,
    )
    window_weight = _sigmoid(
        (window_support - float(params.min_window_occupancy))
        / float(params.shape_occupancy_softness),
        cp,
    )
    shape_map = (shape_map * window_weight).astype(cp.float32, copy=False)
    shape_activity = _top_k_period_mean(shape_map, params.shape_top_k, cp)
    return CPROActivityCudaResult(
        shape_activity=shape_activity,
        shape_map=shape_map,
        occupancy_map=occupancy.astype(cp.float32, copy=False),
        threshold=threshold,
    )
