"""CUDA implementation of Calibrated Persistent Ridge Occupancy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cpro import CPROParameters, MIN_POSITIVE


@dataclass(frozen=True)
class CPROActivityCudaResult:
    activity: object
    score_map: object
    occupancy_map: object
    ridge_mask: object
    ridge_time_mask: object
    window_occupancy: object
    active_mask: object
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


def _contiguous_period_support(mask, bins: int, cp, ndimage):
    width = max(1, min(int(bins), int(mask.shape[0])))
    if width <= 1:
        return mask.astype(cp.bool_, copy=False)
    count = ndimage.uniform_filter1d(
        mask.astype(cp.float32),
        size=width,
        axis=0,
        mode="constant",
    )
    return count >= (1.0 - 0.5 / float(width))


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
    persistent = occupancy >= float(params.min_occupancy)
    persistent = _contiguous_period_support(
        persistent,
        params.period_support_bins,
        cp,
        ndimage,
    )

    occupied_power = _edge_corrected_time_mean(
        calibrated * exceedance,
        params.support_records,
        cp,
        ndimage,
    )
    bright_mean = occupied_power / cp.maximum(
        occupancy,
        1.0 / float(max(1, params.support_records)),
    )
    ridge_time_mask = cp.any(persistent, axis=0)
    window_occupancy_map = persistent.astype(cp.float32)
    if params.window_support_records > 1 and params.min_window_occupancy > 0.0:
        window_occupancy_map = _edge_corrected_time_mean(
            persistent.astype(cp.float32),
            params.window_support_records,
            cp,
            ndimage,
        )
        window_ridge_mask = window_occupancy_map >= float(params.min_window_occupancy)
    else:
        window_ridge_mask = persistent
    score_map = cp.where(window_ridge_mask, bright_mean, 0.0).astype(cp.float32, copy=False)
    activity = cp.max(score_map, axis=0).astype(cp.float32, copy=False)
    active_mask = cp.any(window_ridge_mask, axis=0)
    return CPROActivityCudaResult(
        activity=activity,
        score_map=score_map,
        occupancy_map=occupancy.astype(cp.float32, copy=False),
        ridge_mask=persistent,
        ridge_time_mask=ridge_time_mask,
        window_occupancy=cp.max(window_occupancy_map, axis=0).astype(cp.float32, copy=False),
        active_mask=active_mask,
        threshold=threshold,
    )
