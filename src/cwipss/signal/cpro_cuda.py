"""CUDA implementation of Calibrated Period-Ridge Observation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .cpro import CPROParameters, MIN_POSITIVE


_CONTINUITY_KERNEL = r"""
extern "C" __global__
void cpro_bidirectional_continuity(
    const float* values,
    float* evidence,
    const int periods,
    const int records,
    const float decay,
    const float exponent
) {
    const int period = blockDim.x * blockIdx.x + threadIdx.x;
    if (period >= periods) return;
    const int offset = period * records;
    const float coefficient = 1.0f - decay;
    float state = values[offset];
    for (int time = 0; time < records; ++time) {
        const int index = offset + time;
        state = coefficient * values[index] + decay * state;
        evidence[index] = state;
    }
    state = values[offset + records - 1];
    for (int time = records - 1; time >= 0; --time) {
        const int index = offset + time;
        const float value = values[index];
        state = coefficient * value + decay * state;
        const float support = sqrtf(fmaxf(evidence[index] * state, 0.0f));
        const float ratio = value > 0.0f ? fminf(support, value) / value : 0.0f;
        evidence[index] = value * powf(ratio, exponent);
    }
}
"""


_WINDOW_RIDGE_SUM_KERNEL = r"""
extern "C" __global__
void cpro_window_ridge_sum(
    const float* values,
    const long long* starts,
    const long long* stops,
    double* sums,
    const int periods,
    const int records,
    const int windows
) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= periods * windows) return;
    const int period = index / windows;
    const int window = index - period * windows;
    double total = 0.0;
    const int offset = period * records;
    for (long long time = starts[window]; time < stops[window]; ++time) {
        total += (double)values[offset + time];
    }
    sums[index] = total;
}
"""


@dataclass(frozen=True)
class CPROActivityCudaResult:
    shape_activity: object
    shape_map: object
    threshold: object


def _cupy_modules():
    try:
        import cupy as cp
        from cupyx.scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("CPRO CUDA requires CuPy with cupyx.scipy") from exc
    return cp, ndimage


@lru_cache(maxsize=1)
def _cached_continuity_kernel():
    cp, _ndimage = _cupy_modules()
    return cp.RawKernel(_CONTINUITY_KERNEL, "cpro_bidirectional_continuity")


@lru_cache(maxsize=1)
def _cached_window_ridge_sum_kernel():
    cp, _ndimage = _cupy_modules()
    return cp.RawKernel(_WINDOW_RIDGE_SUM_KERNEL, "cpro_window_ridge_sum")


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


def _period_mean(values, bins: int, cp, ndimage):
    width = max(1, min(int(bins), int(values.shape[0])))
    if width <= 1:
        return values.astype(cp.float32, copy=False)
    return ndimage.uniform_filter1d(
        values,
        size=width,
        axis=0,
        mode="nearest",
    ).astype(cp.float32, copy=False)


def _sigmoid(values, cp):
    clipped = cp.clip(values, -40.0, 40.0)
    return (1.0 / (1.0 + cp.exp(-clipped))).astype(cp.float32, copy=False)


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
    target_period_mask: np.ndarray | None = None,
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
    target = cp.ones(values.shape[0], dtype=cp.bool_)
    if target_period_mask is not None:
        target_host = np.asarray(target_period_mask, dtype=bool)
        if target_host.shape != (values.shape[0],) or not np.any(target_host):
            raise ValueError("target_period_mask must select at least one period row")
        target = cp.asarray(target_host)
    sigma = cp.asarray(noise_std, dtype=cp.float64)
    if not bool(cp.isfinite(sigma) & (sigma > 0.0)):
        raise ValueError("noise_std must be finite and positive")
    values = cp.maximum(cp.where(cp.isfinite(values), values, 0.0), 0.0)
    denominator = cp.maximum(sigma**2 * gain[:, None], MIN_POSITIVE)
    calibrated = values / denominator
    threshold = cp.asarray(params.threshold_snr, dtype=cp.float64)
    if params.texture_quantile > 0.0:
        threshold = cp.maximum(
            threshold,
            cp.quantile(calibrated[target], params.texture_quantile),
        )
    period_contrast = _period_ridge_contrast(calibrated, params, cp, ndimage)
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
    shape_map = _period_mean(
        shape_power * contrast_weight,
        params.period_support_bins,
        cp,
        ndimage,
    )
    shape_map = cp.where(target[:, None], shape_map, 0.0).astype(cp.float32, copy=False)
    shape_activity = cp.max(shape_map, axis=0).astype(cp.float32, copy=False)
    return CPROActivityCudaResult(
        shape_activity=shape_activity,
        shape_map=shape_map,
        threshold=threshold,
    )


def cpro_continuity_map_cuda(shape_map, *, decay: float, power: float):
    """Compute bidirectional single-ridge continuity without a host transfer."""
    if not 0.0 <= float(decay) < 1.0:
        raise ValueError("continuity decay must be in [0, 1)")
    if float(power) <= 0.0:
        raise ValueError("continuity power must be positive")
    cp, _ndimage = _cupy_modules()
    values = cp.maximum(cp.asarray(shape_map, dtype=cp.float32), 0.0)
    if values.ndim != 2 or int(values.shape[1]) == 0:
        raise ValueError("shape_map must have shape (periods, non-empty records)")
    evidence = cp.empty_like(values)
    kernel = _cached_continuity_kernel()
    threads = 128
    blocks = (int(values.shape[0]) + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            values,
            evidence,
            np.int32(values.shape[0]),
            np.int32(values.shape[1]),
            np.float32(decay),
            np.float32(power),
        ),
    )
    return evidence


def cpro_continuity_features_cuda(
    continuity_map,
    windows: list[dict],
    *,
    threshold,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce GPU continuity evidence for CPU-provided PELT window indices."""
    if not windows:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty
    cp, _ndimage = _cupy_modules()
    values = cp.asarray(continuity_map, dtype=cp.float32)
    threshold_device = cp.asarray(threshold, dtype=cp.float64)
    if not bool(cp.isfinite(threshold_device) & (threshold_device > 0.0)):
        raise ValueError("CPRO threshold must be finite and positive")
    starts = cp.asarray([int(row["record_start"]) for row in windows], dtype=cp.int64)
    stops = cp.asarray([int(row["record_stop"]) for row in windows], dtype=cp.int64)
    if bool(cp.any(starts < 0) | cp.any(stops <= starts) | cp.any(stops > values.shape[1])):
        raise ValueError("continuity windows must be non-empty in-range intervals")
    activity = cp.max(values, axis=0)
    activity_prefix = cp.concatenate(
        (cp.zeros(1, dtype=cp.float64), cp.cumsum(activity, dtype=cp.float64))
    )
    durations = stops - starts
    continuity_mean = (
        (activity_prefix[stops] - activity_prefix[starts]) / durations / threshold_device
    )
    ridge_energy = cp.empty((values.shape[0], starts.size), dtype=cp.float64)
    kernel = _cached_window_ridge_sum_kernel()
    threads = 128
    work_items = int(values.shape[0]) * int(starts.size)
    kernel(
        ((work_items + threads - 1) // threads,),
        (threads,),
        (
            values,
            starts,
            stops,
            ridge_energy,
            np.int32(values.shape[0]),
            np.int32(values.shape[1]),
            np.int32(starts.size),
        ),
    )
    total = activity_prefix[stops] - activity_prefix[starts]
    ridge_lock = cp.max(ridge_energy, axis=0) / cp.maximum(total, MIN_POSITIVE)
    packed = cp.stack((continuity_mean, ridge_lock), axis=1).astype(cp.float32)
    host = cp.asnumpy(packed)
    return host[:, 0], host[:, 1]
