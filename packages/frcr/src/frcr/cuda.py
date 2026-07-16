"""CUDA implementation of Frequency-Referenced Coherent Ridge (FRCR)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .core import FRCRParameters, reference_channel_indices


@dataclass(frozen=True)
class FRCRCudaChannelResult:
    score_map: object
    activity: object
    activity_z: object


def _cupy_modules():
    try:
        import cupy as cp
        from cupyx.scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("FRCR CUDA requires CuPy with cupyx.scipy") from exc
    return cp, ndimage


def frcr_channel_cuda(
    power_cube,
    periods: np.ndarray,
    target_channel: int,
    params: FRCRParameters | None = None,
) -> FRCRCudaChannelResult:
    """Run every period-time operation on CUDA and return GPU arrays."""
    params = params or FRCRParameters()
    params.validate()
    cp, ndimage = _cupy_modules()
    power = cp.asarray(power_cube, dtype=cp.float32)
    period_values = np.asarray(periods, dtype=np.float64)
    if power.ndim != 3 or power.shape[0] != period_values.size:
        raise ValueError("power_cube must have shape (periods, records, channels)")
    power = cp.maximum(cp.where(cp.isfinite(power), power, 0.0), 0.0)
    refs = reference_channel_indices(target_channel, int(power.shape[2]), params)
    target = power[:, :, int(target_channel)]
    reference = power[:, :, cp.asarray(refs)]
    positive = reference[reference > 0.0]
    reference_level = cp.median(positive) if positive.size else cp.asarray(1e-12, dtype=cp.float32)
    eps = cp.maximum(reference_level * 1e-6, cp.asarray(1e-12, dtype=cp.float32))
    target_log = cp.log(target + eps)
    reference_log = cp.log(reference + eps)
    target_log -= cp.median(target_log)
    reference_log -= cp.median(reference_log, axis=(0, 1), keepdims=True)
    signed = cp.clip(
        target_log - cp.max(reference_log, axis=2),
        -params.clip_log_ratio,
        params.clip_log_ratio,
    )
    signed = cp.where(cp.isfinite(signed), signed, 0.0).astype(cp.float32, copy=False)

    records = int(power.shape[1])
    score = cp.zeros_like(signed, dtype=cp.float32)
    for row, period in enumerate(period_values):
        width = max(3, int(round(params.time_support_cycles * max(float(period), 1.0))))
        width = min(width, max(3, records if records % 2 == 1 else records - 1))
        if width % 2 == 0:
            width = max(3, width - 1)
        filtered = ndimage.uniform_filter1d(signed[row], size=width, mode="constant", cval=0.0)
        support = ndimage.uniform_filter1d(
            (signed[row] > 0.0).astype(cp.float32),
            size=width,
            mode="constant",
            cval=0.0,
        )
        weight = cp.clip(
            (support - params.min_positive_support) / max(1e-6, 1.0 - params.min_positive_support),
            0.0,
            1.0,
        )
        filtered *= weight
        margin = min(records // 2, max(int(math.ceil(max(float(period), 1.0))), width // 2))
        if margin:
            filtered[:margin] = 0.0
            filtered[records - margin :] = 0.0
        score[row] = filtered
    score = cp.maximum(score, 0.0)
    score = cp.where(score >= params.score_floor, score, 0.0).astype(cp.float32, copy=False)
    k = min(params.top_k_periods, int(score.shape[0]))
    if k:
        activity = cp.mean(cp.partition(score, int(score.shape[0]) - k, axis=0)[-k:, :], axis=0)
    else:
        activity = cp.zeros(records, dtype=cp.float32)
    center = cp.median(activity)
    scale = cp.maximum(1.4826 * cp.median(cp.abs(activity - center)), 1.0)
    activity_z = cp.where(cp.isfinite(activity), (activity - center) / scale, 0.0).astype(
        cp.float32,
        copy=False,
    )
    return FRCRCudaChannelResult(score_map=score, activity=activity, activity_z=activity_z)
