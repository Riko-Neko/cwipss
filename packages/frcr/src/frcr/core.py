"""CPU implementation of Frequency-Referenced Coherent Ridge (FRCR)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


MIN_POSITIVE = 1e-12


@dataclass(frozen=True)
class FRCRParameters:
    reference_channels: int = 8
    guard_channels: int = 0
    time_support_cycles: float = 8.0
    min_positive_support: float = 0.70
    score_floor: float = 0.20
    clip_log_ratio: float = 1.5
    top_k_periods: int = 3

    def validate(self) -> None:
        if self.reference_channels < 1:
            raise ValueError("reference_channels must be at least 1")
        if self.guard_channels < 0:
            raise ValueError("guard_channels must be non-negative")
        if self.time_support_cycles <= 0.0:
            raise ValueError("time_support_cycles must be positive")
        if not 0.0 <= self.min_positive_support < 1.0:
            raise ValueError("min_positive_support must be in [0, 1)")
        if self.score_floor < 0.0:
            raise ValueError("score_floor must be non-negative")
        if self.clip_log_ratio <= 0.0:
            raise ValueError("clip_log_ratio must be positive")
        if self.top_k_periods < 1:
            raise ValueError("top_k_periods must be at least 1")


@dataclass(frozen=True)
class FRCRChannelResult:
    score_map: np.ndarray
    activity: np.ndarray
    activity_z: np.ndarray


def frequency_halo_slice(
    target_start: int,
    target_stop: int,
    channel_count: int,
    params: FRCRParameters,
) -> slice:
    """Return a shifted halo containing enough references for each target."""
    params.validate()
    start, stop, total = int(target_start), int(target_stop), int(channel_count)
    if not 0 <= start < stop <= total:
        raise ValueError("invalid target channel range")
    required = params.reference_channels + 2 * params.guard_channels + 1
    if total < required:
        raise ValueError(f"FRCR requires at least {required} physical channels, received {total}")
    radius = params.guard_channels + params.reference_channels
    return slice(max(0, start - radius), min(total, stop + radius))


def reference_channel_indices(
    target: int,
    channel_count: int,
    params: FRCRParameters,
) -> np.ndarray:
    """Select the nearest non-guard physical channels deterministically."""
    params.validate()
    target, total = int(target), int(channel_count)
    if not 0 <= target < total:
        raise ValueError("target channel is outside the supplied power cube")
    indices = np.arange(total, dtype=np.int64)
    distance = np.abs(indices - target)
    eligible = indices[distance > params.guard_channels]
    if eligible.size < params.reference_channels:
        raise ValueError(
            f"FRCR target channel {target} has {eligible.size} eligible references; "
            f"{params.reference_channels} are required"
        )
    order = np.lexsort((eligible, np.abs(eligible - target)))
    return eligible[order[: params.reference_channels]]


def _clean_power(values: np.ndarray) -> np.ndarray:
    power = np.asarray(values, dtype=np.float32)
    return np.maximum(np.where(np.isfinite(power), power, 0.0), 0.0).astype(np.float32, copy=False)


def _robust_standardize(activity: np.ndarray) -> np.ndarray:
    values = np.asarray(activity, dtype=np.float32)
    center = float(np.nanmedian(values)) if values.size else 0.0
    mad = float(np.nanmedian(np.abs(values - center))) if values.size else 0.0
    scale = max(1.4826 * mad, 1.0)
    result = (values - center) / scale
    result[~np.isfinite(result)] = 0.0
    return result.astype(np.float32, copy=False)


def _coherent_score(
    signed: np.ndarray,
    periods: np.ndarray,
    params: FRCRParameters,
) -> np.ndarray:
    records = int(signed.shape[1])
    coherent = np.zeros_like(signed, dtype=np.float32)
    support_floor = float(params.min_positive_support)
    for row, period in enumerate(np.asarray(periods, dtype=np.float64)):
        width = max(3, int(round(params.time_support_cycles * max(float(period), 1.0))))
        width = min(width, max(3, records if records % 2 == 1 else records - 1))
        if width % 2 == 0:
            width = max(3, width - 1)
        filtered = ndimage.uniform_filter1d(signed[row], size=width, mode="constant", cval=0.0)
        support = ndimage.uniform_filter1d(
            (signed[row] > 0.0).astype(np.float32),
            size=width,
            mode="constant",
            cval=0.0,
        )
        weight = np.clip(
            (support - support_floor) / max(1e-6, 1.0 - support_floor),
            0.0,
            1.0,
        )
        filtered *= weight
        margin = min(records // 2, max(int(math.ceil(max(float(period), 1.0))), width // 2))
        if margin:
            filtered[:margin] = 0.0
            filtered[records - margin :] = 0.0
        coherent[row] = filtered
    coherent = np.maximum(coherent, 0.0)
    coherent[coherent < params.score_floor] = 0.0
    coherent[~np.isfinite(coherent)] = 0.0
    return coherent.astype(np.float32, copy=False)


def frcr_channel(
    power_cube: np.ndarray,
    periods: np.ndarray,
    target_channel: int,
    params: FRCRParameters | None = None,
) -> FRCRChannelResult:
    """Compute the FRCR score map and time activity for one target channel."""
    params = params or FRCRParameters()
    params.validate()
    power = _clean_power(power_cube)
    period_values = np.asarray(periods, dtype=np.float64)
    if power.ndim != 3 or power.shape[0] != period_values.size:
        raise ValueError("power_cube must have shape (periods, records, channels)")
    refs = reference_channel_indices(target_channel, power.shape[2], params)
    target = power[:, :, target_channel]
    reference = power[:, :, refs]
    positive = reference[reference > 0.0]
    reference_level = float(np.nanmedian(positive)) if positive.size else MIN_POSITIVE
    eps = max(reference_level * 1e-6, MIN_POSITIVE)
    target_log = np.log(target + eps)
    reference_log = np.log(reference + eps)
    target_log -= float(np.nanmedian(target_log))
    reference_log -= np.nanmedian(reference_log, axis=(0, 1), keepdims=True)
    background = np.nanmax(reference_log, axis=2)
    signed = np.clip(target_log - background, -params.clip_log_ratio, params.clip_log_ratio)
    signed[~np.isfinite(signed)] = 0.0
    score = _coherent_score(signed.astype(np.float32, copy=False), period_values, params)
    k = min(params.top_k_periods, score.shape[0])
    if k:
        activity = np.mean(np.partition(score, score.shape[0] - k, axis=0)[-k:, :], axis=0)
    else:
        activity = np.zeros(power.shape[1], dtype=np.float32)
    activity = activity.astype(np.float32, copy=False)
    return FRCRChannelResult(
        score_map=score,
        activity=activity,
        activity_z=_robust_standardize(activity),
    )
