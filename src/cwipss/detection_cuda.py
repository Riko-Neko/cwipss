from __future__ import annotations

from math import ceil
from time import perf_counter
from typing import Any

import numpy as np

from .activity import valid_period_mask
from .cwt_cuda import _cupy
from .detection import _detect_preprocessed_channel_periods, _timing_add, _timing_increment


def _scalar_float(value) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _gpu_low_fraction_noise_floor(cp: Any, power, fraction: float = 0.20) -> float:
    finite = power[cp.isfinite(power)]
    finite_size = int(finite.size)
    if finite_size == 0:
        return 1.0
    fraction = min(max(float(fraction), 1.0 / finite_size), 1.0)
    k = max(1, int(ceil(fraction * finite_size)))
    low = cp.partition(finite, k - 1)[:k]
    floor = _scalar_float(cp.nanmean(low))
    if not np.isfinite(floor) or floor <= 0.0:
        positive = finite[finite > 0.0]
        if int(positive.size) == 0:
            return 1.0
        floor = _scalar_float(cp.nanmedian(positive))
    return max(floor, 1e-12)


def _gpu_relative_excess(cp: Any, power, noise_floor: float, eps_fraction: float = 1e-6):
    floor = max(float(noise_floor), 1e-12)
    eps = max(1e-12, abs(floor) * float(eps_fraction))
    excess = power / (floor + eps) - 1.0
    return cp.where(cp.isfinite(excess), excess, 0.0).astype(cp.float32, copy=False)


def _gpu_period_robust_zscore(
    cp: Any,
    excess,
    *,
    baseline_quantile: float = 0.10,
    scale_quantile: float = 0.20,
):
    if excess.ndim != 2:
        raise ValueError("excess must have shape (periods, records)")
    q_bg = min(max(float(baseline_quantile), 0.0), 0.45)
    q_scale = min(max(float(scale_quantile), q_bg + 1e-6), 0.50)
    baseline = cp.nanquantile(excess, q_bg, axis=1, keepdims=True)
    centered = excess - baseline
    low_cut = cp.nanquantile(excess, q_scale, axis=1, keepdims=True)
    low_centered = cp.where(excess <= low_cut, centered, cp.nan)
    low_median = cp.nanmedian(low_centered, axis=1, keepdims=True)
    low_mad = cp.nanmedian(cp.abs(low_centered - low_median), axis=1, keepdims=True)
    scale = 1.4826 * low_mad
    low_std = cp.nanstd(low_centered, axis=1, keepdims=True)
    scale = cp.where(cp.isfinite(scale) & (scale > 1e-6), scale, low_std)
    fallback = cp.nanstd(centered, axis=1, keepdims=True)
    scale = cp.where(cp.isfinite(scale) & (scale > 1e-6), scale, fallback)
    scale = cp.where(cp.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    z = centered / scale
    return cp.where(cp.isfinite(z), z, 0.0).astype(cp.float32, copy=False)


def _gpu_coherent_structure_map(
    cp: Any,
    uniform_filter1d,
    excess,
    *,
    baseline_quantile: float = 0.10,
    scale_quantile: float = 0.20,
    z_threshold: float = 1.0,
    time_support_records: int = 64,
    period_support_bins: int = 3,
    min_support_fraction: float = 0.10,
):
    z = _gpu_period_robust_zscore(
        cp,
        excess,
        baseline_quantile=baseline_quantile,
        scale_quantile=scale_quantile,
    )
    positive = cp.maximum(z, 0.0)
    support = (z > float(z_threshold)).astype(cp.float32)
    time_width = max(1, int(time_support_records))
    period_width = max(1, int(period_support_bins))
    if time_width > 1:
        support = uniform_filter1d(support, size=time_width, axis=1, mode="nearest")
    if period_width > 1:
        support = uniform_filter1d(support, size=period_width, axis=0, mode="nearest")
    floor = min(max(float(min_support_fraction), 0.0), 0.95)
    if floor > 0.0:
        weight = cp.clip((support - floor) / max(1e-6, 1.0 - floor), 0.0, 1.0)
    else:
        weight = cp.clip(support, 0.0, 1.0)
    structured = positive * weight
    return cp.where(cp.isfinite(structured), structured, 0.0).astype(cp.float32, copy=False)


def _gpu_signed_trimmed_period_activity(cp: Any, excess, trim_low: float = 0.05, trim_high: float = 0.95):
    if excess.ndim != 2:
        raise ValueError("excess must have shape (periods, records)")
    lo = min(max(float(trim_low), 0.0), 0.49)
    hi = min(max(float(trim_high), lo + 1e-6), 1.0)
    period_count = int(excess.shape[0])
    if period_count == 0:
        return cp.zeros(excess.shape[1], dtype=cp.float32)
    start = int(np.floor(lo * period_count))
    stop = int(np.ceil(hi * period_count))
    stop = min(max(start + 1, stop), period_count)
    sorted_values = cp.sort(excess, axis=0)
    activity = cp.nanmean(sorted_values[start:stop, :], axis=0)
    return cp.where(cp.isfinite(activity), activity, 0.0).astype(cp.float32, copy=False)


def _gpu_smooth_activity(cp: Any, uniform_filter1d, activity, smooth_records: int = 1):
    width = max(1, int(smooth_records))
    if width <= 1 or int(activity.size) == 0:
        return activity.astype(cp.float32, copy=False)
    return uniform_filter1d(activity, size=width, mode="nearest").astype(cp.float32, copy=False)


def _gpu_robust_standardize(cp: Any, values):
    finite = cp.isfinite(values)
    if not bool(cp.any(finite).item()):
        return cp.zeros_like(values, dtype=cp.float32)
    median = cp.nanmedian(values[finite])
    centered = values - median
    mad = cp.nanmedian(cp.abs(centered[finite]))
    scale = _scalar_float(1.4826 * mad)
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = _scalar_float(cp.nanstd(centered[finite]))
    if not np.isfinite(scale) or scale <= 1e-6:
        return cp.zeros_like(values, dtype=cp.float32)
    z = centered / scale
    z = cp.where(finite, z, 0.0)
    return z.astype(cp.float32, copy=False)


def _gpu_windowed_period_profile(cp: Any, structured, start: int, stop: int) -> np.ndarray:
    start = max(0, min(int(start), int(structured.shape[1])))
    stop = max(start + 1, min(int(stop), int(structured.shape[1])))
    duration = max(1, int(stop - start))
    profile = cp.nansum(structured[:, start:stop], axis=1) / np.sqrt(duration)
    profile = cp.where(cp.isfinite(profile), profile, 0.0).astype(cp.float32, copy=False)
    return cp.asnumpy(profile)


def detect_block_periods_cuda_power(
    power_cube,
    periods: np.ndarray,
    freqs_mhz: np.ndarray,
    record_start: int,
    *,
    candidate_period_min_records: float | None,
    candidate_period_max_records: float | None,
    noise_floor_fraction: float,
    excess_eps_fraction: float,
    structure_baseline_quantile: float,
    structure_scale_quantile: float,
    structure_z_threshold: float,
    structure_time_support_records: int,
    structure_period_support_bins: int,
    structure_min_support_fraction: float,
    activity_trim_low: float,
    activity_trim_high: float,
    activity_smooth_records: int,
    pelt_penalty: float,
    pelt_min_size_records: int,
    window_min_duration_records: int,
    window_min_activity_mean: float,
    window_min_activity_raw_mean: float,
    window_merge_gap_records: int,
    profile_min_prominence: float,
    profile_max_peaks_per_window: int,
    max_candidates_per_channel: int,
    max_candidates: int | None = None,
    timing: dict[str, float] | None = None,
) -> tuple[list[dict], list[dict]]:
    cp = _cupy()
    from cupyx.scipy.ndimage import uniform_filter1d

    power = cp.asarray(power_cube, dtype=cp.float32)
    period_values = np.asarray(periods, dtype=np.float64)
    freqs = np.asarray(freqs_mhz, dtype=np.float64)
    if power.ndim != 3:
        raise ValueError("power_cube must have shape (periods, records, channels)")
    if power.shape[0] != period_values.size or power.shape[2] != freqs.size:
        raise ValueError("power_cube shape must match periods and freqs_mhz")
    mask = valid_period_mask(period_values, candidate_period_min_records, candidate_period_max_records)
    if not np.any(mask):
        raise ValueError("No CWT periods remain after candidate period filtering.")
    mask_gpu = cp.asarray(mask)
    valid_periods = period_values[mask]

    candidates: list[dict] = []
    windows: list[dict] = []
    for channel_idx in range(power.shape[2]):
        channel_start = perf_counter() if timing is not None else 0.0

        stage_start = perf_counter() if timing is not None else 0.0
        valid_power = power[mask_gpu, :, channel_idx]
        noise_floor = _gpu_low_fraction_noise_floor(cp, valid_power, fraction=noise_floor_fraction)
        excess = _gpu_relative_excess(cp, valid_power, noise_floor, eps_fraction=excess_eps_fraction)
        if timing is not None:
            cp.cuda.Stream.null.synchronize()
            _timing_add(timing, "floor_excess_seconds", perf_counter() - stage_start)

        stage_start = perf_counter() if timing is not None else 0.0
        structured = _gpu_coherent_structure_map(
            cp,
            uniform_filter1d,
            excess,
            baseline_quantile=structure_baseline_quantile,
            scale_quantile=structure_scale_quantile,
            z_threshold=structure_z_threshold,
            time_support_records=structure_time_support_records,
            period_support_bins=structure_period_support_bins,
            min_support_fraction=structure_min_support_fraction,
        )
        if timing is not None:
            cp.cuda.Stream.null.synchronize()
            _timing_add(timing, "structure_seconds", perf_counter() - stage_start)

        stage_start = perf_counter() if timing is not None else 0.0
        activity_raw = _gpu_signed_trimmed_period_activity(
            cp,
            structured,
            trim_low=activity_trim_low,
            trim_high=activity_trim_high,
        )
        activity = _gpu_smooth_activity(cp, uniform_filter1d, activity_raw, smooth_records=activity_smooth_records)
        activity_z = _gpu_robust_standardize(cp, activity)
        activity_cpu = cp.asnumpy(activity)
        activity_z_cpu = cp.asnumpy(activity_z)
        if timing is not None:
            _timing_add(timing, "activity_seconds", perf_counter() - stage_start)

        def profile_getter(start: int, stop: int, structured_gpu=structured) -> np.ndarray:
            return _gpu_windowed_period_profile(cp, structured_gpu, start, stop)

        channel_candidates, channel_windows, _diagnostics = _detect_preprocessed_channel_periods(
            valid_periods=valid_periods,
            structured=None,
            activity=activity_cpu,
            activity_z=activity_z_cpu,
            noise_floor=float(noise_floor),
            freq_mhz=float(freqs[channel_idx]),
            channel_idx=channel_idx,
            record_start=record_start,
            pelt_penalty=pelt_penalty,
            pelt_min_size_records=pelt_min_size_records,
            window_min_duration_records=window_min_duration_records,
            window_min_activity_mean=window_min_activity_mean,
            window_min_activity_raw_mean=window_min_activity_raw_mean,
            window_merge_gap_records=window_merge_gap_records,
            profile_min_prominence=profile_min_prominence,
            profile_max_peaks_per_window=profile_max_peaks_per_window,
            max_candidates_per_channel=max_candidates_per_channel,
            timing=timing,
            profile_getter=profile_getter,
        )
        _timing_increment(timing, "channels", 1)
        if timing is not None:
            _timing_add(timing, "channel_total_seconds", perf_counter() - channel_start)
        candidates.extend(channel_candidates)
        windows.extend(channel_windows)
        del valid_power, excess, structured, activity_raw, activity, activity_z

    candidates.sort(key=lambda row: (row["integrated_score"], row["period_peak_prominence"]), reverse=True)
    if max_candidates is not None:
        candidates = candidates[: max(0, int(max_candidates))]
    return candidates, windows
