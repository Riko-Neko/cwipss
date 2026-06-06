"""CUDA preprocessing for single-channel candidate detection."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import Any

import numpy as np

from .activity import valid_period_mask
from .cwt_cuda import _cupy
from .detection import (
    _detect_preprocessed_channel_periods,
    _timing_add,
    _timing_increment,
    resolve_channel_candidate_cap,
)
from .windows import pelt_mean_shift_batch


def _scalar_float(value) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _cupy_array_device_id(array) -> int | None:
    device = getattr(array, "device", None)
    if device is None:
        return None
    device_id = getattr(device, "id", None)
    if device_id is not None:
        return int(device_id)
    try:
        return int(device)
    except (TypeError, ValueError):
        return None


def _resolve_cuda_device(cp: Any, array, cuda_device: int | None) -> int:
    array_device = _cupy_array_device_id(array)
    if cuda_device is None:
        if array_device is not None:
            return array_device
        return int(cp.cuda.runtime.getDevice())
    device_id = int(cuda_device)
    if array_device is not None and array_device != device_id:
        raise ValueError(
            f"power_cube is on CUDA device {array_device}, but cuda_device={device_id} was requested"
        )
    return device_id


def _gpu_nanquantile(cp: Any, values, q: float, *, axis: int | None = None, keepdims: bool = False):
    nanquantile = getattr(cp, "nanquantile", None)
    if nanquantile is not None:
        return nanquantile(values, q, axis=axis, keepdims=keepdims)
    return cp.quantile(values, q, axis=axis, keepdims=keepdims)


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


def _gpu_low_fraction_noise_floor_batch(cp: Any, power, fraction: float = 0.20):
    if power.ndim != 3:
        raise ValueError("power must have shape (periods, records, channels)")
    channels = int(power.shape[2])
    finite_mask = cp.isfinite(power)
    if not bool(cp.all(finite_mask).item()):
        floors = [
            _gpu_low_fraction_noise_floor(cp, power[:, :, channel_idx], fraction=fraction)
            for channel_idx in range(channels)
        ]
        return cp.asarray(floors, dtype=cp.float32)

    flat = cp.transpose(power, (2, 0, 1)).reshape(channels, -1)
    finite_size = int(flat.shape[1])
    if finite_size == 0:
        return cp.ones(channels, dtype=cp.float32)
    fraction = min(max(float(fraction), 1.0 / finite_size), 1.0)
    k = max(1, int(ceil(fraction * finite_size)))
    low = cp.partition(flat, k - 1, axis=1)[:, :k]
    floors = cp.nanmean(low, axis=1)
    bad = ~cp.isfinite(floors) | (floors <= 0.0)
    if bool(cp.any(bad).item()):
        floor_values = cp.asnumpy(floors)
        for channel_idx in cp.asnumpy(cp.where(bad)[0]):
            floor_values[int(channel_idx)] = _gpu_low_fraction_noise_floor(
                cp,
                power[:, :, int(channel_idx)],
                fraction=fraction,
            )
        floors = cp.asarray(floor_values, dtype=cp.float32)
    return cp.maximum(floors, cp.asarray(1e-12, dtype=cp.float32))


def _gpu_relative_excess(cp: Any, power, noise_floor: float, eps_fraction: float = 1e-6):
    floor = max(float(noise_floor), 1e-12)
    eps = max(1e-12, abs(floor) * float(eps_fraction))
    excess = power / (floor + eps) - 1.0
    return cp.where(cp.isfinite(excess), excess, 0.0).astype(cp.float32, copy=False)


def _gpu_relative_excess_batch(cp: Any, power, noise_floor, eps_fraction: float = 1e-6):
    floor = cp.maximum(noise_floor.astype(cp.float32, copy=False), cp.asarray(1e-12, dtype=cp.float32))
    eps = cp.maximum(cp.asarray(1e-12, dtype=cp.float32), cp.abs(floor) * float(eps_fraction))
    denom = (floor + eps).reshape(1, 1, -1)
    excess = power / denom - 1.0
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
    baseline = _gpu_nanquantile(cp, excess, q_bg, axis=1, keepdims=True)
    centered = excess - baseline
    low_cut = _gpu_nanquantile(cp, excess, q_scale, axis=1, keepdims=True)
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


def _gpu_period_robust_zscore_batch(
    cp: Any,
    excess,
    *,
    baseline_quantile: float = 0.10,
    scale_quantile: float = 0.20,
):
    if excess.ndim != 3:
        raise ValueError("excess must have shape (periods, records, channels)")
    q_bg = min(max(float(baseline_quantile), 0.0), 0.45)
    q_scale = min(max(float(scale_quantile), q_bg + 1e-6), 0.50)
    baseline = _gpu_nanquantile(cp, excess, q_bg, axis=1, keepdims=True)
    centered = excess - baseline
    low_cut = _gpu_nanquantile(cp, excess, q_scale, axis=1, keepdims=True)
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


def _gpu_coherent_structure_map_batch(
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
    z = _gpu_period_robust_zscore_batch(
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


def _gpu_signed_trimmed_period_activity_batch(cp: Any, excess, trim_low: float = 0.05, trim_high: float = 0.95):
    if excess.ndim != 3:
        raise ValueError("excess must have shape (periods, records, channels)")
    lo = min(max(float(trim_low), 0.0), 0.49)
    hi = min(max(float(trim_high), lo + 1e-6), 1.0)
    period_count = int(excess.shape[0])
    if period_count == 0:
        return cp.zeros(excess.shape[1:], dtype=cp.float32)
    start = int(np.floor(lo * period_count))
    stop = int(np.ceil(hi * period_count))
    stop = min(max(start + 1, stop), period_count)
    sorted_values = cp.sort(excess, axis=0)
    activity = cp.nanmean(sorted_values[start:stop, :, :], axis=0)
    return cp.where(cp.isfinite(activity), activity, 0.0).astype(cp.float32, copy=False)


def _gpu_smooth_activity(cp: Any, uniform_filter1d, activity, smooth_records: int = 1):
    width = max(1, int(smooth_records))
    if width <= 1 or int(activity.size) == 0:
        return activity.astype(cp.float32, copy=False)
    return uniform_filter1d(activity, size=width, mode="nearest").astype(cp.float32, copy=False)


def _gpu_smooth_activity_batch(cp: Any, uniform_filter1d, activity, smooth_records: int = 1):
    width = max(1, int(smooth_records))
    if width <= 1 or int(activity.size) == 0:
        return activity.astype(cp.float32, copy=False)
    return uniform_filter1d(activity, size=width, axis=0, mode="nearest").astype(cp.float32, copy=False)


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


def _gpu_robust_standardize_batch(cp: Any, values):
    if values.ndim != 2:
        raise ValueError("values must have shape (records, channels)")
    finite = cp.isfinite(values)
    median = cp.nanmedian(cp.where(finite, values, cp.nan), axis=0, keepdims=True)
    centered = values - median
    finite_centered = cp.where(finite, centered, cp.nan)
    mad = cp.nanmedian(cp.abs(finite_centered), axis=0, keepdims=True)
    scale = 1.4826 * mad
    fallback = cp.nanstd(finite_centered, axis=0, keepdims=True)
    scale = cp.where(cp.isfinite(scale) & (scale > 1e-6), scale, fallback)
    scale = cp.where(cp.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    z = centered / scale
    z = cp.where(finite & cp.isfinite(z), z, 0.0)
    return z.astype(cp.float32, copy=False)


def _gpu_windowed_period_profile(cp: Any, structured, start: int, stop: int) -> np.ndarray:
    start = max(0, min(int(start), int(structured.shape[1])))
    stop = max(start + 1, min(int(stop), int(structured.shape[1])))
    duration = max(1, int(stop - start))
    profile = cp.nansum(structured[:, start:stop], axis=1) / np.sqrt(duration)
    profile = cp.where(cp.isfinite(profile), profile, 0.0).astype(cp.float32, copy=False)
    return cp.asnumpy(profile)


@dataclass
class PreparedCudaPeriodChunk:
    structured: Any
    activity: np.ndarray
    activity_z: np.ndarray
    noise_floor: np.ndarray
    valid_periods: np.ndarray
    freqs_mhz: np.ndarray
    channel_start: int
    record_start: int
    channel_cap: int
    cuda_device: int
    pelt_penalty: float
    pelt_min_size_records: int
    pelt_jump_records: int
    pelt_threads: int
    window_min_duration_records: int
    window_min_activity_mean: float
    window_min_activity_raw_mean: float
    window_merge_gap_records: int
    profile_min_prominence: float
    profile_max_peaks_per_window: int


def prepare_block_period_chunks_cuda_power(
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
    max_candidates_per_channel: int | str,
    max_candidates_per_record: float = 3.0 / 4096.0,
    pelt_jump_records: int = 1,
    pelt_threads: int = 1,
    cuda_structure_batch_channels: int | None = None,
    cuda_device: int | None = None,
    timing: dict[str, float] | None = None,
):
    cp = _cupy()
    device_id = _resolve_cuda_device(cp, power_cube, cuda_device)
    cp.cuda.Device(device_id).use()
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
    channel_cap = resolve_channel_candidate_cap(
        max_candidates_per_channel,
        max_candidates_per_record,
        int(power.shape[1]),
    )
    if cuda_structure_batch_channels is None:
        batch_channels = int(power.shape[2])
    else:
        batch_channels = min(max(1, int(cuda_structure_batch_channels)), int(power.shape[2]))

    for chunk_start in range(0, int(power.shape[2]), batch_channels):
        chunk_stop = min(chunk_start + batch_channels, int(power.shape[2]))
        stage_start = perf_counter() if timing is not None else 0.0
        valid_power = power[mask_gpu, :, chunk_start:chunk_stop]
        noise_floor_gpu = _gpu_low_fraction_noise_floor_batch(cp, valid_power, fraction=noise_floor_fraction)
        excess = _gpu_relative_excess_batch(cp, valid_power, noise_floor_gpu, eps_fraction=excess_eps_fraction)
        if timing is not None:
            cp.cuda.Stream.null.synchronize()
            _timing_add(timing, "floor_excess_seconds", perf_counter() - stage_start)

        stage_start = perf_counter() if timing is not None else 0.0
        structured = _gpu_coherent_structure_map_batch(
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
        activity_raw = _gpu_signed_trimmed_period_activity_batch(
            cp,
            structured,
            trim_low=activity_trim_low,
            trim_high=activity_trim_high,
        )
        activity = _gpu_smooth_activity_batch(
            cp,
            uniform_filter1d,
            activity_raw,
            smooth_records=activity_smooth_records,
        )
        activity_z = _gpu_robust_standardize_batch(cp, activity)
        activity_cpu_batch = np.ascontiguousarray(cp.asnumpy(activity).T)
        activity_z_cpu_batch = np.ascontiguousarray(cp.asnumpy(activity_z).T)
        noise_floor_cpu = np.ascontiguousarray(cp.asnumpy(noise_floor_gpu))
        if timing is not None:
            _timing_add(timing, "activity_seconds", perf_counter() - stage_start)

        prepared = PreparedCudaPeriodChunk(
            structured=structured,
            activity=activity_cpu_batch,
            activity_z=activity_z_cpu_batch,
            noise_floor=noise_floor_cpu,
            valid_periods=valid_periods,
            freqs_mhz=freqs[chunk_start:chunk_stop],
            channel_start=chunk_start,
            record_start=int(record_start),
            channel_cap=int(channel_cap or 0),
            cuda_device=device_id,
            pelt_penalty=float(pelt_penalty),
            pelt_min_size_records=int(pelt_min_size_records),
            pelt_jump_records=max(1, int(pelt_jump_records)),
            pelt_threads=max(1, int(pelt_threads)),
            window_min_duration_records=int(window_min_duration_records),
            window_min_activity_mean=float(window_min_activity_mean),
            window_min_activity_raw_mean=float(window_min_activity_raw_mean),
            window_merge_gap_records=int(window_merge_gap_records),
            profile_min_prominence=float(profile_min_prominence),
            profile_max_peaks_per_window=int(profile_max_peaks_per_window),
        )
        del valid_power, noise_floor_gpu, excess, activity_raw, activity, activity_z
        yield prepared


def run_prepared_cuda_pelt(prepared_chunks: list[PreparedCudaPeriodChunk]):
    if not prepared_chunks:
        return [], 0.0
    first = prepared_chunks[0]
    if len(prepared_chunks) == 1:
        activity_z = first.activity_z
    else:
        activity_z = np.ascontiguousarray(
            np.concatenate([prepared.activity_z for prepared in prepared_chunks], axis=0)
        )
    start = perf_counter()
    segments = pelt_mean_shift_batch(
        activity_z,
        penalty=first.pelt_penalty,
        min_size=first.pelt_min_size_records,
        jump=first.pelt_jump_records,
        threads=first.pelt_threads,
    )
    return segments, perf_counter() - start


def finalize_prepared_cuda_period_chunk(
    prepared: PreparedCudaPeriodChunk,
    segments_batch,
    *,
    timing: dict[str, float] | None = None,
) -> tuple[list[dict], list[dict]]:
    cp = _cupy()
    cp.cuda.Device(prepared.cuda_device).use()
    structured = prepared.structured
    candidates: list[dict] = []
    windows: list[dict] = []
    try:
        for local_channel_idx, segments in enumerate(segments_batch):
            channel_idx = prepared.channel_start + local_channel_idx

            def profile_getter(start: int, stop: int, local_index=local_channel_idx) -> np.ndarray:
                return _gpu_windowed_period_profile(cp, structured[:, :, local_index], start, stop)

            channel_candidates, channel_windows, _diagnostics = _detect_preprocessed_channel_periods(
                valid_periods=prepared.valid_periods,
                structured=None,
                activity=prepared.activity[local_channel_idx],
                activity_z=prepared.activity_z[local_channel_idx],
                noise_floor=float(prepared.noise_floor[local_channel_idx]),
                freq_mhz=float(prepared.freqs_mhz[local_channel_idx]),
                channel_idx=channel_idx,
                record_start=prepared.record_start,
                pelt_penalty=prepared.pelt_penalty,
                pelt_min_size_records=prepared.pelt_min_size_records,
                pelt_jump_records=prepared.pelt_jump_records,
                window_min_duration_records=prepared.window_min_duration_records,
                window_min_activity_mean=prepared.window_min_activity_mean,
                window_min_activity_raw_mean=prepared.window_min_activity_raw_mean,
                window_merge_gap_records=prepared.window_merge_gap_records,
                profile_min_prominence=prepared.profile_min_prominence,
                profile_max_peaks_per_window=prepared.profile_max_peaks_per_window,
                max_candidates_per_channel=prepared.channel_cap,
                segments=segments,
                timing=timing,
                profile_getter=profile_getter,
            )
            _timing_increment(timing, "channels", 1)
            candidates.extend(channel_candidates)
            windows.extend(channel_windows)
    finally:
        prepared.structured = None

    candidates.sort(key=lambda row: (row["integrated_score"], row["period_peak_prominence"]), reverse=True)
    return candidates, windows


def finalize_prepared_cuda_period_chunks(
    prepared_chunks: list[PreparedCudaPeriodChunk],
    segments_batch,
    *,
    timing: dict[str, float] | None = None,
) -> tuple[list[dict], list[dict]]:
    candidates: list[dict] = []
    windows: list[dict] = []
    offset = 0
    try:
        for prepared in prepared_chunks:
            channel_count = int(prepared.activity_z.shape[0])
            chunk_candidates, chunk_windows = finalize_prepared_cuda_period_chunk(
                prepared,
                segments_batch[offset:offset + channel_count],
                timing=timing,
            )
            candidates.extend(chunk_candidates)
            windows.extend(chunk_windows)
            offset += channel_count
    finally:
        for prepared in prepared_chunks:
            prepared.structured = None
    if offset != len(segments_batch):
        raise ValueError("PELT segment batch does not match prepared CUDA channel count")
    candidates.sort(key=lambda row: (row["integrated_score"], row["period_peak_prominence"]), reverse=True)
    return candidates, windows


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
    max_candidates_per_channel: int | str,
    max_candidates_per_record: float = 3.0 / 4096.0,
    pelt_jump_records: int = 1,
    pelt_threads: int = 1,
    cuda_structure_batch: bool = False,
    cuda_structure_batch_channels: int | None = None,
    cuda_device: int | None = None,
    timing: dict[str, float] | None = None,
) -> tuple[list[dict], list[dict]]:
    cp = _cupy()
    cp.cuda.Device(_resolve_cuda_device(cp, power_cube, cuda_device)).use()
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
    channel_cap = resolve_channel_candidate_cap(
        max_candidates_per_channel,
        max_candidates_per_record,
        int(power.shape[1]),
    )
    pelt_threads = max(1, int(pelt_threads))
    if cuda_structure_batch_channels is None:
        batch_channels = int(power.shape[2])
    else:
        batch_channels = min(max(1, int(cuda_structure_batch_channels)), int(power.shape[2]))

    candidates: list[dict] = []
    windows: list[dict] = []
    if cuda_structure_batch:
        prepared_chunks = list(
            prepare_block_period_chunks_cuda_power(
                power,
                period_values,
                freqs,
                record_start,
                candidate_period_min_records=candidate_period_min_records,
                candidate_period_max_records=candidate_period_max_records,
                noise_floor_fraction=noise_floor_fraction,
                excess_eps_fraction=excess_eps_fraction,
                structure_baseline_quantile=structure_baseline_quantile,
                structure_scale_quantile=structure_scale_quantile,
                structure_z_threshold=structure_z_threshold,
                structure_time_support_records=structure_time_support_records,
                structure_period_support_bins=structure_period_support_bins,
                structure_min_support_fraction=structure_min_support_fraction,
                activity_trim_low=activity_trim_low,
                activity_trim_high=activity_trim_high,
                activity_smooth_records=activity_smooth_records,
                pelt_penalty=pelt_penalty,
                pelt_min_size_records=pelt_min_size_records,
                pelt_jump_records=pelt_jump_records,
                pelt_threads=pelt_threads,
                cuda_structure_batch_channels=batch_channels,
                cuda_device=cuda_device,
                window_min_duration_records=window_min_duration_records,
                window_min_activity_mean=window_min_activity_mean,
                window_min_activity_raw_mean=window_min_activity_raw_mean,
                window_merge_gap_records=window_merge_gap_records,
                profile_min_prominence=profile_min_prominence,
                profile_max_peaks_per_window=profile_max_peaks_per_window,
                max_candidates_per_channel=max_candidates_per_channel,
                max_candidates_per_record=max_candidates_per_record,
                timing=timing,
            )
        )
        segments_batch, pelt_seconds = run_prepared_cuda_pelt(prepared_chunks)
        _timing_add(timing, "pelt_seconds", pelt_seconds)
        return finalize_prepared_cuda_period_chunks(prepared_chunks, segments_batch, timing=timing)

    preprocessed_channels: list[tuple[int, Any, np.ndarray, np.ndarray, float]] = []
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

        if pelt_threads > 1:
            preprocessed_channels.append(
                (int(channel_idx), structured, activity_cpu, activity_z_cpu, float(noise_floor))
            )
            del valid_power, excess, activity_raw, activity, activity_z
            continue

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
            pelt_jump_records=pelt_jump_records,
            window_min_duration_records=window_min_duration_records,
            window_min_activity_mean=window_min_activity_mean,
            window_min_activity_raw_mean=window_min_activity_raw_mean,
            window_merge_gap_records=window_merge_gap_records,
            profile_min_prominence=profile_min_prominence,
            profile_max_peaks_per_window=profile_max_peaks_per_window,
            max_candidates_per_channel=int(channel_cap or 0),
            timing=timing,
            profile_getter=profile_getter,
        )
        _timing_increment(timing, "channels", 1)
        if timing is not None:
            _timing_add(timing, "channel_total_seconds", perf_counter() - channel_start)
        candidates.extend(channel_candidates)
        windows.extend(channel_windows)
        del valid_power, excess, structured, activity_raw, activity, activity_z

    if pelt_threads > 1 and preprocessed_channels:
        stage_start = perf_counter() if timing is not None else 0.0
        activity_z_batch = np.stack([row[3] for row in preprocessed_channels], axis=0)
        segments_batch = pelt_mean_shift_batch(
            activity_z_batch,
            penalty=pelt_penalty,
            min_size=pelt_min_size_records,
            jump=pelt_jump_records,
            threads=pelt_threads,
        )
        if timing is not None:
            _timing_add(timing, "pelt_seconds", perf_counter() - stage_start)

        for (channel_idx, structured, activity_cpu, activity_z_cpu, noise_floor), segments in zip(
            preprocessed_channels,
            segments_batch,
            strict=True,
        ):

            def profile_getter(start: int, stop: int, structured_gpu=structured) -> np.ndarray:
                return _gpu_windowed_period_profile(cp, structured_gpu, start, stop)

            channel_candidates, channel_windows, _diagnostics = _detect_preprocessed_channel_periods(
                valid_periods=valid_periods,
                structured=None,
                activity=activity_cpu,
                activity_z=activity_z_cpu,
                noise_floor=noise_floor,
                freq_mhz=float(freqs[channel_idx]),
                channel_idx=channel_idx,
                record_start=record_start,
                pelt_penalty=pelt_penalty,
                pelt_min_size_records=pelt_min_size_records,
                pelt_jump_records=pelt_jump_records,
                window_min_duration_records=window_min_duration_records,
                window_min_activity_mean=window_min_activity_mean,
                window_min_activity_raw_mean=window_min_activity_raw_mean,
                window_merge_gap_records=window_merge_gap_records,
                profile_min_prominence=profile_min_prominence,
                profile_max_peaks_per_window=profile_max_peaks_per_window,
                max_candidates_per_channel=int(channel_cap or 0),
                segments=segments,
                timing=timing,
                profile_getter=profile_getter,
            )
            _timing_increment(timing, "channels", 1)
            candidates.extend(channel_candidates)
            windows.extend(channel_windows)
            del structured

    candidates.sort(key=lambda row: (row["integrated_score"], row["period_peak_prominence"]), reverse=True)
    return candidates, windows
