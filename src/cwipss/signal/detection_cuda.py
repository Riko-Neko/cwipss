"""CPRO candidate generation with a mandatory native-PELT CPU bridge."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .activity import robust_standardize
from .cpro import CPROParameters, cpro_period_mask
from .cpro_cuda import cpro_activity_cuda, difference_noise_std_cuda
from .cprf import CPRFParameters
from .cprf_cuda import cprf_normalization_threshold_cuda, evaluate_cprf_cuda
from .detection import (
    _timing_add,
    _timing_increment,
    activity_can_reach_window_mean,
    build_channel_candidates,
    pelt_windows_from_segments,
    resolve_channel_candidate_cap,
)
from .windows import PeltCancellation, Segment, pelt_mean_shift_batch


@dataclass
class PreparedCudaPeriodChannel:
    """One channel held across the GPU -> PELT -> GPU bridge."""

    output_channel: int
    target_channel: int
    power_map: object
    noise_gain_device: object
    noise_std_device: object
    cprf_normalization_threshold: object
    cprf_params: CPRFParameters
    shape_activity: np.ndarray
    activity_z: np.ndarray
    noise_std: float
    calibrated_threshold: float
    valid_periods: np.ndarray
    freqs_mhz: np.ndarray
    record_start: int
    channel_cap: int
    cuda_device: int
    pelt_penalty: float
    pelt_min_size_records: int
    pelt_jump_records: int
    pelt_threads: int
    window_min_duration_records: int
    window_min_activity_mean: float
    window_merge_gap_records: int


def _cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("CPRO CUDA requires CuPy") from exc
    return cp


def _invalid_noise_channel_record_cuda(values, *, channel: int, freq_mhz: float, cp) -> dict:
    finite_values = values[cp.isfinite(values)]
    finite_count = int(finite_values.size)
    if finite_count < 3:
        reason = "insufficient_finite"
        data_min = data_max = float("nan")
    else:
        data_min = float(cp.min(finite_values).item())
        data_max = float(cp.max(finite_values).item())
        if data_min == 0.0 and data_max == 0.0:
            reason = "all_zero"
        elif data_min == data_max:
            reason = "constant"
        else:
            reason = "invalid_sigma"
    return {
        "channel": int(channel),
        "freq_mhz": float(freq_mhz),
        "finite_records": finite_count,
        "data_min": data_min,
        "data_max": data_max,
        "reason": reason,
    }


def prepare_block_period_chunks_cuda_power(
    power_cube,
    raw_data,
    periods: np.ndarray,
    freqs_mhz: np.ndarray,
    noise_gain: np.ndarray,
    record_start: int,
    *,
    target_channel_start: int,
    target_channel_stop: int,
    candidate_period_min_records: float | None,
    candidate_period_max_records: float | None,
    cpro_threshold_snr: float,
    cpro_texture_quantile: float,
    cpro_period_center_bins: int,
    cpro_period_context_bins: int,
    cpro_min_period_contrast: float,
    cpro_support_records: int,
    cpro_min_occupancy: float,
    cpro_period_support_bins: int,
    cpro_window_support_records: int,
    cpro_min_window_occupancy: float,
    cpro_shape_power_softness: float,
    cpro_shape_contrast_softness: float,
    cpro_shape_occupancy_softness: float,
    cpro_shape_top_k: int,
    pelt_penalty: float,
    pelt_min_size_records: int,
    pelt_jump_records: int,
    pelt_threads: int,
    window_min_duration_records: int,
    window_min_activity_mean: float,
    window_merge_gap_records: int,
    cprf_params: CPRFParameters,
    max_candidates_per_channel: int | str,
    max_candidates_per_record: float = 3.0 / 4096.0,
    cuda_device: int | None = None,
    timing: dict[str, float] | None = None,
    invalid_channel_mask: np.ndarray | None = None,
    invalid_channels: list[dict] | None = None,
) -> list[PreparedCudaPeriodChannel]:
    """Run all 2D CPRO work on CUDA and return only 1D PELT inputs to host."""
    cp = _cupy()
    device = int(cuda_device or 0)
    with cp.cuda.Device(device):
        power = cp.asarray(power_cube, dtype=cp.float32)
        raw = cp.asarray(raw_data, dtype=cp.float32)
        period_values = np.asarray(periods, dtype=np.float64)
        freqs = np.asarray(freqs_mhz, dtype=np.float64)
        if power.ndim != 3 or power.shape[0] != period_values.size or power.shape[2] != freqs.size:
            raise ValueError("power_cube shape must match periods and freqs_mhz")
        if raw.shape != (power.shape[1], power.shape[2]):
            raise ValueError("raw_data must have shape (records, channels)")
        start, stop = int(target_channel_start), int(target_channel_stop)
        if not 0 <= start < stop <= power.shape[2]:
            raise ValueError("invalid target channel offsets")
        excluded = np.zeros(int(power.shape[2]), dtype=bool)
        if invalid_channel_mask is not None:
            excluded = np.asarray(invalid_channel_mask, dtype=bool)
            if excluded.shape != (int(power.shape[2]),):
                raise ValueError("invalid_channel_mask must match the block channel axis")
        mask = cpro_period_mask(
            period_values,
            candidate_period_min_records,
            candidate_period_max_records,
        )
        valid_power = power[cp.asarray(mask)]
        valid_periods = period_values[mask]
        gain = np.asarray(noise_gain, dtype=np.float32)
        if gain.shape != (valid_periods.size,):
            raise ValueError("noise_gain must match the candidate period domain")
        params = CPROParameters(
            threshold_snr=cpro_threshold_snr,
            texture_quantile=cpro_texture_quantile,
            period_center_bins=cpro_period_center_bins,
            period_context_bins=cpro_period_context_bins,
            min_period_contrast=cpro_min_period_contrast,
            support_records=cpro_support_records,
            min_occupancy=cpro_min_occupancy,
            period_support_bins=cpro_period_support_bins,
            window_support_records=cpro_window_support_records,
            min_window_occupancy=cpro_min_window_occupancy,
            shape_power_softness=cpro_shape_power_softness,
            shape_contrast_softness=cpro_shape_contrast_softness,
            shape_occupancy_softness=cpro_shape_occupancy_softness,
            shape_top_k=cpro_shape_top_k,
        )
        params.validate()
        cprf_params.validate()
        gain_device = cp.asarray(gain, dtype=cp.float32)
        cap = resolve_channel_candidate_cap(
            max_candidates_per_channel,
            max_candidates_per_record,
            int(power.shape[1]),
        )
        prepared: list[PreparedCudaPeriodChannel] = []
        for output_channel, target in enumerate(range(start, stop)):
            if excluded[target]:
                continue
            stage_start = perf_counter() if timing is not None else 0.0
            try:
                noise_std_gpu = difference_noise_std_cuda(raw[:, target])
            except ValueError:
                if invalid_channels is not None:
                    invalid_channels.append(
                        _invalid_noise_channel_record_cuda(
                            raw[:, target],
                            channel=output_channel,
                            freq_mhz=float(freqs[target]),
                            cp=cp,
                        )
                    )
                continue
            result = cpro_activity_cuda(
                valid_power[:, :, target],
                noise_std=noise_std_gpu,
                noise_gain=gain_device,
                params=params,
            )
            cprf_threshold = cprf_normalization_threshold_cuda(
                valid_power[:, :, target],
                noise_std=noise_std_gpu,
                noise_gain=gain_device,
                params=cprf_params,
            )
            # Only the 1D proposal axis crosses the boundary; CWT2D remains resident for CPRF.
            shape_activity = cp.asnumpy(result.shape_activity).astype(np.float32, copy=False)
            prepared.append(
                PreparedCudaPeriodChannel(
                    output_channel=output_channel,
                    target_channel=target,
                    power_map=valid_power[:, :, target],
                    noise_gain_device=gain_device,
                    noise_std_device=noise_std_gpu,
                    cprf_normalization_threshold=cprf_threshold,
                    cprf_params=cprf_params,
                    shape_activity=shape_activity,
                    activity_z=robust_standardize(shape_activity),
                    noise_std=float(noise_std_gpu.item()),
                    calibrated_threshold=float(result.threshold.item()),
                    valid_periods=valid_periods,
                    freqs_mhz=freqs,
                    record_start=int(record_start),
                    channel_cap=cap,
                    cuda_device=device,
                    pelt_penalty=float(pelt_penalty),
                    pelt_min_size_records=int(pelt_min_size_records),
                    pelt_jump_records=int(pelt_jump_records),
                    pelt_threads=int(pelt_threads),
                    window_min_duration_records=int(window_min_duration_records),
                    window_min_activity_mean=float(window_min_activity_mean),
                    window_merge_gap_records=int(window_merge_gap_records),
                )
            )
            _timing_add(timing, "cpro_seconds", perf_counter() - stage_start if timing is not None else 0.0)
            del result
        return prepared


def run_prepared_cuda_pelt(
    prepared: list[PreparedCudaPeriodChannel],
    cancellation: PeltCancellation | None = None,
    timing: dict[str, float] | None = None,
) -> tuple[list[list[Segment]], float]:
    """Run the required native C++ batch PELT outside the CUDA worker."""
    if not prepared:
        return [], 0.0
    start = perf_counter()
    segments_batch: list[list[Segment]] = [[] for _channel in prepared]
    native_indices = [
        index
        for index, channel in enumerate(prepared)
        if activity_can_reach_window_mean(channel.activity_z, channel.window_min_activity_mean)
    ]
    skipped_channels = len(prepared) - len(native_indices)
    _timing_increment(timing, "pelt_short_circuit_channels", skipped_channels)
    if not native_indices:
        if timing is not None:
            timing.update(
                {
                    "pelt_candidates_mean": 0.0,
                    "pelt_candidates_max": 0.0,
                    "pelt_candidate_observations": 0.0,
                    "pelt_constant_channels": 0.0,
                }
            )
        return segments_batch, perf_counter() - start

    reference = prepared[native_indices[0]]
    activity_z_batch = np.ascontiguousarray(
        np.stack([prepared[index].activity_z for index in native_indices], axis=0),
        dtype=np.float64,
    )
    native_diagnostics: dict[str, float] | None = {} if timing is not None else None
    native_segments = pelt_mean_shift_batch(
        activity_z_batch,
        penalty=reference.pelt_penalty,
        min_size=reference.pelt_min_size_records,
        jump=reference.pelt_jump_records,
        threads=reference.pelt_threads,
        cancellation=cancellation,
        diagnostics=native_diagnostics,
    )
    for index, segments in zip(native_indices, native_segments, strict=True):
        segments_batch[index] = segments
    if timing is not None and native_diagnostics is not None:
        timing.update(native_diagnostics)
    return segments_batch, perf_counter() - start


def finalize_prepared_cuda_period_chunks(
    prepared: list[PreparedCudaPeriodChannel],
    segments_batch: list[list[Segment]],
    *,
    timing: dict[str, float] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Consume PELT window indices and finish CPRF evaluation on CUDA."""
    if len(prepared) != len(segments_batch):
        raise ValueError("PELT segment batch must match prepared CUDA channels")
    if not prepared:
        return [], []
    cp = _cupy()
    candidates: list[dict] = []
    windows: list[dict] = []
    with cp.cuda.Device(prepared[0].cuda_device):
        for channel, segments in zip(prepared, segments_batch, strict=True):
            channel_windows_raw = pelt_windows_from_segments(
                channel.shape_activity,
                channel.activity_z,
                segments,
                penalty=channel.pelt_penalty,
                min_duration=channel.window_min_duration_records,
                min_mean=channel.window_min_activity_mean,
                merge_gap=channel.window_merge_gap_records,
            )

            def cprf_getter(local_start: int, local_stop: int):
                return evaluate_cprf_cuda(
                    channel.power_map[:, local_start:local_stop],
                    channel.valid_periods,
                    noise_std=channel.noise_std_device,
                    noise_gain=channel.noise_gain_device,
                    normalization_threshold=channel.cprf_normalization_threshold,
                    params=channel.cprf_params,
                )

            rows, channel_windows = build_channel_candidates(
                shape_activity=channel.shape_activity,
                windows=channel_windows_raw,
                noise_std=channel.noise_std,
                calibrated_threshold=channel.calibrated_threshold,
                freq_mhz=float(channel.freqs_mhz[channel.target_channel]),
                channel_idx=channel.output_channel,
                record_start=channel.record_start,
                max_candidates_per_channel=channel.channel_cap,
                timing=timing,
                cprf_getter=cprf_getter,
            )
            candidates.extend(rows)
            windows.extend(channel_windows)
            _timing_increment(timing, "segments", len(segments))
            _timing_increment(timing, "channels", 1)
            channel.power_map = None
            channel.noise_gain_device = None
            channel.noise_std_device = None
            channel.cprf_normalization_threshold = None
    candidates.sort(
        key=lambda row: row["score"],
        reverse=True,
    )
    return candidates, windows


def detect_block_periods_cuda_power(*args, timing: dict[str, float] | None = None, **kwargs):
    """Run the mandatory three-stage CUDA/CPU/CUDA detection bridge synchronously."""
    prepared = prepare_block_period_chunks_cuda_power(*args, timing=timing, **kwargs)
    segments_batch, pelt_seconds = run_prepared_cuda_pelt(prepared, timing=timing)
    _timing_add(timing, "pelt_seconds", pelt_seconds)
    return finalize_prepared_cuda_period_chunks(prepared, segments_batch, timing=timing)
