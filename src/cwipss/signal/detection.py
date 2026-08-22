"""CPRO candidate generation and CPU orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from math import ceil
from time import perf_counter

import numpy as np

from .activity import robust_standardize
from .cpro import CPRO_DETECTOR, CPROParameters, cpro_activity, cpro_period_mask, difference_noise_std
from .cprf import CPRF_METHOD, CPRFParameters, CPRFResult, evaluate_cprf, normalize_cwt_power
from .windows import Segment, active_windows_from_segments, merge_close_windows, pelt_mean_shift


WINDOW_METHOD = f"{CPRO_DETECTOR}_pelt"
DETECTION_METHOD = f"{WINDOW_METHOD}_{CPRF_METHOD}"
CPRFGetter = Callable[[int, int], CPRFResult]


def _timing_add(timing: dict[str, float] | None, key: str, seconds: float) -> None:
    if timing is not None:
        timing[key] = float(timing.get(key, 0.0)) + float(seconds)


def _timing_increment(timing: dict[str, float] | None, key: str, value: int = 1) -> None:
    if timing is not None:
        timing[key] = float(timing.get(key, 0.0)) + float(value)


def resolve_channel_candidate_cap(
    max_candidates_per_channel: int | str,
    max_candidates_per_record: float,
    records: int,
) -> int:
    if isinstance(max_candidates_per_channel, str):
        value = max_candidates_per_channel.strip().lower()
        if value == "auto":
            return max(0, int(ceil(max(0.0, float(max_candidates_per_record)) * max(0, int(records)))))
        return max(0, int(value))
    return max(0, int(max_candidates_per_channel))


def invalid_noise_channel_record(
    values: np.ndarray,
    *,
    channel: int,
    freq_mhz: float,
) -> dict[str, float | int | str]:
    """Describe a channel that cannot support absolute noise calibration."""
    series = np.asarray(values, dtype=np.float64)
    finite_values = series[np.isfinite(series)]
    finite_count = int(finite_values.size)
    if finite_count < 3:
        reason = "insufficient_finite"
        data_min = data_max = float("nan")
    else:
        data_min = float(np.min(finite_values))
        data_max = float(np.max(finite_values))
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


def _peak_record(activity: np.ndarray, start: int, stop: int, record_start: int) -> int:
    values = np.asarray(activity, dtype=np.float32)
    start = max(0, min(int(start), values.size))
    stop = max(start + 1, min(int(stop), values.size))
    local = values[start:stop]
    return int(record_start + start + (int(np.nanargmax(local)) if local.size else 0))


def activity_can_reach_window_mean(activity_z: np.ndarray, min_mean: float) -> bool:
    """Return whether any segment can possibly satisfy the activity-mean gate."""
    values = np.asarray(activity_z, dtype=np.float32)
    if values.size == 0:
        return False
    effective = np.where(np.isfinite(values), values, 0.0)
    return float(np.max(effective)) >= float(min_mean)


def pelt_windows_from_activity(
    shape_activity: np.ndarray,
    *,
    penalty: float,
    min_size: int,
    jump: int,
    min_duration: int,
    min_mean: float,
    merge_gap: int,
) -> tuple[list[dict[str, float | int]], np.ndarray, int]:
    """Segment the sole CPRO shape axis with the required native PELT bridge."""
    shape = np.asarray(shape_activity, dtype=np.float32)
    if shape.ndim != 1:
        raise ValueError("CPRO shape activity must be a 1D array")
    activity_z = robust_standardize(shape)
    if not activity_can_reach_window_mean(activity_z, min_mean):
        return [], activity_z, 0
    segments = pelt_mean_shift(
        activity_z,
        penalty=penalty,
        min_size=min_size,
        jump=jump,
    )
    accepted = pelt_windows_from_segments(
        shape,
        activity_z,
        segments,
        penalty=penalty,
        min_duration=min_duration,
        min_mean=min_mean,
        merge_gap=merge_gap,
    )
    return accepted, activity_z, len(segments)


def pelt_windows_from_segments(
    shape_activity: np.ndarray,
    activity_z: np.ndarray,
    segments: list[Segment],
    *,
    penalty: float,
    min_duration: int,
    min_mean: float,
    merge_gap: int,
) -> list[dict[str, float | int]]:
    """Apply boundary-only gates to native PELT segments."""
    shape = np.asarray(shape_activity, dtype=np.float32)
    standardized = np.asarray(activity_z, dtype=np.float32)
    if shape.ndim != 1 or standardized.shape != shape.shape:
        raise ValueError("CPRO activity products must be matching 1D arrays")
    windows = active_windows_from_segments(
        segments,
        standardized,
        min_duration=min_duration,
        min_mean=min_mean,
    )
    windows = merge_close_windows(windows, max_gap=merge_gap)
    accepted: list[dict[str, float | int]] = []
    for window in windows:
        start = int(window["record_start"])
        stop = int(window["record_stop"])
        shape_window = shape[start:stop]
        if shape_window.size == 0:
            continue
        z_window = standardized[start:stop]
        accepted.append(
            {
                **window,
                "shape_activity_mean": float(np.mean(shape_window)),
                "shape_activity_max": float(np.max(shape_window)),
                "pelt_activity_mean": float(np.mean(z_window)),
                "pelt_activity_max": float(np.max(z_window)),
                "pelt_penalty": float(penalty),
            }
        )
    return accepted


def build_channel_candidates(
    *,
    shape_activity: np.ndarray,
    windows: Sequence[dict[str, float | int]],
    noise_std: float,
    calibrated_threshold: float,
    freq_mhz: float,
    channel_idx: int,
    record_start: int,
    max_candidates_per_channel: int,
    timing: dict[str, float] | None = None,
    cprf_getter: CPRFGetter,
) -> tuple[list[dict], list[dict]]:
    """Apply CPRF to accepted PELT windows and emit accepted candidates."""
    candidate_rows: list[dict] = []
    window_rows: list[dict] = []
    stage_start = perf_counter() if timing is not None else 0.0
    for window_index, window in enumerate(windows, start=1):
        local_start = int(window["record_start"])
        local_stop = int(window["record_stop"])
        window_id = f"ch{int(channel_idx):04d}_w{window_index:04d}"
        window_row = {
            "method": WINDOW_METHOD,
            "window_id": window_id,
            "channel": int(channel_idx),
            "freq_mhz": float(freq_mhz),
            "t0_rec": int(record_start + local_start),
            "t1_rec": int(record_start + local_stop),
            "dur_rec": int(local_stop - local_start),
            "noise_sigma": float(noise_std),
            "cpro_thr": float(calibrated_threshold),
            "shape_mean": float(window.get("shape_activity_mean", 0.0)),
            "shape_max": float(window.get("shape_activity_max", 0.0)),
            "pelt_z_mean": float(window.get("pelt_activity_mean", 0.0)),
            "pelt_z_max": float(window.get("pelt_activity_max", 0.0)),
            "pelt_pen": float(window.get("pelt_penalty", 0.0)),
        }
        result = cprf_getter(local_start, local_stop)
        window_row.update(
            {
                "accepted": int(result.accepted),
                "cprf_thr": result.normalization_threshold,
                "period_rec": result.peak_period_records,
                "p0_rec": result.period_start_records,
                "p1_rec": result.period_stop_records,
                "p_bins": result.width_bins,
                "ridge_peak": result.peak_strength,
                "ridge_int": result.integrated_strength,
                "band_conc": result.band_concentration,
                "band_persist": result.band_persistence,
                "local_contrast": result.local_contrast,
                "h2": result.harmonic_2_score,
                "h3": result.harmonic_3_score,
                "harm_n": result.harmonic_support_count,
                "core_score": result.base_score,
                "score": result.total_score,
            }
        )
        window_rows.append(window_row)
        if result.accepted:
            candidate_rows.append(
                {
                    "method": DETECTION_METHOD,
                    "window_id": window_id,
                    "channel": int(channel_idx),
                    "freq_mhz": float(freq_mhz),
                    "t0_rec": window_row["t0_rec"],
                    "t1_rec": window_row["t1_rec"],
                    "dur_rec": window_row["dur_rec"],
                    "t_peak_rec": _peak_record(
                        shape_activity,
                        local_start,
                        local_stop,
                        record_start,
                    ),
                    "period_rec": result.peak_period_records,
                    "p0_rec": result.period_start_records,
                    "p1_rec": result.period_stop_records,
                    "p_span_rec": abs(
                        result.period_stop_records - result.period_start_records
                    ),
                    "p_bins": result.width_bins,
                    "noise_sigma": window_row["noise_sigma"],
                    "cpro_thr": window_row["cpro_thr"],
                    "shape_mean": window_row["shape_mean"],
                    "shape_max": window_row["shape_max"],
                    "pelt_z_mean": window_row["pelt_z_mean"],
                    "pelt_z_max": window_row["pelt_z_max"],
                    "pelt_pen": window_row["pelt_pen"],
                    "cprf_thr": result.normalization_threshold,
                    "ridge_peak": result.peak_strength,
                    "ridge_int": result.integrated_strength,
                    "band_conc": result.band_concentration,
                    "band_persist": result.band_persistence,
                    "local_contrast": result.local_contrast,
                    "h2": result.harmonic_2_score,
                    "h3": result.harmonic_3_score,
                    "harm_n": result.harmonic_support_count,
                    "core_score": result.base_score,
                    "score": result.total_score,
                }
            )
    if timing is not None:
        _timing_add(timing, "cprf_seconds", perf_counter() - stage_start)
        _timing_increment(timing, "windows", len(window_rows))
    candidate_rows.sort(
        key=lambda row: row["score"],
        reverse=True,
    )
    return candidate_rows[: max(0, int(max_candidates_per_channel))], window_rows


def detect_block_periods(
    power_cube: np.ndarray,
    raw_data: np.ndarray,
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
) -> tuple[list[dict], list[dict]]:
    del pelt_threads  # CPU channels call the native single-series kernel directly.
    del cuda_device
    power = np.asarray(power_cube, dtype=np.float32)
    raw = np.asarray(raw_data, dtype=np.float32)
    period_values = np.asarray(periods, dtype=np.float64)
    freqs = np.asarray(freqs_mhz, dtype=np.float64)
    if power.ndim != 3 or power.shape[0] != period_values.size or power.shape[2] != freqs.size:
        raise ValueError("power_cube shape must match periods and freqs_mhz")
    if raw.shape != (power.shape[1], power.shape[2]):
        raise ValueError("raw_data must have shape (records, channels)")
    start, stop = int(target_channel_start), int(target_channel_stop)
    if not 0 <= start < stop <= power.shape[2]:
        raise ValueError("invalid target channel offsets")
    excluded = np.zeros(power.shape[2], dtype=bool)
    if invalid_channel_mask is not None:
        excluded = np.asarray(invalid_channel_mask, dtype=bool)
        if excluded.shape != (power.shape[2],):
            raise ValueError("invalid_channel_mask must match the block channel axis")
    mask = cpro_period_mask(
        period_values,
        candidate_period_min_records,
        candidate_period_max_records,
    )
    valid_power, valid_periods = power[mask], period_values[mask]
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
    cap = resolve_channel_candidate_cap(
        max_candidates_per_channel,
        max_candidates_per_record,
        power.shape[1],
    )
    candidates: list[dict] = []
    windows: list[dict] = []
    for output_channel, target in enumerate(range(start, stop)):
        if excluded[target]:
            continue
        stage_start = perf_counter() if timing is not None else 0.0
        try:
            noise_std = difference_noise_std(raw[:, target])
        except ValueError:
            if invalid_channels is not None:
                invalid_channels.append(
                    invalid_noise_channel_record(
                        raw[:, target],
                        channel=output_channel,
                        freq_mhz=float(freqs[target]),
                    )
                )
            continue
        result = cpro_activity(
            valid_power[:, :, target],
            noise_std=noise_std,
            noise_gain=gain,
            params=params,
        )
        _timing_add(timing, "cpro_seconds", perf_counter() - stage_start if timing is not None else 0.0)
        normalized_cwt, cprf_threshold = normalize_cwt_power(
            valid_power[:, :, target],
            noise_std=noise_std,
            noise_gain=gain,
            params=cprf_params,
        )
        stage_start = perf_counter() if timing is not None else 0.0
        pelt_windows, _activity_z, segment_count = pelt_windows_from_activity(
            result.shape_activity,
            penalty=pelt_penalty,
            min_size=pelt_min_size_records,
            jump=pelt_jump_records,
            min_duration=window_min_duration_records,
            min_mean=window_min_activity_mean,
            merge_gap=window_merge_gap_records,
        )
        _timing_add(timing, "pelt_seconds", perf_counter() - stage_start if timing is not None else 0.0)
        _timing_increment(timing, "segments", segment_count)
        def cprf_getter(local_start: int, local_stop: int) -> CPRFResult:
            return evaluate_cprf(
                normalized_cwt[:, local_start:local_stop],
                valid_periods,
                normalization_threshold=cprf_threshold,
                params=cprf_params,
            )

        rows, channel_windows = build_channel_candidates(
            shape_activity=result.shape_activity,
            windows=pelt_windows,
            noise_std=noise_std,
            calibrated_threshold=result.threshold,
            freq_mhz=float(freqs[target]),
            channel_idx=output_channel,
            record_start=record_start,
            max_candidates_per_channel=cap,
            timing=timing,
            cprf_getter=cprf_getter,
        )
        candidates.extend(rows)
        windows.extend(channel_windows)
        _timing_increment(timing, "channels", 1)
    candidates.sort(
        key=lambda row: row["score"],
        reverse=True,
    )
    return candidates, windows


def add_candidate_ids(candidates: Iterable[dict]) -> list[dict]:
    rows = sorted(
        candidates,
        key=lambda row: float(row.get("score", 0.0) or 0.0),
        reverse=True,
    )
    for idx, row in enumerate(rows, start=1):
        row["candidate_id"] = idx
    return rows
