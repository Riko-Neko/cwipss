from __future__ import annotations

from time import perf_counter
from typing import Iterable

import numpy as np

from .activity import (
    coherent_structure_map,
    crop_valid_periods,
    low_fraction_noise_floor,
    relative_excess,
    robust_standardize,
    signed_trimmed_period_activity,
    smooth_activity,
)
from .profile import find_period_profile_peaks, windowed_period_profile
from .windows import active_windows_from_segments, merge_close_windows, pelt_mean_shift


DETECTION_METHOD = "single_channel_lowfloor_pelt_profile"
WINDOW_METHOD = "single_channel_lowfloor_pelt"


def _timing_add(timing: dict[str, float] | None, key: str, seconds: float) -> None:
    if timing is not None:
        timing[key] = float(timing.get(key, 0.0)) + float(seconds)


def _timing_increment(timing: dict[str, float] | None, key: str, value: int = 1) -> None:
    if timing is not None:
        timing[key] = float(timing.get(key, 0.0)) + float(value)


def _peak_record(activity: np.ndarray, start: int, stop: int, record_start: int) -> int:
    values = np.asarray(activity, dtype=np.float32)
    start = max(0, min(int(start), values.size))
    stop = max(start + 1, min(int(stop), values.size))
    local = values[start:stop]
    if local.size == 0:
        return int(record_start + start)
    return int(record_start + start + int(np.nanargmax(local)))


def detect_channel_periods(
    power_channel: np.ndarray,
    periods: np.ndarray,
    *,
    freq_mhz: float,
    channel_idx: int,
    record_start: int,
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
    timing: dict[str, float] | None = None,
) -> tuple[list[dict], list[dict], dict[str, np.ndarray | float]]:
    """Detect period candidates from one channel's CWT power map.

    The detector uses a single low-fraction noise floor per channel over the
    trusted CWT period domain, suppresses isolated 2D texture with per-period
    low-quantile standardization and local time-period support, then detects
    PELT time windows from the compressed activity curve and searches
    period-profile peaks inside each window.
    """
    channel_start = perf_counter() if timing is not None else 0.0
    power = np.asarray(power_channel, dtype=np.float32)
    if power.ndim != 2:
        raise ValueError("power_channel must have shape (periods, records)")
    stage_start = perf_counter() if timing is not None else 0.0
    valid_power, valid_periods, _mask = crop_valid_periods(
        power,
        periods,
        candidate_period_min_records,
        candidate_period_max_records,
    )
    noise_floor = low_fraction_noise_floor(valid_power, fraction=noise_floor_fraction)
    excess = relative_excess(valid_power, noise_floor, eps_fraction=excess_eps_fraction)
    if timing is not None:
        _timing_add(timing, "floor_excess_seconds", perf_counter() - stage_start)

    stage_start = perf_counter() if timing is not None else 0.0
    structured = coherent_structure_map(
        excess,
        baseline_quantile=structure_baseline_quantile,
        scale_quantile=structure_scale_quantile,
        z_threshold=structure_z_threshold,
        time_support_records=structure_time_support_records,
        period_support_bins=structure_period_support_bins,
        min_support_fraction=structure_min_support_fraction,
    )
    if timing is not None:
        _timing_add(timing, "structure_seconds", perf_counter() - stage_start)

    stage_start = perf_counter() if timing is not None else 0.0
    activity_raw = signed_trimmed_period_activity(
        structured,
        trim_low=activity_trim_low,
        trim_high=activity_trim_high,
    )
    activity = smooth_activity(activity_raw, smooth_records=activity_smooth_records)
    activity_z = robust_standardize(activity)
    if timing is not None:
        _timing_add(timing, "activity_seconds", perf_counter() - stage_start)

    stage_start = perf_counter() if timing is not None else 0.0
    segments = pelt_mean_shift(activity_z, penalty=pelt_penalty, min_size=pelt_min_size_records)
    windows = active_windows_from_segments(
        segments,
        activity_z,
        min_duration=window_min_duration_records,
        min_mean=window_min_activity_mean,
    )
    windows = merge_close_windows(windows, max_gap=window_merge_gap_records)
    if timing is not None:
        _timing_add(timing, "pelt_seconds", perf_counter() - stage_start)
        _timing_increment(timing, "segments", len(segments))
        _timing_increment(timing, "windows_before_raw_floor", len(windows))

    window_rows: list[dict] = []
    candidate_rows: list[dict] = []
    raw_threshold = max(0.0, float(window_min_activity_raw_mean))
    stage_start = perf_counter() if timing is not None else 0.0
    for window_index, window in enumerate(windows, start=1):
        local_start = int(window["record_start"])
        local_stop = int(window["record_stop"])
        duration = int(local_stop - local_start)
        peak_record = _peak_record(activity_z, local_start, local_stop, record_start)
        window_id = f"ch{int(channel_idx):04d}_w{window_index:04d}"
        activity_window = activity_z[local_start:local_stop]
        raw_activity_window = activity[local_start:local_stop]
        raw_activity_mean = float(np.nanmean(raw_activity_window)) if raw_activity_window.size else 0.0
        raw_activity_max = float(np.nanmax(raw_activity_window)) if raw_activity_window.size else 0.0
        if raw_activity_mean < raw_threshold:
            continue
        window_row = {
            "detection_method": WINDOW_METHOD,
            "window_id": window_id,
            "channel_index": int(channel_idx),
            "freq_mhz": float(freq_mhz),
            "record_start": int(record_start + local_start),
            "record_stop": int(record_start + local_stop),
            "duration_records": duration,
            "activity_mean": float(np.nanmean(activity_window)) if activity_window.size else 0.0,
            "activity_max": float(np.nanmax(activity_window)) if activity_window.size else 0.0,
            "activity_raw_mean": raw_activity_mean,
            "activity_raw_max": raw_activity_max,
            "noise_floor": float(noise_floor),
            "pelt_penalty": float(pelt_penalty),
            "pelt_cost": float(window.get("pelt_cost", 0.0)),
        }
        window_rows.append(window_row)

        profile = windowed_period_profile(structured, local_start, local_stop)
        peaks = find_period_profile_peaks(
            profile,
            valid_periods,
            min_prominence=profile_min_prominence,
            max_peaks=profile_max_peaks_per_window,
        )
        for peak in peaks:
            candidate_rows.append(
                {
                    "detection_method": DETECTION_METHOD,
                    "window_id": window_id,
                    "channel_index": int(channel_idx),
                    "region_pixels": 0,
                    "record_start": int(record_start + local_start),
                    "record_stop": int(record_start + local_stop),
                    "duration_records": duration,
                    "period_start_records": peak["period_start_records"],
                    "period_stop_records": peak["period_stop_records"],
                    "period_width_records": peak["period_width_records"],
                    "period_width_bins": peak["period_width_bins"],
                    "peak_period_records": peak["peak_period_records"],
                    "freq_start_mhz": float(freq_mhz),
                    "freq_stop_mhz": float(freq_mhz),
                    "bandwidth_mhz": 0.0,
                    "peak_record": peak_record,
                    "peak_freq_mhz": float(freq_mhz),
                    "peak_score": float(peak["profile_score"]),
                    "mean_score": float(window_row["activity_mean"]),
                    "integrated_score": float(peak["profile_score"]),
                    "activity_mean": float(window_row["activity_mean"]),
                    "activity_max": float(window_row["activity_max"]),
                    "activity_raw_mean": float(window_row["activity_raw_mean"]),
                    "activity_raw_max": float(window_row["activity_raw_max"]),
                    "noise_floor": float(noise_floor),
                    "period_peak_prominence": float(peak["period_peak_prominence"]),
                }
            )
    if timing is not None:
        _timing_add(timing, "profile_seconds", perf_counter() - stage_start)
        _timing_increment(timing, "windows_after_raw_floor", len(window_rows))

    stage_start = perf_counter() if timing is not None else 0.0
    candidate_rows.sort(key=lambda row: (row["integrated_score"], row["period_peak_prominence"]), reverse=True)
    max_rows = max(1, int(max_candidates_per_channel))
    diagnostics: dict[str, np.ndarray | float] = {
        "valid_periods": valid_periods,
        "excess": excess,
        "structured": structured,
        "activity": activity_z,
        "noise_floor": float(noise_floor),
    }
    if timing is not None:
        _timing_add(timing, "candidate_sort_seconds", perf_counter() - stage_start)
        _timing_increment(timing, "channel_candidate_rows_before_cap", len(candidate_rows))
        _timing_add(timing, "channel_total_seconds", perf_counter() - channel_start)
    return candidate_rows[:max_rows], window_rows, diagnostics


def detect_block_periods(
    power_cube: np.ndarray,
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
    power = np.asarray(power_cube, dtype=np.float32)
    period_values = np.asarray(periods, dtype=np.float64)
    freqs = np.asarray(freqs_mhz, dtype=np.float64)
    if power.ndim != 3:
        raise ValueError("power_cube must have shape (periods, records, channels)")
    if power.shape[0] != period_values.size or power.shape[2] != freqs.size:
        raise ValueError("power_cube shape must match periods and freqs_mhz")

    candidates: list[dict] = []
    windows: list[dict] = []
    for channel_idx in range(power.shape[2]):
        channel_candidates, channel_windows, _diagnostics = detect_channel_periods(
            power[:, :, channel_idx],
            period_values,
            freq_mhz=float(freqs[channel_idx]),
            channel_idx=channel_idx,
            record_start=record_start,
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
            window_min_duration_records=window_min_duration_records,
            window_min_activity_mean=window_min_activity_mean,
            window_min_activity_raw_mean=window_min_activity_raw_mean,
            window_merge_gap_records=window_merge_gap_records,
            profile_min_prominence=profile_min_prominence,
            profile_max_peaks_per_window=profile_max_peaks_per_window,
            max_candidates_per_channel=max_candidates_per_channel,
            timing=timing,
        )
        _timing_increment(timing, "channels", 1)
        candidates.extend(channel_candidates)
        windows.extend(channel_windows)

    candidates.sort(key=lambda row: (row["integrated_score"], row["period_peak_prominence"]), reverse=True)
    if max_candidates is not None:
        candidates = candidates[: max(0, int(max_candidates))]
    return candidates, windows


def add_candidate_ids(candidates: Iterable[dict]) -> list[dict]:
    rows = sorted(
        candidates,
        key=lambda row: float(row.get("integrated_score", row.get("peak_score", 0.0)) or 0.0),
        reverse=True,
    )
    for idx, row in enumerate(rows, start=1):
        row["candidate_id"] = idx
    return rows
