from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import ndimage as ndi


def robust_score_2d(image: np.ndarray, local_time: int = 9, local_freq: int = 9) -> np.ndarray:
    """Local robust S/N using median and MAD.

    The input is expected to be a log-power map with shape `(periods, channels)`.
    A local baseline is important because large bright areas should not become
    candidates solely due to absolute intensity.
    """
    values = np.asarray(image, dtype=np.float32)
    local_time = max(3, int(local_time) | 1)
    local_freq = max(3, int(local_freq) | 1)
    median = ndi.median_filter(values, size=(local_time, local_freq), mode="nearest")
    abs_dev = np.abs(values - median)
    mad = ndi.median_filter(abs_dev, size=(local_time, local_freq), mode="nearest")
    scale = 1.4826 * mad
    score = (values - median) / np.where(scale > 1e-6, scale, 1.0)
    return np.asarray(score, dtype=np.float32)


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, count = ndi.label(mask, structure=structure)
    return labels.astype(np.int32, copy=False), int(count)


def summarize_period_components(
    score: np.ndarray,
    power_cube: np.ndarray,
    periods: np.ndarray,
    freqs_mhz: np.ndarray,
    record_start: int,
    record_stop: int,
    threshold: float,
    min_pixels: int,
    max_components: int | None = None,
) -> list[dict]:
    """Summarize connected components on a `(period, channel)` score map."""
    mask = np.asarray(score >= float(threshold), dtype=bool)
    labels, count = label_components(mask)
    component_slices = ndi.find_objects(labels)
    period_values = np.asarray(periods, dtype=np.float64)
    rows: list[dict] = []
    for component_id, component_slice in enumerate(component_slices, start=1):
        if component_slice is None:
            continue
        label_view = labels[component_slice]
        local_period_idx, local_channel_idx = np.nonzero(label_view == component_id)
        area = int(local_period_idx.size)
        if area < min_pixels:
            continue
        period_offset = int(component_slice[0].start or 0)
        channel_offset = int(component_slice[1].start or 0)
        period_idx = local_period_idx + period_offset
        channel_idx = local_channel_idx + channel_offset
        values = score[period_idx, channel_idx]
        peak_local = int(np.nanargmax(values))
        peak_period_idx = int(period_idx[peak_local])
        peak_channel_idx = int(channel_idx[peak_local])
        period0 = int(period_idx.min())
        period1 = int(period_idx.max())
        freq0 = int(channel_idx.min())
        freq1 = int(channel_idx.max())
        time_power = power_cube[peak_period_idx, :, peak_channel_idx]
        peak_record = int(record_start + int(np.nanargmax(time_power)))
        period_start = float(period_values[period0])
        period_stop = float(period_values[period1])
        peak_period = float(period_values[peak_period_idx])
        rows.append(
            {
                "component_id": int(component_id),
                "area_pixels": area,
                "record_start": int(record_start),
                "record_stop": int(record_stop),
                "duration_records": int(record_stop - record_start),
                "period_start_records": period_start,
                "period_stop_records": period_stop,
                "period_width_records": float(abs(period_stop - period_start)),
                "peak_period_records": peak_period,
                "freq_start_mhz": float(freqs_mhz[freq0]),
                "freq_stop_mhz": float(freqs_mhz[freq1]),
                "bandwidth_mhz": float(freqs_mhz[freq1] - freqs_mhz[freq0]) if freq1 > freq0 else 0.0,
                "peak_record": peak_record,
                "peak_freq_mhz": float(freqs_mhz[peak_channel_idx]),
                "peak_score": float(values[peak_local]),
                "mean_score": float(np.nanmean(values)),
            }
        )
    rows.sort(key=lambda row: row["peak_score"], reverse=True)
    if max_components is not None:
        rows = rows[: max(0, int(max_components))]
    return rows


def add_candidate_ids(candidates: Iterable[dict]) -> list[dict]:
    rows = sorted(candidates, key=lambda row: row["peak_score"], reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["candidate_id"] = idx
    return rows
