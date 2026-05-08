from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import ndimage as ndi


def robust_score_2d(image: np.ndarray, local_time: int = 513, local_freq: int = 9) -> np.ndarray:
    """Local robust S/N using median and MAD.

    The input is expected to be a log-power map with shape `(records, channels)`.
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


def summarize_components(
    score: np.ndarray,
    freqs_mhz: np.ndarray,
    record_start: int,
    level_number: int,
    threshold: float,
    min_pixels: int,
    max_components: int | None = None,
) -> list[dict]:
    mask = np.asarray(score >= float(threshold), dtype=bool)
    labels, count = label_components(mask)
    rows: list[dict] = []
    for component_id in range(1, count + 1):
        coords = np.argwhere(labels == component_id)
        area = int(coords.shape[0])
        if area < min_pixels:
            continue
        values = score[coords[:, 0], coords[:, 1]]
        peak_local = int(np.nanargmax(values))
        peak_y = int(coords[peak_local, 0])
        peak_x = int(coords[peak_local, 1])
        t0 = int(coords[:, 0].min())
        t1 = int(coords[:, 0].max())
        f0 = int(coords[:, 1].min())
        f1 = int(coords[:, 1].max())
        rows.append(
            {
                "swt_level": int(level_number),
                "component_id": int(component_id),
                "area_pixels": area,
                "record_start": int(record_start + t0),
                "record_stop": int(record_start + t1 + 1),
                "duration_records": int(t1 - t0 + 1),
                "freq_start_mhz": float(freqs_mhz[f0]),
                "freq_stop_mhz": float(freqs_mhz[f1]),
                "bandwidth_mhz": float(freqs_mhz[f1] - freqs_mhz[f0]) if f1 > f0 else 0.0,
                "peak_record": int(record_start + peak_y),
                "peak_freq_mhz": float(freqs_mhz[peak_x]),
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
