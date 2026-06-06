"""Native PELT time-window segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .. import _pelt_ext
except ImportError as exc:
    _pelt_ext = None
    _pelt_import_error = exc
else:
    _pelt_import_error = None


@dataclass(frozen=True)
class Segment:
    start: int
    stop: int
    cost: float
    mean: float

    @property
    def duration(self) -> int:
        return max(0, int(self.stop) - int(self.start))


def native_pelt_available() -> bool:
    return _pelt_ext is not None


def require_native_pelt() -> None:
    if _pelt_ext is not None:
        return
    message = (
        "The native cwipss._pelt_ext extension is required; Python PELT fallback "
        "is intentionally unsupported. Build/install the project with "
        "`python -m pip install -e .` using a C++17 compiler and CMake."
    )
    raise RuntimeError(message) from _pelt_import_error


def pelt_mean_shift(
    activity: np.ndarray,
    penalty: float = 16.0,
    min_size: int = 384,
    jump: int = 1,
) -> list[Segment]:
    require_native_pelt()
    rows = _pelt_ext.pelt_mean_shift(activity, penalty=float(penalty), min_size=int(min_size), jump=int(jump))
    return [Segment(int(start), int(stop), float(cost), float(mean)) for start, stop, cost, mean in rows]


def pelt_mean_shift_batch(
    activity: np.ndarray,
    penalty: float = 16.0,
    min_size: int = 384,
    jump: int = 1,
    threads: int = 1,
) -> list[list[Segment]]:
    values = np.asarray(activity, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("activity must have shape (channels, records)")
    threads = max(1, int(threads))
    require_native_pelt()
    batch_rows = _pelt_ext.pelt_mean_shift_batch(
        values,
        penalty=float(penalty),
        min_size=int(min_size),
        jump=int(jump),
        threads=threads,
    )
    return [
        [Segment(int(start), int(stop), float(cost), float(mean)) for start, stop, cost, mean in rows]
        for rows in batch_rows
    ]


def active_windows_from_segments(
    segments: list[Segment],
    activity: np.ndarray,
    *,
    min_duration: int,
    min_mean: float,
) -> list[dict]:
    values = np.asarray(activity, dtype=np.float32)
    rows: list[dict] = []
    for segment in segments:
        if segment.duration < int(min_duration) or segment.mean < float(min_mean):
            continue
        window = values[segment.start:segment.stop]
        rows.append(
            {
                "record_start": int(segment.start),
                "record_stop": int(segment.stop),
                "duration_records": int(segment.duration),
                "activity_mean": float(np.nanmean(window)) if window.size else 0.0,
                "activity_max": float(np.nanmax(window)) if window.size else 0.0,
                "pelt_cost": float(segment.cost),
            }
        )
    return rows


def merge_close_windows(windows: list[dict], max_gap: int = 0) -> list[dict]:
    if not windows:
        return []
    max_gap = max(0, int(max_gap))
    ordered = sorted(windows, key=lambda row: (int(row["record_start"]), int(row["record_stop"])))
    merged: list[dict] = [dict(ordered[0])]
    for row in ordered[1:]:
        last = merged[-1]
        if int(row["record_start"]) - int(last["record_stop"]) <= max_gap:
            last["record_stop"] = max(int(last["record_stop"]), int(row["record_stop"]))
            last["duration_records"] = int(last["record_stop"]) - int(last["record_start"])
            last["activity_mean"] = max(float(last.get("activity_mean", 0.0)), float(row.get("activity_mean", 0.0)))
            last["activity_max"] = max(float(last.get("activity_max", 0.0)), float(row.get("activity_max", 0.0)))
            last["pelt_cost"] = float(last.get("pelt_cost", 0.0)) + float(row.get("pelt_cost", 0.0))
        else:
            merged.append(dict(row))
    return merged
