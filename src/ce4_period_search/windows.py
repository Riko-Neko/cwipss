from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    start: int
    stop: int
    cost: float
    mean: float

    @property
    def duration(self) -> int:
        return max(0, int(self.stop) - int(self.start))


def _segment_cost(prefix_sum: np.ndarray, prefix_sq: np.ndarray, start: int, stop: int) -> float:
    n = int(stop) - int(start)
    if n <= 0:
        return 0.0
    total = float(prefix_sum[stop] - prefix_sum[start])
    total_sq = float(prefix_sq[stop] - prefix_sq[start])
    return max(0.0, total_sq - total * total / n)


def _segment_mean(prefix_sum: np.ndarray, start: int, stop: int) -> float:
    n = int(stop) - int(start)
    if n <= 0:
        return 0.0
    return float((prefix_sum[stop] - prefix_sum[start]) / n)


def pelt_mean_shift(activity: np.ndarray, penalty: float = 8.0, min_size: int = 256) -> list[Segment]:
    """Segment one activity curve with a mean-shift PELT cost.

    The cost is within-segment squared error. This implementation uses the
    standard PELT pruning rule and is intended for record-scale curves.
    """
    y = np.asarray(activity, dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    n = int(y.size)
    min_size = max(1, int(min_size))
    penalty = max(0.0, float(penalty))
    if n == 0:
        return []
    if n <= min_size:
        prefix_sum = np.concatenate([[0.0], np.cumsum(y)])
        prefix_sq = np.concatenate([[0.0], np.cumsum(y * y)])
        return [Segment(0, n, _segment_cost(prefix_sum, prefix_sq, 0, n), _segment_mean(prefix_sum, 0, n))]

    prefix_sum = np.concatenate([[0.0], np.cumsum(y)])
    prefix_sq = np.concatenate([[0.0], np.cumsum(y * y)])
    best = np.full(n + 1, np.inf, dtype=np.float64)
    previous = np.full(n + 1, -1, dtype=np.int64)
    best[0] = -penalty
    candidates: list[int] = [0]

    for t in range(min_size, n + 1):
        valid = [s for s in candidates if t - s >= min_size]
        if not valid:
            candidates.append(t - min_size + 1)
            continue
        costs = np.array(
            [best[s] + _segment_cost(prefix_sum, prefix_sq, s, t) + penalty for s in valid],
            dtype=np.float64,
        )
        idx = int(np.argmin(costs))
        best[t] = float(costs[idx])
        previous[t] = int(valid[idx])
        cutoff = best[t] + penalty
        candidates = [
            s
            for s in candidates
            if t - s < min_size or best[s] + _segment_cost(prefix_sum, prefix_sq, s, t) <= cutoff
        ]
        candidates.append(t - min_size + 1)

    if previous[n] < 0:
        return [Segment(0, n, _segment_cost(prefix_sum, prefix_sq, 0, n), _segment_mean(prefix_sum, 0, n))]

    bounds = [n]
    cursor = n
    while cursor > 0 and previous[cursor] >= 0:
        cursor = int(previous[cursor])
        bounds.append(cursor)
    bounds = sorted(set(bounds))
    if bounds[0] != 0:
        bounds.insert(0, 0)

    segments: list[Segment] = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop <= start:
            continue
        segments.append(
            Segment(
                int(start),
                int(stop),
                _segment_cost(prefix_sum, prefix_sq, int(start), int(stop)),
                _segment_mean(prefix_sum, int(start), int(stop)),
            )
        )
    return segments


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
