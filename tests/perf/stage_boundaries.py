"""Shared scientific boundaries for the three-stage performance ranks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cwipss.signal.activity import robust_standardize
from cwipss.signal.windows import (
    Segment,
    active_windows_from_segments,
    merge_close_windows,
    pelt_mean_shift,
    pelt_mean_shift_batch,
    require_native_pelt,
)


@dataclass(frozen=True)
class PELTWindowParameters:
    penalty: float = 16.0
    min_size_records: int = 384
    jump_records: int = 1
    min_duration_records: int = 384
    min_activity_mean: float = 0.05
    merge_gap_records: int = 256
    adapt_to_short_series: bool = True

    def validate(self) -> None:
        if self.penalty < 0.0:
            raise ValueError("PELT penalty must be non-negative")
        if self.min_size_records < 1 or self.jump_records < 1:
            raise ValueError("PELT record parameters must be positive")
        if self.min_duration_records < 1:
            raise ValueError("PELT minimum window duration must be positive")
        if self.merge_gap_records < 0:
            raise ValueError("PELT window merge gap must be non-negative")


@dataclass(frozen=True)
class PELTWindowResult:
    activity_z: np.ndarray
    segments: tuple[Segment, ...]
    windows: tuple[dict[str, float | int], ...]
    effective_min_size_records: int
    effective_min_duration_records: int


def pelt_parameters_from_config(config: Any) -> PELTWindowParameters:
    return PELTWindowParameters(
        penalty=float(config.pelt_penalty),
        min_size_records=int(config.pelt_min_size_records),
        jump_records=int(config.pelt_jump_records),
        min_duration_records=int(config.window_min_duration_records),
        min_activity_mean=float(config.window_min_activity_mean),
        merge_gap_records=int(config.window_merge_gap_records),
    )


def standardize_activity_for_pelt(
    activity: np.ndarray,
    *,
    absolute_calibrated: bool,
    native_absolute: bool = False,
) -> np.ndarray:
    """Map stage-1 activity to the same robust units used by production PELT."""
    values = np.asarray(activity, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("stage-1 activity must be a 1D array")
    if native_absolute:
        finite = np.isfinite(values)
        if not np.any(finite):
            return np.zeros_like(values, dtype=np.float32)
        center = float(np.nanmedian(values[finite]))
        centered = values - center
        mad = float(np.nanmedian(np.abs(centered[finite])))
        maximum = float(np.nanmax(np.abs(values[finite])))
        scale = max(
            1.4826 * mad,
            abs(center) * 1e-6,
            maximum * 1e-12,
            float(np.finfo(np.float32).tiny),
        )
        standardized = np.zeros_like(values, dtype=np.float32)
        standardized[finite] = centered[finite] / scale
        return standardized.astype(np.float32, copy=False)
    if absolute_calibrated and np.any(values < 0.0):
        raise ValueError("absolute-calibrated activity must be non-negative")
    return robust_standardize(values)


def segment_activity_with_pelt(
    activity: np.ndarray,
    parameters: PELTWindowParameters,
    *,
    activity_z: np.ndarray | None = None,
) -> PELTWindowResult:
    """Stage 2: map one 1D activity axis to windows using native PELT only."""
    parameters.validate()
    require_native_pelt()
    raw = np.asarray(activity, dtype=np.float32)
    if raw.ndim != 1:
        raise ValueError("stage-1 activity must be a 1D array")
    standardized = (
        robust_standardize(raw)
        if activity_z is None
        else np.asarray(activity_z, dtype=np.float32)
    )
    if standardized.shape != raw.shape:
        raise ValueError("activity_z must match stage-1 activity")
    records = int(raw.size)
    if parameters.adapt_to_short_series:
        short_limit = max(8, records // 8)
        min_size = min(int(parameters.min_size_records), short_limit)
        min_duration = min(int(parameters.min_duration_records), short_limit)
    else:
        min_size = int(parameters.min_size_records)
        min_duration = int(parameters.min_duration_records)
    segments = pelt_mean_shift(
        np.asarray(standardized, dtype=np.float64),
        penalty=float(parameters.penalty),
        min_size=min_size,
        jump=int(parameters.jump_records),
    )
    windows = merge_close_windows(
        active_windows_from_segments(
            segments,
            standardized,
            min_duration=min_duration,
            min_mean=float(parameters.min_activity_mean),
        ),
        max_gap=int(parameters.merge_gap_records),
    )
    return PELTWindowResult(
        activity_z=standardized,
        segments=tuple(segments),
        windows=tuple(windows),
        effective_min_size_records=min_size,
        effective_min_duration_records=min_duration,
    )


def segment_activity_batch_with_pelt(
    activities: np.ndarray,
    parameters: PELTWindowParameters,
    *,
    activities_z: np.ndarray,
    threads: int,
) -> tuple[PELTWindowResult, ...]:
    """Batch the same native PELT boundary across independent activity axes."""
    parameters.validate()
    require_native_pelt()
    raw = np.asarray(activities, dtype=np.float32)
    standardized = np.asarray(activities_z, dtype=np.float32)
    if raw.ndim != 2 or standardized.shape != raw.shape:
        raise ValueError("activities and activities_z must have shape (cases, records)")
    records = int(raw.shape[1])
    if parameters.adapt_to_short_series:
        short_limit = max(8, records // 8)
        min_size = min(int(parameters.min_size_records), short_limit)
        min_duration = min(int(parameters.min_duration_records), short_limit)
    else:
        min_size = int(parameters.min_size_records)
        min_duration = int(parameters.min_duration_records)
    segments_batch = pelt_mean_shift_batch(
        np.asarray(standardized, dtype=np.float64),
        penalty=float(parameters.penalty),
        min_size=min_size,
        jump=int(parameters.jump_records),
        threads=max(1, int(threads)),
    )
    results: list[PELTWindowResult] = []
    for row, row_z, segments in zip(raw, standardized, segments_batch, strict=True):
        windows = merge_close_windows(
            active_windows_from_segments(
                segments,
                row_z,
                min_duration=min_duration,
                min_mean=float(parameters.min_activity_mean),
            ),
            max_gap=int(parameters.merge_gap_records),
        )
        results.append(
            PELTWindowResult(
                activity_z=row_z,
                segments=tuple(segments),
                windows=tuple(windows),
                effective_min_size_records=min_size,
                effective_min_duration_records=min_duration,
            )
        )
    return tuple(results)


def stage3_windows(
    windows: tuple[dict[str, float | int], ...] | list[dict[str, float | int]],
    *,
    minimum_duration_records: int,
) -> tuple[dict[str, float | int], ...]:
    """Boundary-only gate allowed between PELT and period-axis compression."""
    minimum = max(1, int(minimum_duration_records))
    return tuple(
        dict(window)
        for window in windows
        if int(window["record_stop"]) - int(window["record_start"]) >= minimum
    )


__all__ = [
    "PELTWindowParameters",
    "PELTWindowResult",
    "pelt_parameters_from_config",
    "segment_activity_batch_with_pelt",
    "segment_activity_with_pelt",
    "standardize_activity_for_pelt",
    "stage3_windows",
]
