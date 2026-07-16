"""Strict single-channel absolute occupancy windows for scientific ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, log2, sqrt
from typing import Any

import numpy as np
from scipy import ndimage


MIN_POSITIVE = float(np.finfo(np.float32).tiny)


@dataclass(frozen=True)
class PersistentOccupancyParameters:
    name: str
    threshold_snr: float
    support_records: int
    min_occupancy: float
    period_support_bins: int = 1
    min_duration_records: int = 128
    max_gap_records: int = 64
    texture_quantile: float = 0.0
    period_center_bins: int = 1
    period_context_bins: int = 1
    min_period_contrast: float = 0.0
    window_support_records: int = 1
    min_window_occupancy: float = 0.0

    def validate(self) -> None:
        if self.threshold_snr <= 0.0:
            raise ValueError("threshold_snr must be positive")
        if self.support_records < 1:
            raise ValueError("support_records must be positive")
        if not 0.0 < self.min_occupancy <= 1.0:
            raise ValueError("min_occupancy must be in (0, 1]")
        if self.period_support_bins < 1:
            raise ValueError("period_support_bins must be positive")
        if self.min_duration_records < 1:
            raise ValueError("min_duration_records must be positive")
        if self.max_gap_records < 0:
            raise ValueError("max_gap_records must be non-negative")
        if not 0.0 <= self.texture_quantile < 1.0:
            raise ValueError("texture_quantile must be in [0, 1)")
        if self.period_center_bins < 1:
            raise ValueError("period_center_bins must be positive")
        if self.period_context_bins < self.period_center_bins:
            raise ValueError("period_context_bins must not be narrower than period_center_bins")
        if self.min_period_contrast < 0.0:
            raise ValueError("min_period_contrast must be non-negative")
        if self.window_support_records < 1:
            raise ValueError("window_support_records must be positive")
        if not 0.0 <= self.min_window_occupancy <= 1.0:
            raise ValueError("min_window_occupancy must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PersistentOccupancyResult:
    activity: np.ndarray
    score_map: np.ndarray
    occupancy_map: np.ndarray
    ridge_mask: np.ndarray
    ridge_time_mask: np.ndarray
    window_occupancy: np.ndarray
    active_mask: np.ndarray
    windows: tuple[dict[str, float | int], ...]


def persistent_occupancy_catalog() -> tuple[PersistentOccupancyParameters, ...]:
    """Return the focused CPRO scientific-screening grid."""
    candidates: list[PersistentOccupancyParameters] = []
    window_consensus = ((257, 0.50), (385, 0.40), (385, 0.35), (385, 0.30))
    for threshold in (16.0, 32.0):
        for quantile in (0.775, 0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.9375, 0.95):
            for contrast in (1.50,):
                for occupancy in (0.60, 0.65, 0.70):
                    for window_support, window_occupancy in window_consensus:
                        for period_bins in (3, 5):
                            duration = 96
                            quantile_code = int(round(1000 * quantile))
                            contrast_code = int(round(100 * contrast))
                            candidates.append(
                                PersistentOccupancyParameters(
                                    name=(
                                        f"cpro_e{int(threshold):02d}_q{quantile_code:03d}_"
                                        f"r{contrast_code:03d}_o{int(round(100 * occupancy)):02d}_"
                                        f"b{period_bins}_w{window_support:03d}_"
                                        f"v{int(round(100 * window_occupancy)):02d}_d{duration:03d}"
                                    ),
                                    threshold_snr=threshold,
                                    support_records=65,
                                    min_occupancy=occupancy,
                                    period_support_bins=period_bins,
                                    min_duration_records=duration,
                                    max_gap_records=64,
                                    texture_quantile=quantile,
                                    period_center_bins=3,
                                    period_context_bins=15,
                                    min_period_contrast=contrast,
                                    window_support_records=window_support,
                                    min_window_occupancy=window_occupancy,
                                )
                            )
    return tuple(candidates)


def difference_noise_std(values: np.ndarray) -> float:
    """Estimate white-noise sigma from first differences without a data fallback."""
    series = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(series)
    if series.ndim != 1 or np.count_nonzero(finite) < 3:
        raise ValueError("noise calibration requires a finite 1D series")
    differences = np.diff(series[finite])
    center = float(np.median(differences))
    mad = float(np.median(np.abs(differences - center)))
    sigma = 1.4826 * mad / sqrt(2.0)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("first-difference noise sigma must be finite and positive")
    return sigma


def impulse_cwt_noise_gain(
    periods: np.ndarray,
    *,
    wavelet: str,
    method: str = "fft",
) -> np.ndarray:
    """Return deterministic per-period CWT power gain for unit white noise."""
    from cwipss.signal.cwt import cwt_power_cube

    period_values = np.asarray(periods, dtype=np.float64)
    if period_values.ndim != 1 or period_values.size == 0 or np.any(period_values <= 0.0):
        raise ValueError("periods must be a positive 1D array")
    required = max(4096, int(32.0 * float(np.max(period_values))))
    records = 1 << int(ceil(log2(required)))
    impulse = np.zeros((records, 1), dtype=np.float32)
    impulse[records // 2, 0] = 1.0
    power = cwt_power_cube(
        impulse,
        period_values,
        wavelet=wavelet,
        normalize_channels=False,
        method=method,
        backend="cpu",
    )[:, :, 0]
    gain = np.sum(power, axis=1, dtype=np.float64)
    if not np.all(np.isfinite(gain)) or np.any(gain <= 0.0):
        raise ValueError("CWT impulse gains must be finite and positive")
    return gain.astype(np.float32)


def _edge_corrected_time_mean(values: np.ndarray, width: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    records = int(matrix.shape[1])
    size = max(1, min(int(width), records))
    if size <= 1:
        return matrix.astype(np.float32, copy=False)
    summed = ndimage.uniform_filter1d(matrix, size=size, axis=1, mode="constant", cval=0.0) * float(size)
    counts = ndimage.uniform_filter1d(
        np.ones(records, dtype=np.float32), size=size, mode="constant", cval=0.0
    ) * float(size)
    return (summed / np.maximum(counts[None, :], 1.0)).astype(np.float32, copy=False)


def _contiguous_period_support(mask: np.ndarray, bins: int) -> np.ndarray:
    width = max(1, min(int(bins), int(mask.shape[0])))
    if width <= 1:
        return mask.astype(bool, copy=False)
    count = ndimage.uniform_filter1d(mask.astype(np.float32), size=width, axis=0, mode="constant")
    return count >= (1.0 - 0.5 / float(width))


def _period_ridge_contrast(
    calibrated_power: np.ndarray,
    *,
    center_bins: int,
    context_bins: int,
) -> np.ndarray:
    """Contrast a narrow period band against its non-overlapping local sideband."""
    center_width = max(1, min(int(center_bins), int(calibrated_power.shape[0])))
    context_width = max(center_width, min(int(context_bins), int(calibrated_power.shape[0])))
    center = ndimage.uniform_filter1d(
        calibrated_power, size=center_width, axis=0, mode="nearest"
    )
    if context_width == center_width:
        return np.ones_like(center, dtype=np.float32)
    context = ndimage.uniform_filter1d(
        calibrated_power, size=context_width, axis=0, mode="nearest"
    )
    side = (
        float(context_width) * context - float(center_width) * center
    ) / float(context_width - center_width)
    return (center / np.maximum(side, MIN_POSITIVE)).astype(np.float32, copy=False)


def _runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1 or values.size == 0:
        return []
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    edges = np.concatenate(([0], changes, [values.size]))
    return [(int(start), int(stop), bool(values[start])) for start, stop in zip(edges[:-1], edges[1:])]


def regularize_time_mask(mask: np.ndarray, *, max_gap: int, min_duration: int) -> np.ndarray:
    """Fill bounded internal gaps, then reject short active runs without smoothing boundaries."""
    values = np.asarray(mask, dtype=bool).copy()
    for start, stop, active in _runs(values):
        if not active and start > 0 and stop < values.size and stop - start <= int(max_gap):
            values[start:stop] = True
    for start, stop, active in _runs(values):
        if active and stop - start < int(min_duration):
            values[start:stop] = False
    return values


def mask_windows(mask: np.ndarray, activity: np.ndarray) -> tuple[dict[str, float | int], ...]:
    values = np.asarray(activity, dtype=np.float32)
    windows: list[dict[str, float | int]] = []
    for start, stop, active in _runs(mask):
        if not active:
            continue
        local = values[start:stop]
        windows.append(
            {
                "record_start": start,
                "record_stop": stop,
                "duration_records": stop - start,
                "activity_mean": float(np.mean(local)) if local.size else 0.0,
                "activity_max": float(np.max(local)) if local.size else 0.0,
            }
        )
    return tuple(windows)


def persistent_occupancy_windows(
    power: np.ndarray,
    *,
    noise_std: float,
    noise_gain: np.ndarray,
    params: PersistentOccupancyParameters,
) -> PersistentOccupancyResult:
    """Generate connected time windows from one absolute CWT map only."""
    params.validate()
    values = np.asarray(power, dtype=np.float32)
    gain = np.asarray(noise_gain, dtype=np.float32)
    if values.ndim != 2 or gain.shape != (values.shape[0],):
        raise ValueError("power and noise_gain must match on the period axis")
    if not np.isfinite(noise_std) or noise_std <= 0.0:
        raise ValueError("noise_std must be finite and positive")
    values = np.maximum(np.where(np.isfinite(values), values, 0.0), 0.0)
    denominator = np.maximum(float(noise_std) ** 2 * gain[:, None], MIN_POSITIVE)
    calibrated = values / denominator
    threshold = float(params.threshold_snr)
    if params.texture_quantile > 0.0:
        threshold = max(threshold, float(np.quantile(calibrated, params.texture_quantile)))
    period_contrast = _period_ridge_contrast(
        calibrated,
        center_bins=params.period_center_bins,
        context_bins=params.period_context_bins,
    )
    exceedance = calibrated >= threshold
    if params.min_period_contrast > 0.0:
        exceedance &= period_contrast >= float(params.min_period_contrast)
    occupancy = _edge_corrected_time_mean(exceedance.astype(np.float32), params.support_records)
    persistent = occupancy >= float(params.min_occupancy)
    persistent = _contiguous_period_support(persistent, params.period_support_bins)

    occupied_power = _edge_corrected_time_mean(values * exceedance, params.support_records)
    bright_mean = occupied_power / np.maximum(occupancy, 1.0 / float(max(1, params.support_records)))
    ridge_time_mask = np.any(persistent, axis=0)
    window_ridge_mask = persistent
    window_occupancy = persistent.astype(np.float32)
    if params.window_support_records > 1 and params.min_window_occupancy > 0.0:
        window_occupancy = _edge_corrected_time_mean(
            persistent.astype(np.float32), params.window_support_records
        )
        window_ridge_mask = window_occupancy >= float(params.min_window_occupancy)
    time_mask = np.any(window_ridge_mask, axis=0)
    time_mask = regularize_time_mask(
        time_mask,
        max_gap=params.max_gap_records,
        min_duration=params.min_duration_records,
    )
    score_map = np.where(window_ridge_mask, bright_mean, 0.0).astype(np.float32, copy=False)
    activity = np.max(score_map, axis=0).astype(np.float32, copy=False)
    activity = np.where(time_mask, activity, 0.0).astype(np.float32, copy=False)
    score_map[:, ~time_mask] = 0.0
    return PersistentOccupancyResult(
        activity=activity,
        score_map=score_map,
        occupancy_map=occupancy.astype(np.float32, copy=False),
        ridge_mask=persistent,
        ridge_time_mask=ridge_time_mask,
        window_occupancy=np.max(window_occupancy, axis=0).astype(np.float32, copy=False),
        active_mask=time_mask,
        windows=mask_windows(time_mask, activity),
    )
