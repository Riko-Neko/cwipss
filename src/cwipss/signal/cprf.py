"""Concentrated Periodic Ridge Filter (CPRF)."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy import ndimage


CPRF_METHOD = "concentrated_periodic_ridge_filter"
MIN_POSITIVE = float(np.finfo(np.float32).tiny)


@dataclass(frozen=True)
class CPRFParameters:
    """Frozen scientific parameters of the production CPRF method."""

    threshold_snr: float = 32.0
    texture_quantile: float = 0.9375
    smooth_bins: int = 3
    peak_band_fraction: float = 0.50
    min_width_bins: int = 3
    min_peak_strength: float = 1.25
    min_integrated_strength: float = 0.0
    min_band_persistence: float = 0.40
    min_band_concentration: float = 0.50
    min_local_contrast: float = 1.20
    harmonic_weight: float = 0.20
    harmonic_min_relative: float = 0.12
    harmonic_window_scale: float = 1.25
    max_peak_hypotheses: int = 8

    def validate(self) -> None:
        if self.threshold_snr <= 0.0:
            raise ValueError("cprf_threshold_snr must be positive")
        if not 0.0 <= self.texture_quantile < 1.0:
            raise ValueError("cprf_texture_quantile must be in [0, 1)")
        if self.smooth_bins < 1 or self.min_width_bins < 1:
            raise ValueError("CPRF period-bin widths must be positive")
        if not 0.0 < self.peak_band_fraction < 1.0:
            raise ValueError("cprf_peak_band_fraction must be in (0, 1)")
        if self.min_peak_strength < 0.0 or self.min_integrated_strength < 0.0:
            raise ValueError("CPRF strength thresholds must be non-negative")
        if not 0.0 <= self.min_band_persistence <= 1.0:
            raise ValueError("cprf_min_band_persistence must be in [0, 1]")
        if not 0.0 <= self.min_band_concentration <= 1.0:
            raise ValueError("cprf_min_band_concentration must be in [0, 1]")
        if self.min_local_contrast < 0.0:
            raise ValueError("cprf_min_local_contrast must be non-negative")
        if self.harmonic_weight < 0.0 or self.harmonic_min_relative < 0.0:
            raise ValueError("CPRF harmonic parameters must be non-negative")
        if self.harmonic_window_scale <= 0.0 or self.max_peak_hypotheses < 1:
            raise ValueError("CPRF harmonic scale and peak count must be positive")


@dataclass(frozen=True)
class CPRFResult:
    accepted: bool
    peak_period_records: float
    period_start_records: float
    period_stop_records: float
    peak_index: int
    band_start_index: int
    band_stop_index: int
    width_bins: int
    peak_strength: float
    integrated_strength: float
    band_concentration: float
    band_persistence: float
    local_contrast: float
    harmonic_2_score: float
    harmonic_3_score: float
    harmonic_support_count: int
    base_score: float
    total_score: float
    normalization_threshold: float
    profile: np.ndarray | None = None


def normalize_cwt_power(
    power: np.ndarray,
    *,
    noise_std: float,
    noise_gain: np.ndarray,
    params: CPRFParameters,
) -> tuple[np.ndarray, float]:
    """Normalize unmasked absolute CWT power into CPRF threshold units."""
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
    return (calibrated / threshold).astype(np.float32, copy=False), threshold


def _reduce_time(values: np.ndarray, params: CPRFParameters) -> tuple[np.ndarray, np.ndarray]:
    occupied = values >= 1.0
    occupied_values = np.where(occupied, values, 0.0)
    occupancy = np.mean(occupied, axis=1, dtype=np.float64).astype(np.float32)
    occupied_mean = np.sum(occupied_values, axis=1, dtype=np.float64) / np.maximum(
        np.count_nonzero(occupied, axis=1), 1
    )
    profile = np.asarray(occupied_mean * np.sqrt(occupancy), dtype=np.float32)
    if params.smooth_bins > 1:
        profile = ndimage.uniform_filter1d(
            profile,
            size=min(int(params.smooth_bins), profile.size),
            mode="nearest",
        ).astype(np.float32, copy=False)
    return profile, occupancy


def _peak_indices(profile: np.ndarray, maximum: int) -> np.ndarray:
    if profile.size == 1:
        return np.array([0], dtype=np.int64)
    peaks = np.flatnonzero(
        np.concatenate(
            (
                [profile[0] >= profile[1]],
                (profile[1:-1] > profile[:-2]) & (profile[1:-1] >= profile[2:]),
                [profile[-1] > profile[-2]],
            )
        )
    )
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(profile))], dtype=np.int64)
    order = np.argsort(profile[peaks], kind="stable")[::-1]
    return peaks[order[: max(1, int(maximum))]]


def _peak_band(profile: np.ndarray, peak: int, fraction: float) -> tuple[int, int]:
    positive = profile[profile > 0.0]
    baseline = float(np.median(positive)) if positive.size else 0.0
    threshold = baseline + float(fraction) * max(0.0, float(profile[peak]) - baseline)
    start = peak
    stop = peak + 1
    while start > 0 and profile[start - 1] >= threshold:
        start -= 1
    while stop < profile.size and profile[stop] >= threshold:
        stop += 1
    return start, stop


def _local_background(profile: np.ndarray, start: int, stop: int) -> float:
    width = max(1, stop - start)
    lo = max(0, start - 2 * width)
    hi = min(profile.size, stop + 2 * width)
    side = np.concatenate((profile[lo:start], profile[stop:hi]))
    if side.size == 0:
        side = np.concatenate((profile[:start], profile[stop:]))
    return float(np.median(side)) if side.size else 0.0


def _harmonic_score(
    profile: np.ndarray,
    periods: np.ndarray,
    *,
    peak_period: float,
    band_start: int,
    band_stop: int,
    order: int,
    window_scale: float,
    main_peak: float,
) -> float:
    frequencies = 1.0 / periods
    center_frequency = float(order) / peak_period
    band_frequencies = frequencies[band_start:band_stop]
    main_half_width = 0.5 * abs(float(np.max(band_frequencies)) - float(np.min(band_frequencies)))
    if frequencies.size > 1:
        nearest = int(np.argmin(np.abs(frequencies - center_frequency)))
        neighbor = min(frequencies.size - 1, nearest + 1) if nearest == 0 else nearest - 1
        grid_half_width = 0.5 * abs(float(frequencies[nearest] - frequencies[neighbor]))
    else:
        grid_half_width = 0.0
    half_width = max(float(order) * main_half_width * window_scale, grid_half_width)
    mask = np.abs(frequencies - center_frequency) <= half_width
    if not np.any(mask):
        return 0.0
    return float(np.clip(np.max(profile[mask]) / max(main_peak, MIN_POSITIVE), 0.0, 1.0))


def _candidate_metrics(
    profile: np.ndarray,
    occupancy: np.ndarray,
    periods: np.ndarray,
    peak: int,
    params: CPRFParameters,
) -> dict[str, float | int]:
    start, stop = _peak_band(profile, peak, params.peak_band_fraction)
    band = profile[start:stop]
    baseline = _local_background(profile, start, stop)
    width = max(1, stop - start)
    excess = np.maximum(band - baseline, 0.0)
    peak_strength = max(0.0, float(profile[peak]) - baseline)
    integrated = float(np.sum(excess, dtype=np.float64) / sqrt(width))
    concentration = float(
        np.sum(band, dtype=np.float64) / max(np.sum(profile, dtype=np.float64), MIN_POSITIVE)
    )
    persistence = float(
        np.sum(occupancy[start:stop] * np.maximum(band, 0.0), dtype=np.float64)
        / max(np.sum(np.maximum(band, 0.0), dtype=np.float64), MIN_POSITIVE)
    )
    local_contrast = float((np.mean(band, dtype=np.float64) - baseline) / max(abs(baseline), 1.0))
    harmonic_2 = _harmonic_score(
        profile,
        periods,
        peak_period=float(periods[peak]),
        band_start=start,
        band_stop=stop,
        order=2,
        window_scale=params.harmonic_window_scale,
        main_peak=float(profile[peak]),
    )
    harmonic_3 = _harmonic_score(
        profile,
        periods,
        peak_period=float(periods[peak]),
        band_start=start,
        band_stop=stop,
        order=3,
        window_scale=params.harmonic_window_scale,
        main_peak=float(profile[peak]),
    )
    base_score = (
        max(peak_strength, MIN_POSITIVE) ** 0.40
        * max(integrated, MIN_POSITIVE) ** 0.30
        * max(concentration, MIN_POSITIVE) ** 0.15
        * max(persistence, MIN_POSITIVE) ** 0.15
    )
    total_score = base_score * (1.0 + params.harmonic_weight * 0.5 * (harmonic_2 + harmonic_3))
    return {
        "peak": peak,
        "start": start,
        "stop": stop,
        "width": width,
        "peak_strength": peak_strength,
        "integrated_strength": integrated,
        "band_concentration": concentration,
        "band_persistence": persistence,
        "local_contrast": local_contrast,
        "harmonic_2_score": harmonic_2,
        "harmonic_3_score": harmonic_3,
        "harmonic_support_count": int(harmonic_2 >= params.harmonic_min_relative)
        + int(harmonic_3 >= params.harmonic_min_relative),
        "base_score": base_score,
        "total_score": total_score,
    }


def evaluate_cprf(
    normalized_cwt: np.ndarray,
    periods: np.ndarray,
    *,
    normalization_threshold: float,
    params: CPRFParameters | None = None,
) -> CPRFResult:
    """Evaluate one PELT window using the unique production CPRF method."""
    params = params or CPRFParameters()
    params.validate()
    values = np.asarray(normalized_cwt, dtype=np.float32)
    period_values = np.asarray(periods, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != period_values.size or values.shape[1] == 0:
        raise ValueError("normalized_cwt must have shape (periods, non-empty records)")
    values = np.maximum(np.where(np.isfinite(values), values, 0.0), 0.0)
    profile, occupancy = _reduce_time(values, params)
    candidates = [
        _candidate_metrics(profile, occupancy, period_values, int(peak), params)
        for peak in _peak_indices(profile, params.max_peak_hypotheses)
    ]
    selected = max(candidates, key=lambda item: float(item["total_score"]))
    accepted = (
        int(selected["width"]) >= params.min_width_bins
        and float(selected["peak_strength"]) >= params.min_peak_strength
        and float(selected["integrated_strength"]) >= params.min_integrated_strength
        and float(selected["band_persistence"]) >= params.min_band_persistence
        and float(selected["band_concentration"]) >= params.min_band_concentration
        and float(selected["local_contrast"]) >= params.min_local_contrast
    )
    peak = int(selected["peak"])
    start = int(selected["start"])
    stop = int(selected["stop"])
    return CPRFResult(
        accepted=bool(accepted),
        peak_period_records=float(period_values[peak]),
        period_start_records=float(period_values[start]),
        period_stop_records=float(period_values[stop - 1]),
        peak_index=peak,
        band_start_index=start,
        band_stop_index=stop,
        width_bins=int(selected["width"]),
        peak_strength=float(selected["peak_strength"]),
        integrated_strength=float(selected["integrated_strength"]),
        band_concentration=float(selected["band_concentration"]),
        band_persistence=float(selected["band_persistence"]),
        local_contrast=float(selected["local_contrast"]),
        harmonic_2_score=float(selected["harmonic_2_score"]),
        harmonic_3_score=float(selected["harmonic_3_score"]),
        harmonic_support_count=int(selected["harmonic_support_count"]),
        base_score=float(selected["base_score"]),
        total_score=float(selected["total_score"]),
        normalization_threshold=float(normalization_threshold),
        profile=profile,
    )
