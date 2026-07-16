"""Linear period-profile scoring filters for second-stage scientific ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any

import numpy as np
from scipy import ndimage


MIN_POSITIVE = float(np.finfo(np.float32).tiny)


@dataclass(frozen=True)
class PeriodProfileAlgorithm:
    name: str
    reducer: str
    score_mode: str
    smooth_bins: int = 1
    width_fraction: float = 0.50
    min_width_bins: int = 2
    min_peak_strength: float = 0.75
    min_integrated_strength: float = 1.25
    min_band_persistence: float = 0.35
    min_band_concentration: float = 0.0
    min_local_contrast: float = 0.0
    min_total_score: float = 0.0
    harmonic_weight: float = 0.0
    harmonic_min_relative: float = 0.12
    harmonic_window_scale: float = 1.25
    top_fraction: float = 0.25
    max_peak_hypotheses: int = 8

    def validate(self) -> None:
        if self.reducer not in {
            "mean",
            "rms",
            "occupied_mean",
            "persistence_sqrt",
            "stable_mean",
            "top_fraction_mean",
        }:
            raise ValueError(f"Unknown period-profile reducer: {self.reducer}")
        if self.score_mode not in {"linear", "geometric", "strength_area"}:
            raise ValueError(f"Unknown period-profile score mode: {self.score_mode}")
        if self.smooth_bins < 1 or self.min_width_bins < 1:
            raise ValueError("period widths must be positive")
        if not 0.0 < self.width_fraction < 1.0:
            raise ValueError("width_fraction must be in (0, 1)")
        if self.min_peak_strength < 0.0 or self.min_integrated_strength < 0.0:
            raise ValueError("main-peak strength thresholds must be non-negative")
        if not 0.0 <= self.min_band_persistence <= 1.0:
            raise ValueError("min_band_persistence must be in [0, 1]")
        if not 0.0 <= self.min_band_concentration <= 1.0:
            raise ValueError("min_band_concentration must be in [0, 1]")
        if self.min_local_contrast < 0.0 or self.min_total_score < 0.0:
            raise ValueError("score-filter thresholds must be non-negative")
        if self.harmonic_weight < 0.0 or self.harmonic_min_relative < 0.0:
            raise ValueError("harmonic parameters must be non-negative")
        if self.harmonic_window_scale <= 0.0:
            raise ValueError("harmonic_window_scale must be positive")
        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1]")
        if self.max_peak_hypotheses < 1:
            raise ValueError("max_peak_hypotheses must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PeriodProfileResult:
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
    profile: np.ndarray


def period_profile_catalog() -> tuple[PeriodProfileAlgorithm, ...]:
    """Return a focused score-filter grid rather than unrelated image operators."""
    algorithms: list[PeriodProfileAlgorithm] = [
        PeriodProfileAlgorithm(
            name="placeholder_sumlike_s1",
            reducer="mean",
            score_mode="strength_area",
            smooth_bins=1,
            width_fraction=0.50,
            min_width_bins=1,
            min_peak_strength=0.0,
            min_integrated_strength=0.0,
            min_band_persistence=0.0,
        ),
        # CPRF is the minimal frozen candidate found by the full-band ablation.
        # Integrated strength and persistence did not change any decision once
        # prominence, width, concentration, and contrast were present.
        PeriodProfileAlgorithm(
            name="cprf_absolute_ridge_c35_r140",
            reducer="persistence_sqrt",
            score_mode="geometric",
            smooth_bins=3,
            width_fraction=0.50,
            min_width_bins=3,
            min_peak_strength=1.25,
            min_integrated_strength=0.0,
            min_band_persistence=0.0,
            min_band_concentration=0.35,
            min_local_contrast=1.40,
            harmonic_weight=0.20,
        ),
        # Frozen after the strict CPRO activity -> PELT -> raw-CWT profile rank.
        PeriodProfileAlgorithm(
            name="cprf_concentrated_ridge_c45",
            reducer="persistence_sqrt",
            score_mode="geometric",
            smooth_bins=3,
            width_fraction=0.50,
            min_width_bins=3,
            min_peak_strength=1.25,
            min_integrated_strength=2.0,
            min_band_persistence=0.35,
            min_band_concentration=0.45,
            harmonic_weight=0.20,
        ),
    ]
    reducers = (
        "mean",
        "rms",
        "occupied_mean",
        "persistence_sqrt",
        "stable_mean",
        "top_fraction_mean",
    )
    for reducer in reducers:
        for smooth_bins in (1, 3):
            for width_fraction in (0.35, 0.50):
                for score_mode in ("linear", "geometric"):
                    for harmonic_weight in (0.0, 0.20):
                        algorithms.append(
                            PeriodProfileAlgorithm(
                                name=(
                                    f"pbsf_{reducer}_s{smooth_bins}_"
                                    f"w{int(round(100 * width_fraction)):02d}_"
                                    f"{score_mode}_h{int(round(100 * harmonic_weight)):02d}"
                                ),
                                reducer=reducer,
                                score_mode=score_mode,
                                smooth_bins=smooth_bins,
                                width_fraction=width_fraction,
                                harmonic_weight=harmonic_weight,
                            )
                        )
    for peak_strength in (0.50, 1.00, 1.50):
        for width_bins in (2, 3, 4):
            algorithms.append(
                PeriodProfileAlgorithm(
                    name=f"pbsf_persistence_sqrt_gate_e{int(peak_strength * 100):03d}_b{width_bins}",
                    reducer="persistence_sqrt",
                    score_mode="geometric",
                    smooth_bins=3,
                    width_fraction=0.50,
                    min_width_bins=width_bins,
                    min_peak_strength=peak_strength,
                    min_integrated_strength=max(1.0, peak_strength * sqrt(width_bins)),
                    harmonic_weight=0.20,
                )
            )
    focused_base = {
        "reducer": "persistence_sqrt",
        "score_mode": "geometric",
        "smooth_bins": 3,
        "width_fraction": 0.50,
        "min_width_bins": 3,
        "min_peak_strength": 1.25,
        "min_integrated_strength": 2.0,
        "min_band_persistence": 0.35,
        "harmonic_weight": 0.20,
    }
    for concentration in (0.30, 0.35, 0.40, 0.45, 0.50):
        algorithms.append(
            PeriodProfileAlgorithm(
                name=f"pbsf_focus_concentration_c{int(round(100 * concentration)):02d}",
                min_band_concentration=concentration,
                **focused_base,
            )
        )
    for contrast in (0.80, 1.00, 1.20, 1.40, 1.60):
        algorithms.append(
            PeriodProfileAlgorithm(
                name=f"pbsf_focus_contrast_r{int(round(100 * contrast)):03d}",
                min_local_contrast=contrast,
                **focused_base,
            )
        )
    for integrated in (2.5, 3.0, 3.5, 4.0, 4.5):
        algorithms.append(
            PeriodProfileAlgorithm(
                name=f"pbsf_focus_integrated_a{int(round(100 * integrated)):03d}",
                min_integrated_strength=integrated,
                **{key: value for key, value in focused_base.items() if key != "min_integrated_strength"},
            )
        )
    for total in (1.25, 1.50, 1.75, 2.00, 2.25):
        algorithms.append(
            PeriodProfileAlgorithm(
                name=f"pbsf_focus_total_s{int(round(100 * total)):03d}",
                min_total_score=total,
                **focused_base,
            )
        )
    for concentration in (0.35, 0.40, 0.45):
        for contrast in (1.00, 1.20, 1.40):
            algorithms.append(
                PeriodProfileAlgorithm(
                    name=(
                        f"pbsf_focus_joint_c{int(round(100 * concentration)):02d}_"
                        f"r{int(round(100 * contrast)):03d}"
                    ),
                    min_band_concentration=concentration,
                    min_local_contrast=contrast,
                    **focused_base,
                )
            )
    unique = {item.name: item for item in algorithms}
    return tuple(unique[name] for name in sorted(unique))


def _reduce_time(score_map: np.ndarray, algorithm: PeriodProfileAlgorithm) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(score_map, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("score_map must have shape (periods, records) with non-zero dimensions")
    values = np.maximum(np.where(np.isfinite(values), values, 0.0), 0.0)
    # Inputs are in CPRO-threshold units, so occupancy has a physical threshold.
    occupied = values >= 1.0
    occupied_values = np.where(occupied, values, 0.0)
    occupancy = np.mean(occupied, axis=1, dtype=np.float64).astype(np.float32)
    mean = np.mean(values, axis=1, dtype=np.float64).astype(np.float32)
    if algorithm.reducer == "mean":
        profile = mean
    elif algorithm.reducer == "rms":
        profile = np.sqrt(np.mean(np.square(values, dtype=np.float64), axis=1)).astype(np.float32)
    elif algorithm.reducer == "occupied_mean":
        profile = np.sum(occupied_values, axis=1, dtype=np.float64) / np.maximum(
            np.count_nonzero(occupied, axis=1), 1
        )
    elif algorithm.reducer == "persistence_sqrt":
        occupied_mean = np.sum(occupied_values, axis=1, dtype=np.float64) / np.maximum(
            np.count_nonzero(occupied, axis=1), 1
        )
        profile = occupied_mean * np.sqrt(occupancy)
    elif algorithm.reducer == "stable_mean":
        std = np.std(values, axis=1, dtype=np.float64)
        coefficient = std / np.maximum(mean, MIN_POSITIVE)
        profile = mean / (1.0 + coefficient)
    else:
        count = max(1, int(np.ceil(float(algorithm.top_fraction) * values.shape[1])))
        split = values.shape[1] - count
        top = np.partition(values, split, axis=1)[:, split:]
        profile = np.mean(top, axis=1, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float32)
    if algorithm.smooth_bins > 1:
        profile = ndimage.uniform_filter1d(
            profile,
            size=min(int(algorithm.smooth_bins), profile.size),
            mode="nearest",
        ).astype(np.float32, copy=False)
    return profile, occupancy


def _peak_indices(profile: np.ndarray, maximum: int) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float32)
    if values.size == 1:
        return np.array([0], dtype=np.int64)
    peaks = np.flatnonzero(
        np.concatenate(
            (
                [values[0] >= values[1]],
                (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]),
                [values[-1] > values[-2]],
            )
        )
    )
    if peaks.size == 0:
        peaks = np.array([int(np.nanargmax(values))], dtype=np.int64)
    order = np.argsort(values[peaks], kind="stable")[::-1]
    return peaks[order[: max(1, int(maximum))]]


def _peak_band(profile: np.ndarray, peak: int, fraction: float) -> tuple[int, int]:
    values = np.asarray(profile, dtype=np.float32)
    positive = values[values > 0.0]
    baseline = float(np.median(positive)) if positive.size else 0.0
    threshold = baseline + float(fraction) * max(0.0, float(values[peak]) - baseline)
    start = int(peak)
    stop = int(peak) + 1
    while start > 0 and values[start - 1] >= threshold:
        start -= 1
    while stop < values.size and values[stop] >= threshold:
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
    frequencies = 1.0 / np.asarray(periods, dtype=np.float64)
    center_frequency = float(order) / float(peak_period)
    band_frequencies = frequencies[band_start:band_stop]
    if band_frequencies.size == 0:
        return 0.0
    main_half_width = 0.5 * abs(float(np.max(band_frequencies)) - float(np.min(band_frequencies)))
    if frequencies.size > 1:
        nearest = int(np.argmin(np.abs(frequencies - center_frequency)))
        neighbor = min(frequencies.size - 1, nearest + 1) if nearest == 0 else nearest - 1
        grid_half_width = 0.5 * abs(float(frequencies[nearest] - frequencies[neighbor]))
    else:
        grid_half_width = 0.0
    half_width = max(float(order) * main_half_width * float(window_scale), grid_half_width)
    mask = np.abs(frequencies - center_frequency) <= half_width
    if not np.any(mask):
        return 0.0
    harmonic_peak = float(np.max(profile[mask]))
    return float(np.clip(harmonic_peak / max(float(main_peak), MIN_POSITIVE), 0.0, 1.0))


def _candidate_metrics(
    profile: np.ndarray,
    occupancy: np.ndarray,
    periods: np.ndarray,
    peak: int,
    algorithm: PeriodProfileAlgorithm,
) -> dict[str, float | int]:
    start, stop = _peak_band(profile, peak, algorithm.width_fraction)
    band = profile[start:stop]
    baseline = _local_background(profile, start, stop)
    excess = np.maximum(band - baseline, 0.0)
    width = max(1, stop - start)
    peak_strength = max(0.0, float(profile[peak]) - baseline)
    integrated = float(np.sum(excess, dtype=np.float64) / sqrt(width))
    concentration = float(np.sum(band, dtype=np.float64) / max(np.sum(profile, dtype=np.float64), MIN_POSITIVE))
    persistence = float(
        np.sum(occupancy[start:stop] * np.maximum(band, 0.0), dtype=np.float64)
        / max(np.sum(np.maximum(band, 0.0), dtype=np.float64), MIN_POSITIVE)
    )
    local_contrast = float((np.mean(band, dtype=np.float64) - baseline) / max(abs(baseline), 1.0))
    peak_period = float(periods[peak])
    harmonic_2 = _harmonic_score(
        profile,
        periods,
        peak_period=peak_period,
        band_start=start,
        band_stop=stop,
        order=2,
        window_scale=algorithm.harmonic_window_scale,
        main_peak=float(profile[peak]),
    )
    harmonic_3 = _harmonic_score(
        profile,
        periods,
        peak_period=peak_period,
        band_start=start,
        band_stop=stop,
        order=3,
        window_scale=algorithm.harmonic_window_scale,
        main_peak=float(profile[peak]),
    )
    harmonic_support = int(harmonic_2 >= algorithm.harmonic_min_relative) + int(
        harmonic_3 >= algorithm.harmonic_min_relative
    )
    if algorithm.score_mode == "linear":
        base_score = (
            0.40 * peak_strength
            + 0.30 * integrated
            + 0.15 * concentration
            + 0.15 * persistence
        )
    elif algorithm.score_mode == "geometric":
        base_score = (
            max(peak_strength, MIN_POSITIVE) ** 0.40
            * max(integrated, MIN_POSITIVE) ** 0.30
            * max(concentration, MIN_POSITIVE) ** 0.15
            * max(persistence, MIN_POSITIVE) ** 0.15
        )
    else:
        base_score = peak_strength * integrated
    harmonic_mean = 0.5 * (harmonic_2 + harmonic_3)
    total_score = float(base_score * (1.0 + algorithm.harmonic_weight * harmonic_mean))
    return {
        "peak": int(peak),
        "start": start,
        "stop": stop,
        "width": width,
        "peak_strength": peak_strength,
        "integrated": integrated,
        "concentration": concentration,
        "persistence": persistence,
        "local_contrast": local_contrast,
        "harmonic_2": harmonic_2,
        "harmonic_3": harmonic_3,
        "harmonic_support": harmonic_support,
        "base_score": float(base_score),
        "total_score": total_score,
    }


def evaluate_period_profile(
    score_map: np.ndarray,
    periods: np.ndarray,
    algorithm: PeriodProfileAlgorithm,
) -> PeriodProfileResult:
    """Evaluate one CPRO window in linear time and return one peak-family decision."""
    algorithm.validate()
    period_values = np.asarray(periods, dtype=np.float64)
    if period_values.ndim != 1 or period_values.size != np.asarray(score_map).shape[0]:
        raise ValueError("periods must match score_map's period axis")
    profile, occupancy = _reduce_time(score_map, algorithm)
    candidates = [
        _candidate_metrics(profile, occupancy, period_values, int(peak), algorithm)
        for peak in _peak_indices(profile, algorithm.max_peak_hypotheses)
    ]
    selected = max(candidates, key=lambda item: float(item["total_score"]))
    accepted = (
        int(selected["width"]) >= algorithm.min_width_bins
        and float(selected["peak_strength"]) >= algorithm.min_peak_strength
        and float(selected["integrated"]) >= algorithm.min_integrated_strength
        and float(selected["persistence"]) >= algorithm.min_band_persistence
        and float(selected["concentration"]) >= algorithm.min_band_concentration
        and float(selected["local_contrast"]) >= algorithm.min_local_contrast
        and float(selected["total_score"]) >= algorithm.min_total_score
    )
    start = int(selected["start"])
    stop = int(selected["stop"])
    peak = int(selected["peak"])
    return PeriodProfileResult(
        accepted=bool(accepted),
        peak_period_records=float(period_values[peak]),
        period_start_records=float(period_values[start]),
        period_stop_records=float(period_values[stop - 1]),
        peak_index=peak,
        band_start_index=start,
        band_stop_index=stop,
        width_bins=int(selected["width"]),
        peak_strength=float(selected["peak_strength"]),
        integrated_strength=float(selected["integrated"]),
        band_concentration=float(selected["concentration"]),
        band_persistence=float(selected["persistence"]),
        local_contrast=float(selected["local_contrast"]),
        harmonic_2_score=float(selected["harmonic_2"]),
        harmonic_3_score=float(selected["harmonic_3"]),
        harmonic_support_count=int(selected["harmonic_support"]),
        base_score=float(selected["base_score"]),
        total_score=float(selected["total_score"]),
        profile=profile,
    )
