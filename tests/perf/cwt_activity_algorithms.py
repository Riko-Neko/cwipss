"""Independent post-CWT 2D-to-1D activity candidates.

Each candidate receives the raw CWT power map with shape `(periods, records)`
and returns a 1D time activity plus its own diagnostic score map.  This module
intentionally avoids a shared structure-preprocessing stage.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from cwipss.signal.activity import smooth_activity


MIN_POSITIVE = 1e-12


@dataclass(frozen=True)
class CWTActivityAlgorithm:
    name: str
    family: str
    description: str
    method: str
    source_refs: tuple[str, ...]
    complexity: str
    migratability: str
    input_denoiser: str = "none"
    params: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True)
class CWTActivityResult:
    activity: np.ndarray
    score_map: np.ndarray


DEFAULT_CWT_ACTIVITY_ALGORITHMS = (
    "row_mad_topk_mean",
    "ridge_cfar_wide_4cycle",
    "post_freq_max8_4cycle",
    "post_freq_max8_6cycle_s20",
    "post_freq_max8_8cycle_s50",
    "post_freq_max8_center_8cycle_s70_f20",
)

TOP1_SINGLE_GATE_FLOOR_ALGORITHMS = (
    "post_freq_max8_center_8cycle_f25",
    "post_freq_max8_center_8cycle_f30",
    "post_freq_max8_center_8cycle_f40",
    "post_freq_max8_center_8cycle_f50",
    "post_freq_max8_center_8cycle_f52",
    "post_freq_max8_center_8cycle_f54",
    "post_freq_max8_center_8cycle_f56",
    "post_freq_max8_center_8cycle_f58",
    "post_freq_max8_center_8cycle_f60",
    "post_freq_max8_center_8cycle_f70",
    "post_freq_max8_center_8cycle_f80",
)

TOP1_CONTINUOUS_SUPPORT_ALGORITHMS = (
    "post_freq_max8_center_8cycle_q_f30",
    "post_freq_max8_center_8cycle_q_f40",
    "post_freq_max8_center_8cycle_q_f50",
    "post_freq_max8_center_8cycle_q_f52",
    "post_freq_max8_center_8cycle_q_f54",
    "post_freq_max8_center_8cycle_q_f56",
    "post_freq_max8_center_8cycle_q_f58",
    "post_freq_max8_center_8cycle_q_f60",
)

TOP1_SIMPLIFICATION_ALGORITHMS = (
    "post_freq_max8_center_8cycle",
    "post_freq_max8_center_8cycle_f20",
    *TOP1_SINGLE_GATE_FLOOR_ALGORITHMS,
    "post_freq_max8_center_8cycle_s70",
    "post_freq_max8_center_8cycle_s70_f20",
)

TOP1_ADAPTATION_ALGORITHMS = (
    "post_freq_max4_center_8cycle",
    "post_freq_max8_center_4cycle",
    "post_freq_max8_center_6cycle",
    "post_freq_max8_center_8cycle",
    "post_freq_max8_center_12cycle",
    "post_freq_max8_guard2_center_8cycle",
    "post_freq_max12_center_8cycle",
)

HIGH_RANK_MECHANISM_CONTROLS = (
    "spectral_kurtosis_window",
    "viterbi_ridge_path",
    "low_rank_residual_max",
)


def _finite_power(power: np.ndarray) -> np.ndarray:
    values = np.asarray(power, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("CWT power must have shape (periods, records)")
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float32)
    floor = max(float(np.nanmin(values[finite])), 0.0)
    clean = np.where(finite, values, floor)
    clean = np.maximum(clean, 0.0)
    return clean.astype(np.float32, copy=False)


def _low_fraction_floor(values: np.ndarray, fraction: float = 0.20) -> float:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return MIN_POSITIVE
    fraction = min(max(float(fraction), 1.0 / finite.size), 1.0)
    k = max(1, int(math.ceil(fraction * finite.size)))
    low = np.partition(finite, k - 1)[:k]
    floor = float(np.nanmean(low))
    return max(floor if np.isfinite(floor) else MIN_POSITIVE, MIN_POSITIVE)


def _log_floor_ratio(power: np.ndarray, floor_fraction: float = 0.20) -> np.ndarray:
    values = _finite_power(power)
    floor = _low_fraction_floor(values, floor_fraction)
    score = np.log1p(np.maximum(values / floor - 1.0, 0.0))
    score[~np.isfinite(score)] = 0.0
    return score.astype(np.float32, copy=False)


def _row_mad_z(power: np.ndarray, baseline_quantile: float = 0.10) -> np.ndarray:
    values = _finite_power(power)
    q = min(max(float(baseline_quantile), 0.0), 0.45)
    baseline = np.nanquantile(values, q, axis=1, keepdims=True)
    centered = values - baseline
    low = np.where(values <= baseline, centered, np.nan)
    low_median = np.nanmedian(low, axis=1, keepdims=True)
    low_mad = np.nanmedian(np.abs(low - low_median), axis=1, keepdims=True)
    fallback_mad = np.nanmedian(np.abs(centered - np.nanmedian(centered, axis=1, keepdims=True)), axis=1, keepdims=True)
    scale = np.where(np.isfinite(low_mad) & (low_mad > 0.0), low_mad, fallback_mad)
    scale = np.maximum(1.4826 * scale, 1.0)
    score = centered / scale
    score[~np.isfinite(score)] = 0.0
    return score.astype(np.float32, copy=False)


def _positive(values: np.ndarray) -> np.ndarray:
    score = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    score[~np.isfinite(score)] = 0.0
    return score.astype(np.float32, copy=False)


def _topk_mean(values: np.ndarray, top_k: int) -> np.ndarray:
    score = _positive(values)
    if score.shape[0] == 0:
        return np.zeros(score.shape[1], dtype=np.float32)
    k = max(1, min(int(top_k), score.shape[0]))
    top = np.partition(score, kth=score.shape[0] - k, axis=0)[-k:, :]
    return np.nanmean(top, axis=0).astype(np.float32, copy=False)


def _concentration_weighted_period_pool(
    values: np.ndarray,
    *,
    mode: str,
    guard_bins: int,
    power: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a coherent map while penalizing simultaneous independent ridges."""
    score = _positive(values)
    period_count, records = score.shape
    if period_count == 0:
        return np.zeros(records, dtype=np.float32), score
    peak_index = np.argmax(score, axis=0)
    peak = np.take_along_axis(score, peak_index[None, :], axis=0)[0]
    total = np.sum(score, axis=0)
    eps = np.maximum(peak * 1e-6, MIN_POSITIVE)
    guard = max(0, min(int(guard_bins), period_count - 1))
    rows = np.arange(period_count, dtype=np.int64)[:, None]
    local = np.abs(rows - peak_index[None, :]) <= guard
    local_mass = np.sum(np.where(local, score, 0.0), axis=0)
    outside_peak = np.max(np.where(local, 0.0, score), axis=0)
    band_share = np.clip(local_mass / np.maximum(total, eps), 0.0, 1.0)
    outside_ratio = np.clip((peak - outside_peak) / np.maximum(peak, MIN_POSITIVE), 0.0, 1.0)
    squared_mass = np.sum(score * score, axis=0)
    ipr = squared_mass / np.maximum(total * total, MIN_POSITIVE)
    minimum_ipr = 1.0 / float(period_count)
    normalized_ipr = np.clip((ipr - minimum_ipr) / max(1e-6, 1.0 - minimum_ipr), 0.0, 1.0)
    pooling = str(mode).strip().lower()
    if pooling == "band_share":
        weight = band_share
    elif pooling == "normalized_ipr":
        weight = normalized_ipr
    elif pooling == "outside_ratio":
        weight = outside_ratio
    elif pooling == "share_outside":
        weight = np.sqrt(band_share * outside_ratio)
    elif pooling == "ipr_outside":
        weight = np.sqrt(normalized_ipr * outside_ratio)
    else:
        raise ValueError(f"Unknown concentration pooling mode: {mode}")
    exponent = max(0.0, float(power))
    if not math.isclose(exponent, 1.0):
        weight = np.power(weight, exponent)
    weight = np.where(np.isfinite(weight), np.clip(weight, 0.0, 1.0), 0.0).astype(np.float32)
    weighted = (score * weight[None, :]).astype(np.float32, copy=False)
    return (peak * weight).astype(np.float32, copy=False), weighted


def _winner_persistence_weight(
    values: np.ndarray,
    periods: np.ndarray,
    *,
    guard_bins: int,
    support_cycles: float,
) -> np.ndarray:
    """Measure whether the strongest ridge remains near one period over time."""
    score = _positive(values)
    period_values = np.asarray(periods, dtype=np.float64)
    period_count, records = score.shape
    peak_index = np.argmax(score, axis=0)
    rows = np.arange(period_count, dtype=np.int64)[:, None]
    winner_support = (np.abs(rows - peak_index[None, :]) <= max(0, int(guard_bins))).astype(np.float32)
    smoothed = np.zeros_like(winner_support, dtype=np.float32)
    cycles = max(0.0, float(support_cycles))
    for row, period in enumerate(period_values):
        width = max(3, int(round(cycles * max(float(period), 1.0))))
        width = min(width, max(3, records if records % 2 == 1 else records - 1))
        if width % 2 == 0:
            width = max(3, width - 1)
        smoothed[row] = ndimage.uniform_filter1d(
            winner_support[row], size=width, mode="constant", cval=0.0
        )
    weight = np.take_along_axis(smoothed, peak_index[None, :], axis=0)[0]
    return np.clip(weight, 0.0, 1.0).astype(np.float32, copy=False)


def _period_sideband_log_ratio(
    power: np.ndarray,
    *,
    inner_width: int,
    outer_width: int,
    clip_log_ratio: float,
) -> np.ndarray:
    """Remove broad period-axis background with a local sideband reference."""
    values = _finite_power(power)
    period_count = int(values.shape[0])
    inner = max(1, min(int(inner_width), period_count))
    outer = max(inner + 2, min(int(outer_width), period_count))
    if inner % 2 == 0:
        inner = max(1, inner - 1)
    if outer % 2 == 0:
        outer = max(inner + 2, outer - 1)
    if outer <= inner:
        return np.zeros_like(values, dtype=np.float32)

    inner_mean = ndimage.uniform_filter1d(values, size=inner, axis=0, mode="nearest")
    outer_mean = ndimage.uniform_filter1d(values, size=outer, axis=0, mode="nearest")
    sideband = (outer_mean * float(outer) - inner_mean * float(inner)) / float(outer - inner)
    positive_sideband = sideband[sideband > 0.0]
    reference = float(np.nanmedian(positive_sideband)) if positive_sideband.size else MIN_POSITIVE
    eps = max(reference * 1e-6, MIN_POSITIVE)
    ratio = np.log((inner_mean + eps) / (np.maximum(sideband, 0.0) + eps))
    limit = max(0.1, float(clip_log_ratio))
    ratio = np.clip(ratio, -limit, limit)

    # The local reference is incomplete at the period-grid boundaries.
    period_margin = outer // 2
    if period_margin > 0:
        ratio[:period_margin, :] = 0.0
        ratio[-period_margin:, :] = 0.0
    ratio[~np.isfinite(ratio)] = 0.0
    return ratio.astype(np.float32, copy=False)


def _coherent_ridge_cfar(
    power: np.ndarray,
    periods: np.ndarray,
    *,
    inner_width: int,
    outer_width: int,
    time_support_cycles: float,
    clip_log_ratio: float,
    top_k: int,
    min_positive_support: float = 0.0,
    support_weighting: str = "none",
    score_floor: float = 0.0,
    period_pooling: str = "topk_mean",
    period_guard_bins: int = 2,
    concentration_power: float = 1.0,
    winner_support_cycles: float = 4.0,
) -> CWTActivityResult:
    """Keep narrow period ridges that persist for multiple signal cycles.

    Signed local contrast is averaged before positive projection, so isolated
    natural bright pixels cannot accumulate one-sided evidence. The support
    width follows CWT period, preserving signals with different periods while
    retaining O(P*T) complexity.
    """
    ridge = _period_sideband_log_ratio(
        power,
        inner_width=inner_width,
        outer_width=outer_width,
        clip_log_ratio=clip_log_ratio,
    )
    result = _coherent_signed_score(
        ridge,
        periods,
        time_support_cycles=time_support_cycles,
        top_k=top_k,
        min_positive_support=min_positive_support,
        support_weighting=support_weighting,
        score_floor=score_floor,
        max_positive_fraction=0.0,
    )
    pooling = str(period_pooling).strip().lower()
    if pooling == "topk_mean":
        return result
    if pooling in {"winner_persistence", "winner_band_share"}:
        if pooling == "winner_band_share":
            base_activity, base_score = _concentration_weighted_period_pool(
                result.score_map,
                mode="band_share",
                guard_bins=period_guard_bins,
                power=1.0,
            )
        else:
            base_score = result.score_map
            base_activity = np.nanmax(base_score, axis=0).astype(np.float32, copy=False)
        persistence = _winner_persistence_weight(
            result.score_map,
            periods,
            guard_bins=period_guard_bins,
            support_cycles=winner_support_cycles,
        )
        weight = np.power(persistence, max(0.0, float(concentration_power))).astype(np.float32)
        return CWTActivityResult(
            activity=(base_activity * weight).astype(np.float32, copy=False),
            score_map=(base_score * weight[None, :]).astype(np.float32, copy=False),
        )
    activity, weighted = _concentration_weighted_period_pool(
        result.score_map,
        mode=pooling,
        guard_bins=period_guard_bins,
        power=concentration_power,
    )
    return CWTActivityResult(activity=activity, score_map=weighted)


def _top1_single_floor_candidates() -> list[CWTActivityAlgorithm]:
    candidates: list[CWTActivityAlgorithm] = []
    for floor in (0.25, 0.30, 0.40, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.70, 0.80):
        floor_code = int(round(100.0 * floor))
        candidates.append(
            CWTActivityAlgorithm(
                name=f"post_freq_max8_center_8cycle_f{floor_code:02d}",
                family="simplified_post_cwt_frequency_cfar",
                description=(
                    "Top1 without persistence weighting, using one calibrated "
                    f"native score floor at {floor:.2f}."
                ),
                method="post_cwt_frequency_cfar",
                source_refs=("greatest_of_CFAR", "robust_map_calibration"),
                complexity="O(K*P*T)",
                migratability="Nine global map medians, neighbor max, one time filter, and one native floor.",
                input_denoiser="post_cwt_neighbor8",
                params={
                    "reference_statistic": "max",
                    "time_support_cycles": 8.0,
                    "score_floor": floor,
                    "map_normalization": "center",
                    "clip_log_ratio": 1.5,
                    "top_k": 3,
                },
            )
        )
    return candidates


def _top1_continuous_support_candidates() -> list[CWTActivityAlgorithm]:
    candidates: list[CWTActivityAlgorithm] = []
    for floor in (0.30, 0.40, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60):
        floor_code = int(round(100.0 * floor))
        candidates.append(
            CWTActivityAlgorithm(
                name=f"post_freq_max8_center_8cycle_q_f{floor_code:02d}",
                family="simplified_post_cwt_frequency_cfar",
                description=(
                    "Centered max8 CFAR using continuous positive-support weighting "
                    f"and one native score floor at {floor:.2f}."
                ),
                method="post_cwt_frequency_cfar",
                source_refs=("greatest_of_CFAR", "robust_map_calibration", "continuous_persistence_weight"),
                complexity="O(K*P*T)",
                migratability="Nine map medians, neighbor max, two time filters, and one native floor.",
                input_denoiser="post_cwt_neighbor8",
                params={
                    "reference_statistic": "max",
                    "time_support_cycles": 8.0,
                    "support_weighting": "fraction",
                    "score_floor": floor,
                    "map_normalization": "center",
                    "clip_log_ratio": 1.5,
                    "top_k": 3,
                },
            )
        )
    return candidates


def _coherent_signed_score(
    signed_score: np.ndarray,
    periods: np.ndarray,
    *,
    time_support_cycles: float,
    top_k: int,
    min_positive_support: float,
    support_weighting: str,
    score_floor: float,
    max_positive_fraction: float,
) -> CWTActivityResult:
    ridge = np.asarray(signed_score, dtype=np.float32)
    period_values = np.asarray(periods, dtype=np.float64)
    if period_values.size != ridge.shape[0]:
        raise ValueError("period count must match CWT power rows")
    records = int(ridge.shape[1])
    coherent = np.zeros_like(ridge, dtype=np.float32)
    cycles = max(0.0, float(time_support_cycles))
    support_floor = min(max(float(min_positive_support), 0.0), 0.95)
    support_mode = str(support_weighting).strip().lower()
    if support_mode not in {"none", "fraction"}:
        raise ValueError(f"Unknown support weighting: {support_weighting}")
    for row, period in enumerate(period_values):
        if cycles <= 0.0:
            filtered = ridge[row]
            half_support = 0
        else:
            width = max(3, int(round(cycles * max(float(period), 1.0))))
            width = min(width, max(3, records if records % 2 == 1 else records - 1))
            if width % 2 == 0:
                width = max(3, width - 1)
            filtered = ndimage.uniform_filter1d(ridge[row], size=width, mode="constant", cval=0.0)
            half_support = width // 2
        if support_floor > 0.0 or support_mode == "fraction":
            if cycles <= 0.0:
                support = (ridge[row] > 0.0).astype(np.float32)
            else:
                support = ndimage.uniform_filter1d(
                    (ridge[row] > 0.0).astype(np.float32),
                    size=width,
                    mode="constant",
                    cval=0.0,
                )
            if support_mode == "fraction" and support_floor <= 0.0:
                support_weight = np.clip(support, 0.0, 1.0)
            else:
                support_weight = np.clip(
                    (support - support_floor) / max(1e-6, 1.0 - support_floor),
                    0.0,
                    1.0,
                )
            filtered = filtered * support_weight
        # Exclude the CWT cone of influence and the matched-filter edge support.
        margin = min(records // 2, max(int(math.ceil(max(float(period), 1.0))), half_support))
        if margin > 0:
            filtered = np.asarray(filtered, dtype=np.float32).copy()
            filtered[:margin] = 0.0
            filtered[records - margin :] = 0.0
        coherent[row] = filtered
    coherent = _positive(coherent)
    occupancy_limit = min(max(float(max_positive_fraction), 0.0), 1.0)
    if occupancy_limit > 0.0 and float(np.nanmean(coherent > 0.0)) > occupancy_limit:
        coherent.fill(0.0)
    amplitude_floor = max(0.0, float(score_floor))
    if amplitude_floor > 0.0:
        coherent = np.where(coherent >= amplitude_floor, coherent, 0.0).astype(np.float32, copy=False)
    return CWTActivityResult(
        activity=_topk_mean(coherent, top_k=top_k),
        score_map=coherent,
    )


def _post_cwt_frequency_cfar(
    power: np.ndarray,
    reference_power: np.ndarray,
    periods: np.ndarray,
    *,
    reference_statistic: str,
    mad_multiplier: float,
    time_support_cycles: float,
    clip_log_ratio: float,
    top_k: int,
    min_positive_support: float,
    support_weighting: str,
    calibration_quantile: float,
    period_background_width: int,
    map_normalization: str,
    score_floor: float,
    max_positive_fraction: float,
) -> CWTActivityResult:
    target = _finite_power(power)
    references = np.asarray(reference_power, dtype=np.float32)
    if references.ndim != 3 or references.shape[:2] != target.shape:
        raise ValueError("reference CWT power must have shape (periods, records, neighboring_channels)")
    references = np.maximum(np.where(np.isfinite(references), references, 0.0), 0.0)
    positive = references[references > 0.0]
    reference_level = float(np.nanmedian(positive)) if positive.size else MIN_POSITIVE
    eps = max(reference_level * 1e-6, MIN_POSITIVE)
    target_log = np.log(target + eps)
    reference_log = np.log(references + eps)
    normalization = str(map_normalization).strip().lower()
    if normalization != "none":
        if normalization == "mean_center":
            target_center = float(np.nanmean(target_log))
            reference_center = np.nanmean(reference_log, axis=(0, 1), keepdims=True)
        else:
            target_center = _safe_global_quantile(target_log, 0.5)
            reference_center = np.nanmedian(reference_log, axis=(0, 1), keepdims=True)
        target_log = target_log - target_center
        reference_log = reference_log - reference_center
        if normalization == "robust_z":
            target_scale = max(
                1.4826 * _safe_global_quantile(np.abs(target_log), 0.5),
                1e-6,
            )
            reference_scale = 1.4826 * np.nanmedian(
                np.abs(reference_log),
                axis=(0, 1),
                keepdims=True,
            )
            target_log = target_log / target_scale
            reference_log = reference_log / np.maximum(reference_scale, 1e-6)
        elif normalization not in {"center", "mean_center"}:
            raise ValueError(f"Unknown post-CWT map normalization: {map_normalization}")
    statistic = str(reference_statistic).strip().lower()
    if statistic == "max":
        background = np.nanmax(reference_log, axis=2)
    else:
        median = np.nanmedian(reference_log, axis=2)
        if statistic == "median_mad":
            mad = 1.4826 * np.nanmedian(np.abs(reference_log - median[:, :, None]), axis=2)
            background = median + max(0.0, float(mad_multiplier)) * mad
        elif statistic == "median":
            background = median
        else:
            raise ValueError(f"Unknown frequency reference statistic: {reference_statistic}")
    difference = target_log - background
    background_width = max(0, int(period_background_width))
    if background_width > 1:
        if background_width % 2 == 0:
            background_width += 1
        broad_period_background = ndimage.uniform_filter1d(
            difference,
            size=background_width,
            axis=0,
            mode="nearest",
        )
        difference = difference - broad_period_background
    calibration_q = min(max(float(calibration_quantile), 0.0), 0.999)
    if calibration_q > 0.0:
        difference = difference - _safe_global_quantile(difference, calibration_q)
    limit = max(0.1, float(clip_log_ratio))
    signed = np.clip(difference, -limit, limit).astype(np.float32, copy=False)
    signed[~np.isfinite(signed)] = 0.0
    return _coherent_signed_score(
        signed,
        periods,
        time_support_cycles=time_support_cycles,
        top_k=top_k,
        min_positive_support=min_positive_support,
        support_weighting=support_weighting,
        score_floor=score_floor,
        max_positive_fraction=max_positive_fraction,
    )


def _safe_global_quantile(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(np.nanquantile(finite, quantile))


def _period_distribution(power: np.ndarray) -> np.ndarray:
    values = _finite_power(power) + MIN_POSITIVE
    total = np.sum(values, axis=0, keepdims=True)
    return (values / np.maximum(total, MIN_POSITIVE)).astype(np.float32, copy=False)


def _sliding_band_sum(values: np.ndarray, width: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    period_count = int(matrix.shape[0])
    width = max(1, min(int(width), period_count))
    cumsum = np.cumsum(matrix, axis=0, dtype=np.float64)
    padded = np.vstack([np.zeros((1, matrix.shape[1]), dtype=np.float64), cumsum])
    return (padded[width:, :] - padded[:-width, :]).astype(np.float32, copy=False)


def _best_band_score(values: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float32)
    period_count, record_count = matrix.shape
    if period_count == 0:
        empty = np.zeros(record_count, dtype=np.float32)
        return empty, matrix
    width = max(1, min(int(width), period_count))
    sums = _sliding_band_sum(matrix, width)
    cols = np.arange(record_count, dtype=np.int64)
    starts = np.nanargmax(sums, axis=0).astype(np.int64)
    best_mean = sums[starts, cols] / float(width)
    global_mean = np.nanmean(matrix, axis=0)
    global_std = np.maximum(np.nanstd(matrix, axis=0), 1e-6)
    activity = ((best_mean - global_mean) / global_std).astype(np.float32, copy=False)
    score_map = np.zeros_like(matrix, dtype=np.float32)
    for col, start in enumerate(starts):
        score_map[start : start + width, col] = max(float(activity[col]), 0.0)
    return activity, score_map


def _gini(values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(np.maximum(np.asarray(values, dtype=np.float64), 0.0), axis=0)
    n = sorted_values.shape[0]
    if n <= 1:
        return np.zeros(sorted_values.shape[1], dtype=np.float32)
    weights = np.arange(1, n + 1, dtype=np.float64)[:, None]
    total = np.sum(sorted_values, axis=0)
    gini = (2.0 * np.sum(weights * sorted_values, axis=0) / np.maximum(n * total, MIN_POSITIVE)) - (n + 1.0) / n
    gini[~np.isfinite(gini)] = 0.0
    return np.clip(gini, 0.0, 1.0).astype(np.float32, copy=False)


def _viterbi_score_path(score_values: np.ndarray, radius: int, penalty: float) -> CWTActivityResult:
    score = _positive(score_values)
    period_count, record_count = score.shape
    if period_count == 0:
        activity = np.zeros(record_count, dtype=np.float32)
        return CWTActivityResult(activity=activity, score_map=score)
    radius = max(0, int(radius))
    back = np.zeros((period_count, record_count), dtype=np.int16)
    dp = score[:, 0].astype(np.float64)
    for col in range(1, record_count):
        previous = dp
        next_dp = np.empty(period_count, dtype=np.float64)
        for row in range(period_count):
            lo = max(0, row - radius)
            hi = min(period_count, row + radius + 1)
            candidates = previous[lo:hi] - float(penalty) * np.abs(np.arange(lo, hi) - row)
            best = int(np.argmax(candidates))
            back[row, col] = lo + best
            next_dp[row] = float(score[row, col]) + float(candidates[best])
        dp = next_dp
    path = np.zeros(record_count, dtype=np.int64)
    path[-1] = int(np.argmax(dp))
    for col in range(record_count - 1, 0, -1):
        path[col - 1] = int(back[path[col], col])
    activity = score[path, np.arange(record_count)].astype(np.float32, copy=False)
    score_map = np.zeros_like(score, dtype=np.float32)
    score_map[path, np.arange(record_count)] = activity
    return CWTActivityResult(activity=activity, score_map=score_map)


def _viterbi_activity(power: np.ndarray, radius: int, penalty: float) -> CWTActivityResult:
    return _viterbi_score_path(_log_floor_ratio(power), radius=radius, penalty=penalty)


def _window_kurtosis_score(score_values: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(score_values, dtype=np.float32)
    x = np.where(np.isfinite(x), x, 0.0).astype(np.float32, copy=False)
    width = max(3, int(window))
    mean1 = ndimage.uniform_filter1d(x, size=width, axis=1, mode="nearest")
    mean2 = ndimage.uniform_filter1d(np.square(x), size=width, axis=1, mode="nearest")
    mean3 = ndimage.uniform_filter1d(np.power(x, 3), size=width, axis=1, mode="nearest")
    mean4 = ndimage.uniform_filter1d(np.power(x, 4), size=width, axis=1, mode="nearest")
    var = np.maximum(mean2 - np.square(mean1), 0.0)
    var_floor = np.nanquantile(var, 0.20, axis=1, keepdims=True)
    stable = var > np.maximum(var_floor, 1e-4)
    var = np.maximum(var, 1e-4)
    fourth = mean4 - 4.0 * mean1 * mean3 + 6.0 * np.square(mean1) * mean2 - 3.0 * np.power(mean1, 4)
    kurt = fourth / np.square(var)
    score = np.where(stable, np.log1p(np.abs(kurt - 3.0)), 0.0)
    score[~np.isfinite(score)] = 0.0
    return np.clip(score, 0.0, 20.0).astype(np.float32, copy=False)


def _spectral_kurtosis(power: np.ndarray, window: int) -> np.ndarray:
    return _window_kurtosis_score(_log_floor_ratio(power), window=window)


def _rank1_projection_from_score(score_values: np.ndarray) -> CWTActivityResult:
    score = np.asarray(score_values, dtype=np.float32)
    centered = score - np.nanmedian(score, axis=1, keepdims=True)
    centered[~np.isfinite(centered)] = 0.0
    if min(centered.shape) == 0 or not np.any(centered):
        activity = np.zeros(centered.shape[1], dtype=np.float32)
        return CWTActivityResult(activity=activity, score_map=np.zeros_like(centered, dtype=np.float32))
    u, singular, vh = np.linalg.svd(centered.astype(np.float64, copy=False), full_matrices=False)
    projection = singular[0] * vh[0, :]
    activity = np.abs(projection).astype(np.float32, copy=False)
    score_map = (np.abs(u[:, [0]]) * activity[None, :]).astype(np.float32, copy=False)
    return CWTActivityResult(activity=activity, score_map=score_map)


def _rank1_projection(power: np.ndarray) -> CWTActivityResult:
    return _rank1_projection_from_score(_log_floor_ratio(power))


def _low_rank_residual_from_score(score_values: np.ndarray, rank: int) -> CWTActivityResult:
    score = np.asarray(score_values, dtype=np.float32)
    centered = score - np.nanmedian(score, axis=1, keepdims=True)
    centered[~np.isfinite(centered)] = 0.0
    if min(centered.shape) == 0 or not np.any(centered):
        activity = np.zeros(centered.shape[1], dtype=np.float32)
        return CWTActivityResult(activity=activity, score_map=np.zeros_like(centered, dtype=np.float32))
    u, singular, vh = np.linalg.svd(centered.astype(np.float64, copy=False), full_matrices=False)
    r = max(1, min(int(rank), singular.size))
    low_rank = (u[:, :r] * singular[:r]) @ vh[:r, :]
    residual = _positive(centered - low_rank)
    activity = np.nanmax(residual, axis=0).astype(np.float32, copy=False)
    return CWTActivityResult(activity=activity, score_map=residual)


def _low_rank_residual(power: np.ndarray, rank: int) -> CWTActivityResult:
    return _low_rank_residual_from_score(_log_floor_ratio(power), rank=rank)


def _connected_component_activity(power: np.ndarray, quantile: float, min_area: int) -> CWTActivityResult:
    score = _positive(_row_mad_z(power))
    if score.size == 0 or not np.any(score):
        activity = np.zeros(score.shape[1], dtype=np.float32)
        return CWTActivityResult(activity=activity, score_map=score)
    threshold = max(1.5, float(np.nanquantile(score, min(max(float(quantile), 0.50), 0.999))))
    labels, count = ndimage.label(score > threshold, structure=np.ones((3, 3), dtype=bool))
    filtered = np.zeros_like(score, dtype=np.float32)
    if count > 0:
        objects = ndimage.find_objects(labels)
        for label, span in enumerate(objects, start=1):
            if span is None:
                continue
            period_slice, time_slice = span
            mask = labels[span] == label
            area = int(np.count_nonzero(mask))
            time_width = int(time_slice.stop - time_slice.start)
            if area >= int(min_area) and time_width >= 2:
                filtered[span] = np.where(mask, score[span], filtered[span])
    activity = np.nansum(filtered, axis=0).astype(np.float32, copy=False)
    return CWTActivityResult(activity=activity, score_map=filtered)


def _hough_horizontal_vote(power: np.ndarray, window: int) -> CWTActivityResult:
    score = _positive(_row_mad_z(power))
    if score.size == 0:
        activity = np.zeros(score.shape[1], dtype=np.float32)
        return CWTActivityResult(activity=activity, score_map=score)
    threshold = max(1.0, float(np.nanquantile(score, 0.95)))
    edges = (score > threshold).astype(np.float32)
    votes = ndimage.uniform_filter1d(edges, size=max(3, int(window)), axis=1, mode="nearest")
    evidence = (votes * score).astype(np.float32, copy=False)
    activity = np.nanmax(evidence, axis=0).astype(np.float32, copy=False)
    return CWTActivityResult(activity=activity, score_map=evidence)


def _catalog(*, include_rejected: bool = False) -> list[CWTActivityAlgorithm]:
    catalog = [
        CWTActivityAlgorithm(
            name="raw_max_power_ratio",
            family="energy_concentration",
            description="Direct max CWT power weighted by its period-axis energy share.",
            method="raw_max_power_ratio",
            source_refs=("energy_detector",),
            complexity="O(P*T)",
            migratability="Direct NumPy/GPU reduction; no learned parameters.",
        ),
        CWTActivityAlgorithm(
            name="row_mad_max_z",
            family="robust_row_extreme",
            description="Per-period robust row baseline, then max positive z-score per time.",
            method="row_mad_max_z",
            source_refs=("MAD/RFI robust thresholding",),
            complexity="O(P*T)",
            migratability="Existing robust statistics; cheap and streaming-friendly except row quantiles.",
        ),
        CWTActivityAlgorithm(
            name="row_mad_topk_mean",
            family="robust_topk",
            description="Mean of top-k positive per-period row-MAD CWT z-scores.",
            method="row_mad_topk_mean",
            source_refs=("MAD/RFI robust thresholding", "top-k_pooling"),
            complexity="O(P*T)",
            migratability="Partition-based reduction; no global floor-log positive projection.",
            params={"top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="row_mad_topk_ratio",
            family="robust_topk_ratio",
            description="Top-k positive row-MAD z-score mean weighted by its period-axis concentration.",
            method="row_mad_topk_ratio",
            source_refs=("MAD/RFI robust thresholding", "top-k_concentration"),
            complexity="O(P*T)",
            migratability="Partition-based reduction; no global floor-log positive projection.",
            params={"top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="period_band_scan_z",
            family="scan_statistic",
            description="Best contiguous period-band scan statistic on a row-robust CWT map.",
            method="period_band_scan_z",
            source_refs=("scan_statistic",),
            complexity="O(W*P*T)",
            migratability="Uses cumulative sums; cheap for small width set.",
            params={"widths": "1,3,5"},
        ),
        CWTActivityAlgorithm(
            name="row_mad_horizontal_filter",
            family="matched_filter",
            description="Horizontal time-persistence filter on per-period row-MAD CWT z-scores.",
            method="row_mad_horizontal_filter",
            source_refs=("time_frequency_matched_filter", "MAD/RFI robust thresholding"),
            complexity="O(P*T)",
            migratability="One separable uniform filter plus max reduction; no global floor-log positive projection.",
            params={"window": 33},
        ),
        CWTActivityAlgorithm(
            name="ridge_cfar_no_time",
            family="period_sideband_cfar",
            description="Period-sideband log-ratio ridge map without temporal coherence accumulation.",
            method="ridge_cfar",
            source_refs=("cell_averaging_CFAR", "morphological_top_hat"),
            complexity="O(P*T)",
            migratability="Two period-axis box filters and top-k reduction; streaming-friendly.",
            params={"inner_width": 3, "outer_width": 13, "time_support_cycles": 0.0, "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="ridge_cfar_2cycle",
            family="coherent_period_sideband_cfar",
            description="Local period-sideband CFAR followed by two-cycle signed coherence accumulation.",
            method="ridge_cfar",
            source_refs=("cell_averaging_CFAR", "anisotropic_ridge_filter"),
            complexity="O(P*T)",
            migratability="Separable box filters with period-dependent time support.",
            params={"inner_width": 3, "outer_width": 13, "time_support_cycles": 2.0, "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="ridge_cfar_4cycle",
            family="coherent_period_sideband_cfar",
            description="Local period-sideband CFAR followed by four-cycle signed coherence accumulation.",
            method="ridge_cfar",
            source_refs=("cell_averaging_CFAR", "anisotropic_ridge_filter"),
            complexity="O(P*T)",
            migratability="Separable box filters with period-dependent time support.",
            params={"inner_width": 3, "outer_width": 13, "time_support_cycles": 4.0, "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="ridge_cfar_6cycle",
            family="coherent_period_sideband_cfar",
            description="Local period-sideband CFAR followed by six-cycle signed coherence accumulation.",
            method="ridge_cfar",
            source_refs=("cell_averaging_CFAR", "anisotropic_ridge_filter"),
            complexity="O(P*T)",
            migratability="Separable box filters with period-dependent time support.",
            params={"inner_width": 3, "outer_width": 13, "time_support_cycles": 6.0, "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="ridge_cfar_wide_4cycle",
            family="coherent_period_sideband_cfar",
            description="Wider ridge passband with four-cycle signed coherence accumulation.",
            method="ridge_cfar",
            source_refs=("cell_averaging_CFAR", "anisotropic_ridge_filter"),
            complexity="O(P*T)",
            migratability="Separable box filters with period-dependent time support.",
            params={"inner_width": 5, "outer_width": 17, "time_support_cycles": 4.0, "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="freq_linear2_ridge_4cycle",
            family="frequency_common_mode_cfar",
            description="Nearest left/right frequency common-mode removal followed by the fixed four-cycle ridge compressor.",
            method="ridge_cfar",
            source_refs=("frequency_common_mode_rejection", "cell_averaging_CFAR"),
            complexity="O(C*T + P*T)",
            migratability="Two-neighbor arithmetic mean before the existing CWT path.",
            input_denoiser="neighbor_linear2",
            params={"inner_width": 3, "outer_width": 13, "time_support_cycles": 4.0, "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="freq_median4_ridge_4cycle",
            family="frequency_common_mode_cfar",
            description="Four-neighbor frequency median common-mode removal followed by the fixed four-cycle ridge compressor.",
            method="ridge_cfar",
            source_refs=("frequency_common_mode_rejection", "robust_median_filter", "cell_averaging_CFAR"),
            complexity="O(C*T + P*T)",
            migratability="Fixed four-neighbor sorting network or median before CWT.",
            input_denoiser="neighbor_median4",
            params={"inner_width": 3, "outer_width": 13, "time_support_cycles": 4.0, "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="freq_median8_ridge_4cycle",
            family="frequency_common_mode_cfar",
            description="Eight-neighbor frequency median common-mode removal followed by the fixed four-cycle ridge compressor.",
            method="ridge_cfar",
            source_refs=("frequency_common_mode_rejection", "robust_median_filter", "cell_averaging_CFAR"),
            complexity="O(C*T + P*T)",
            migratability="Small fixed-width frequency median before CWT.",
            input_denoiser="neighbor_median8",
            params={"inner_width": 3, "outer_width": 13, "time_support_cycles": 4.0, "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="post_freq_median8_4cycle",
            family="post_cwt_frequency_cfar",
            description="Post-CWT target-to-neighbor median log-power contrast with four-cycle accumulation.",
            method="post_cwt_frequency_cfar",
            source_refs=("frequency_cell_averaging_CFAR", "anisotropic_ridge_filter"),
            complexity="O(K*P*T)",
            migratability="Fixed eight-neighbor median and separable time filter on an existing CWT cube.",
            input_denoiser="post_cwt_neighbor8",
            params={"reference_statistic": "median", "mad_multiplier": 0.0, "time_support_cycles": 4.0, "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="post_freq_mad8_4cycle",
            family="post_cwt_frequency_cfar",
            description="Post-CWT neighbor median plus one MAD reference with four-cycle accumulation.",
            method="post_cwt_frequency_cfar",
            source_refs=("ordered_statistic_CFAR", "anisotropic_ridge_filter"),
            complexity="O(K*P*T)",
            migratability="Fixed eight-neighbor robust statistics and separable time filter.",
            input_denoiser="post_cwt_neighbor8",
            params={"reference_statistic": "median_mad", "mad_multiplier": 1.0, "time_support_cycles": 4.0, "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="post_freq_max8_4cycle",
            family="post_cwt_frequency_cfar",
            description="Post-CWT maximum-neighbor veto with four-cycle accumulation.",
            method="post_cwt_frequency_cfar",
            source_refs=("greatest_of_CFAR", "anisotropic_ridge_filter"),
            complexity="O(K*P*T)",
            migratability="One fixed-width neighbor max and separable time filter.",
            input_denoiser="post_cwt_neighbor8",
            params={"reference_statistic": "max", "mad_multiplier": 0.0, "time_support_cycles": 4.0, "min_positive_support": 0.0, "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="post_freq_max8_4cycle_s20",
            family="post_cwt_frequency_cfar",
            description="Maximum-neighbor veto with four-cycle accumulation and 20% persistence support.",
            method="post_cwt_frequency_cfar",
            source_refs=("greatest_of_CFAR", "persistence_detector"),
            complexity="O(K*P*T)",
            migratability="Neighbor max plus two separable time filters.",
            input_denoiser="post_cwt_neighbor8",
            params={"reference_statistic": "max", "mad_multiplier": 0.0, "time_support_cycles": 4.0, "min_positive_support": 0.20, "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="post_freq_max8_6cycle_s20",
            family="post_cwt_frequency_cfar",
            description="Maximum-neighbor veto with six-cycle accumulation and 20% persistence support.",
            method="post_cwt_frequency_cfar",
            source_refs=("greatest_of_CFAR", "persistence_detector"),
            complexity="O(K*P*T)",
            migratability="Neighbor max plus two separable time filters.",
            input_denoiser="post_cwt_neighbor8",
            params={"reference_statistic": "max", "mad_multiplier": 0.0, "time_support_cycles": 6.0, "min_positive_support": 0.20, "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="post_freq_max8_6cycle_s30",
            family="post_cwt_frequency_cfar",
            description="Maximum-neighbor veto with six-cycle accumulation and 30% persistence support.",
            method="post_cwt_frequency_cfar",
            source_refs=("greatest_of_CFAR", "persistence_detector"),
            complexity="O(K*P*T)",
            migratability="Neighbor max plus two separable time filters.",
            input_denoiser="post_cwt_neighbor8",
            params={"reference_statistic": "max", "mad_multiplier": 0.0, "time_support_cycles": 6.0, "min_positive_support": 0.30, "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="post_freq_max8_6cycle_s50",
            family="post_cwt_frequency_cfar",
            description="Maximum-neighbor veto with six-cycle accumulation and 50% persistence support.",
            method="post_cwt_frequency_cfar",
            source_refs=("greatest_of_CFAR", "persistence_detector"),
            complexity="O(K*P*T)",
            migratability="Neighbor max plus two separable time filters.",
            input_denoiser="post_cwt_neighbor8",
            params={"reference_statistic": "max", "mad_multiplier": 0.0, "time_support_cycles": 6.0, "min_positive_support": 0.50, "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="post_freq_max8_8cycle_s30",
            family="post_cwt_frequency_cfar",
            description="Maximum-neighbor veto with eight-cycle accumulation and 30% persistence support.",
            method="post_cwt_frequency_cfar",
            source_refs=("greatest_of_CFAR", "persistence_detector"),
            complexity="O(K*P*T)",
            migratability="Neighbor max plus two separable time filters.",
            input_denoiser="post_cwt_neighbor8",
            params={"reference_statistic": "max", "mad_multiplier": 0.0, "time_support_cycles": 8.0, "min_positive_support": 0.30, "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="post_freq_max8_8cycle_s50",
            family="post_cwt_frequency_cfar",
            description="Maximum-neighbor veto with eight-cycle accumulation and 50% persistence support.",
            method="post_cwt_frequency_cfar",
            source_refs=("greatest_of_CFAR", "persistence_detector"),
            complexity="O(K*P*T)",
            migratability="Neighbor max plus two separable time filters.",
            input_denoiser="post_cwt_neighbor8",
            params={"reference_statistic": "max", "mad_multiplier": 0.0, "time_support_cycles": 8.0, "min_positive_support": 0.50, "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="post_freq_max4_center_8cycle",
            family="adaptive_post_cwt_frequency_cfar",
            description="Centered four-neighbor CFAR with the simplified eight-cycle signed accumulator.",
            method="post_cwt_frequency_cfar",
            source_refs=("greatest_of_CFAR", "robust_map_calibration"),
            complexity="O(K*P*T)",
            migratability="Four reference maps, neighbor max, and one time filter.",
            input_denoiser="post_cwt_neighbor4",
            params={"reference_statistic": "max", "time_support_cycles": 8.0, "map_normalization": "center", "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="post_freq_max8_center_4cycle",
            family="adaptive_post_cwt_frequency_cfar",
            description="Centered eight-neighbor CFAR with a simplified four-cycle signed accumulator.",
            method="post_cwt_frequency_cfar",
            source_refs=("greatest_of_CFAR", "robust_map_calibration"),
            complexity="O(K*P*T)",
            migratability="Nine global map medians, neighbor max, and one time filter.",
            input_denoiser="post_cwt_neighbor8",
            params={"reference_statistic": "max", "time_support_cycles": 4.0, "map_normalization": "center", "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="post_freq_max8_center_6cycle",
            family="adaptive_post_cwt_frequency_cfar",
            description="Centered eight-neighbor CFAR with a simplified six-cycle signed accumulator.",
            method="post_cwt_frequency_cfar",
            source_refs=("greatest_of_CFAR", "robust_map_calibration"),
            complexity="O(K*P*T)",
            migratability="Nine global map medians, neighbor max, and one time filter.",
            input_denoiser="post_cwt_neighbor8",
            params={"reference_statistic": "max", "time_support_cycles": 6.0, "map_normalization": "center", "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="post_freq_max8_center_8cycle",
            family="simplified_post_cwt_frequency_cfar",
            description="Top1 core: centered eight-neighbor CFAR and one eight-cycle signed accumulator.",
            method="post_cwt_frequency_cfar",
            source_refs=("greatest_of_CFAR", "robust_map_calibration"),
            complexity="O(K*P*T)",
            migratability="Nine global map medians, neighbor max, and one time filter.",
            input_denoiser="post_cwt_neighbor8",
            params={"reference_statistic": "max", "time_support_cycles": 8.0, "map_normalization": "center", "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="post_freq_max8_center_12cycle",
            family="adaptive_post_cwt_frequency_cfar",
            description="Centered eight-neighbor CFAR with a simplified twelve-cycle signed accumulator.",
            method="post_cwt_frequency_cfar",
            source_refs=("greatest_of_CFAR", "robust_map_calibration"),
            complexity="O(K*P*T)",
            migratability="Nine global map medians, neighbor max, and one time filter.",
            input_denoiser="post_cwt_neighbor8",
            params={"reference_statistic": "max", "time_support_cycles": 12.0, "map_normalization": "center", "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="post_freq_max8_guard2_center_8cycle",
            family="adaptive_post_cwt_frequency_cfar",
            description="Simplified eight-reference CFAR with a two-channel signal guard on each side.",
            method="post_cwt_frequency_cfar",
            source_refs=("greatest_of_CFAR", "robust_map_calibration", "CFAR_guard_cells"),
            complexity="O(K*P*T)",
            migratability="Thirteen-channel input tile, eight-reference max, and one time filter.",
            input_denoiser="post_cwt_neighbor8_guard2",
            params={"reference_statistic": "max", "time_support_cycles": 8.0, "map_normalization": "center", "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="post_freq_max12_center_8cycle",
            family="adaptive_post_cwt_frequency_cfar",
            description="Centered twelve-neighbor CFAR with the simplified eight-cycle signed accumulator.",
            method="post_cwt_frequency_cfar",
            source_refs=("greatest_of_CFAR", "robust_map_calibration"),
            complexity="O(K*P*T)",
            migratability="Thirteen global map medians, neighbor max, and one time filter.",
            input_denoiser="post_cwt_neighbor12",
            params={"reference_statistic": "max", "time_support_cycles": 8.0, "map_normalization": "center", "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="post_freq_max8_center_8cycle_f20",
            family="simplified_post_cwt_frequency_cfar",
            description="Top1 without the 70% persistence weighting; retains the native 0.2 score floor.",
            method="post_cwt_frequency_cfar",
            source_refs=("greatest_of_CFAR", "robust_map_calibration"),
            complexity="O(K*P*T)",
            migratability="Nine global map medians, neighbor max, one time filter, and one native floor.",
            input_denoiser="post_cwt_neighbor8",
            params={"reference_statistic": "max", "time_support_cycles": 8.0, "score_floor": 0.20, "map_normalization": "center", "clip_log_ratio": 1.5, "top_k": 3},
        ),
        *_top1_single_floor_candidates(),
        *_top1_continuous_support_candidates(),
        CWTActivityAlgorithm(
            name="post_freq_max8_center_8cycle_s70",
            family="simplified_post_cwt_frequency_cfar",
            description="Top1 without the native 0.2 score floor; retains 70% persistence weighting.",
            method="post_cwt_frequency_cfar",
            source_refs=("greatest_of_CFAR", "robust_map_calibration", "persistence_detector"),
            complexity="O(K*P*T)",
            migratability="Nine global map medians, neighbor max, and two time filters.",
            input_denoiser="post_cwt_neighbor8",
            params={"reference_statistic": "max", "time_support_cycles": 8.0, "min_positive_support": 0.70, "map_normalization": "center", "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="post_freq_max8_center_8cycle_s70_f20",
            family="calibrated_post_cwt_frequency_cfar",
            description="Globally median-centered maximum-neighbor CFAR with eight-cycle, 70% persistence and 0.2 native floor.",
            method="post_cwt_frequency_cfar",
            source_refs=("greatest_of_CFAR", "robust_map_calibration", "persistence_detector"),
            complexity="O(K*P*T)",
            migratability="Nine global map medians, neighbor max, and two separable time filters.",
            input_denoiser="post_cwt_neighbor8",
            params={"reference_statistic": "max", "mad_multiplier": 0.0, "time_support_cycles": 8.0, "min_positive_support": 0.70, "score_floor": 0.20, "map_normalization": "center", "clip_log_ratio": 1.5, "top_k": 3},
        ),
        CWTActivityAlgorithm(
            name="horizontal_matched_filter",
            family="matched_filter",
            description="Legacy horizontal filter on global floor-log CWT power; kept as an unsafe contrast candidate.",
            method="horizontal_matched_filter",
            source_refs=("time_frequency_matched_filter",),
            complexity="O(P*T)",
            migratability="Fast but prone to brightening natural low-frequency texture.",
            params={"window": 33},
        ),
        CWTActivityAlgorithm(
            name="row_mad_viterbi_path",
            family="ridge_dp",
            description="Dynamic-programming ridge path through per-period row-MAD CWT z-scores.",
            method="row_mad_viterbi_path",
            source_refs=("Iatsenko2013_ridge_dynamic_path", "ssqueezepy_ridge_extraction", "MAD/RFI robust thresholding"),
            complexity="O(P*T*R)",
            migratability="Small-radius DP; avoids global floor-log brightening.",
            params={"radius": 2, "penalty": 0.20},
        ),
        CWTActivityAlgorithm(
            name="viterbi_ridge_path",
            family="ridge_dp",
            description="Legacy dynamic-programming ridge path through global floor-log CWT power.",
            method="viterbi_ridge_path",
            source_refs=("Iatsenko2013_ridge_dynamic_path", "ssqueezepy_ridge_extraction"),
            complexity="O(P*T*R)",
            migratability="Small-radius DP, but global floor-log preprocessing is unsafe on raw low-frequency CE4 texture.",
            params={"radius": 2, "penalty": 0.20},
        ),
        CWTActivityAlgorithm(
            name="spectral_entropy_deficit",
            family="information_entropy",
            description="Period-distribution Shannon entropy deficit for each time slice.",
            method="spectral_entropy_deficit",
            source_refs=("Renyi_spectral_change", "spectral_entropy_RFI"),
            complexity="O(P*T)",
            migratability="Pure reductions; no thresholds beyond numerical floor.",
        ),
        CWTActivityAlgorithm(
            name="js_background_divergence",
            family="information_divergence",
            description="Jensen-Shannon divergence from the median period distribution background.",
            method="js_background_divergence",
            source_refs=("Renyi_spectral_change", "spectral_relative_entropy_RFI"),
            complexity="O(P*T)",
            migratability="Pure reductions; stable bounded divergence.",
        ),
        CWTActivityAlgorithm(
            name="period_gini_concentration",
            family="information_concentration",
            description="Gini concentration of each raw CWT period distribution time slice.",
            method="period_gini_concentration",
            source_refs=("spectral_concentration", "Gini_index"),
            complexity="O(P*T*log(P))",
            migratability="Sort-based period-axis reduction; no global floor-log positive projection.",
        ),
        CWTActivityAlgorithm(
            name="row_mad_kurtosis_window",
            family="higher_moment",
            description="Stabilized windowed excess-kurtosis of per-period row-MAD CWT z-scores.",
            method="row_mad_kurtosis_window",
            source_refs=("spectral_kurtosis_RFI", "MAD/RFI robust thresholding"),
            complexity="O(P*T)",
            migratability="Four uniform filters; no global floor-log positive projection.",
            params={"window": 33},
        ),
        CWTActivityAlgorithm(
            name="spectral_kurtosis_window",
            family="higher_moment",
            description="Legacy stabilized windowed excess-kurtosis of global floor-log CWT power.",
            method="spectral_kurtosis_window",
            source_refs=("spectral_kurtosis_RFI",),
            complexity="O(P*T)",
            migratability="Fast but unsafe on raw low-frequency CE4 texture because floor-log creates natural bright spots.",
            params={"window": 33},
        ),
        CWTActivityAlgorithm(
            name="row_mad_svd_rank1",
            family="low_rank_projection",
            description="First SVD mode projection of per-period row-MAD CWT z-score map.",
            method="row_mad_svd_rank1",
            source_refs=("PCA_low_rank", "MAD/RFI robust thresholding"),
            complexity="O(min(P,T)^2*max(P,T))",
            migratability="Cheap because P is small; less suitable for streaming; avoids floor-log brightening.",
        ),
        CWTActivityAlgorithm(
            name="svd_rank1_projection",
            family="low_rank_projection",
            description="Legacy first SVD mode projection of centered global floor-log CWT map.",
            method="svd_rank1_projection",
            source_refs=("PCA_low_rank",),
            complexity="O(min(P,T)^2*max(P,T))",
            migratability="Cheap because P is small, but floor-log preprocessing is unsafe on raw CE4 negatives.",
        ),
        CWTActivityAlgorithm(
            name="row_mad_low_rank_residual",
            family="low_rank_residual",
            description="Rank-1 row-MAD CWT background subtraction followed by max positive residual.",
            method="row_mad_low_rank_residual",
            source_refs=("Candes2009_RPCA", "Lin2010_ALM_RPCA", "MAD/RFI robust thresholding"),
            complexity="O(min(P,T)^2*max(P,T))",
            migratability="SVD-only approximation of RPCA; avoids global floor-log brightening.",
            params={"rank": 1},
        ),
        CWTActivityAlgorithm(
            name="low_rank_residual_max",
            family="low_rank_residual",
            description="Legacy rank-1 global floor-log CWT background subtraction followed by max positive residual.",
            method="low_rank_residual_max",
            source_refs=("Candes2009_RPCA", "Lin2010_ALM_RPCA"),
            complexity="O(min(P,T)^2*max(P,T))",
            migratability="SVD-only approximation of RPCA, but floor-log preprocessing is unsafe on raw CE4 negatives.",
            params={"rank": 1},
        ),
        CWTActivityAlgorithm(
            name="connected_component_mass",
            family="image_segmentation",
            description="Connected-component mass after robust row thresholding in the period-time image.",
            method="connected_component_mass",
            source_refs=("time_frequency_image_segmentation",),
            complexity="O(P*T)",
            migratability="Uses scipy.ndimage label; easy to port to GPU connected components if needed.",
            params={"quantile": 0.975, "min_area": 6},
        ),
        CWTActivityAlgorithm(
            name="hough_horizontal_vote",
            family="hough_vote",
            description="Horizontal line-vote accumulator for persistent ridge-like period-time pixels.",
            method="hough_horizontal_vote",
            source_refs=("Hough_line_transform", "scikit_image_hough_line"),
            complexity="O(P*T)",
            migratability="Restricted zero-slope Hough vote; much cheaper than full angle accumulator.",
            params={"window": 65},
        ),
    ]
    from single_map_activity_algorithms import (
        single_map_absolute_persistence_catalog,
        single_map_catalog,
    )

    catalog += single_map_catalog(CWTActivityAlgorithm)
    if include_rejected:
        catalog += single_map_absolute_persistence_catalog(CWTActivityAlgorithm)
    return catalog


def resolve_cwt_activity_algorithms(names: tuple[str, ...] | list[str]) -> list[CWTActivityAlgorithm]:
    requested = tuple(str(name).strip() for name in names if str(name).strip())
    if not requested or any(name.lower() == "all" for name in requested):
        return _catalog()
    catalog = _catalog(include_rejected=True)
    by_name = {algorithm.name: algorithm for algorithm in catalog}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"Unknown CWT activity algorithms: {', '.join(sorted(missing))}")
    return [by_name[name] for name in requested]


def cwt_activity_algorithm_map() -> dict[str, dict[str, Any]]:
    return {algorithm.name: asdict(algorithm) for algorithm in resolve_cwt_activity_algorithms(("all",))}


def compute_cwt_activity(
    power: np.ndarray,
    periods: np.ndarray,
    algorithm: CWTActivityAlgorithm,
    reference_power: np.ndarray | None = None,
    *,
    noise_std: float | None = None,
    noise_gain: np.ndarray | None = None,
) -> CWTActivityResult:
    method = algorithm.method
    params = algorithm.params
    if method == "single_map_coherent":
        if reference_power is not None:
            raise ValueError("strict single-map algorithms reject reference_power")
        from single_map_activity_algorithms import compute_single_map_activity

        return compute_single_map_activity(power, periods, algorithm, CWTActivityResult)
    if method == "single_map_absolute_persistence":
        if reference_power is not None:
            raise ValueError("strict single-map algorithms reject reference_power")
        from single_map_activity_algorithms import compute_absolute_persistence_activity

        return compute_absolute_persistence_activity(power, algorithm, CWTActivityResult)
    if method == "single_map_cpro_activity":
        if reference_power is not None:
            raise ValueError("CPRO activity rank rejects reference_power")
        if noise_std is None or noise_gain is None:
            raise ValueError("CPRO activity rank requires physical noise_std and wavelet noise_gain")
        from cwipss.signal.cpro import CPROParameters, cpro_activity

        params = CPROParameters(**algorithm.params)
        result = cpro_activity(
            power,
            noise_std=float(noise_std),
            noise_gain=np.asarray(noise_gain, dtype=np.float32),
            params=params,
        )
        return CWTActivityResult(
            activity=np.asarray(result.activity, dtype=np.float32),
            score_map=np.asarray(result.score_map, dtype=np.float32),
        )
    if method == "raw_max_power_ratio":
        score = _finite_power(power)
        max_values = np.nanmax(score, axis=0)
        total = np.maximum(np.nansum(score, axis=0), MIN_POSITIVE)
        activity = (max_values * np.clip(max_values / total, 0.0, 1.0)).astype(np.float32, copy=False)
        return CWTActivityResult(activity=activity, score_map=score)
    if method == "row_mad_max_z":
        score = _positive(_row_mad_z(power))
        activity = np.nanmax(score, axis=0).astype(np.float32, copy=False)
        return CWTActivityResult(activity=activity, score_map=score)
    if method == "row_mad_topk_mean":
        score = _positive(_row_mad_z(power))
        k = max(1, min(int(params.get("top_k", 3)), score.shape[0]))
        top = np.partition(score, kth=score.shape[0] - k, axis=0)[-k:, :]
        activity = np.nanmean(top, axis=0).astype(np.float32, copy=False)
        return CWTActivityResult(activity=activity, score_map=score)
    if method == "row_mad_topk_ratio":
        score = _positive(_row_mad_z(power))
        k = max(1, min(int(params.get("top_k", 3)), score.shape[0]))
        top = np.partition(score, kth=score.shape[0] - k, axis=0)[-k:, :]
        top_mean = np.nanmean(top, axis=0)
        top_sum = np.nansum(top, axis=0)
        total = np.maximum(np.nansum(score, axis=0), MIN_POSITIVE)
        activity = (top_mean * np.clip(top_sum / total, 0.0, 1.0)).astype(np.float32, copy=False)
        return CWTActivityResult(activity=activity, score_map=score)
    if method == "period_band_scan_z":
        row_score = _positive(_row_mad_z(power))
        best_activity = np.full(row_score.shape[1], -np.inf, dtype=np.float32)
        best_map = np.zeros_like(row_score, dtype=np.float32)
        for width_text in str(params.get("widths", "1,3,5")).split(","):
            activity, score_map = _best_band_score(row_score, int(width_text))
            update = activity > best_activity
            best_activity = np.where(update, activity, best_activity)
            best_map[:, update] = score_map[:, update]
        best_activity[~np.isfinite(best_activity)] = 0.0
        return CWTActivityResult(activity=_positive(best_activity), score_map=best_map)
    if method == "row_mad_horizontal_filter":
        score = _row_mad_z(power)
        filtered = ndimage.uniform_filter1d(score, size=max(3, int(params.get("window", 33))), axis=1, mode="nearest")
        filtered = _positive(filtered)
        activity = np.nanmax(filtered, axis=0).astype(np.float32, copy=False)
        return CWTActivityResult(activity=activity, score_map=filtered)
    if method == "ridge_cfar":
        return _coherent_ridge_cfar(
            power,
            periods,
            inner_width=int(params.get("inner_width", 3)),
            outer_width=int(params.get("outer_width", 13)),
            time_support_cycles=float(params.get("time_support_cycles", 4.0)),
            clip_log_ratio=float(params.get("clip_log_ratio", 1.5)),
            top_k=int(params.get("top_k", 3)),
            min_positive_support=float(params.get("min_positive_support", 0.0)),
            support_weighting=str(params.get("support_weighting", "none")),
            score_floor=float(params.get("score_floor", 0.0)),
            period_pooling=str(params.get("period_pooling", "topk_mean")),
            period_guard_bins=int(params.get("period_guard_bins", 2)),
            concentration_power=float(params.get("concentration_power", 1.0)),
            winner_support_cycles=float(params.get("winner_support_cycles", 4.0)),
        )
    if method == "post_cwt_frequency_cfar":
        if reference_power is None:
            raise ValueError(f"{algorithm.name} requires neighboring-channel CWT power")
        return _post_cwt_frequency_cfar(
            power,
            reference_power,
            periods,
            reference_statistic=str(params.get("reference_statistic", "median")),
            mad_multiplier=float(params.get("mad_multiplier", 1.0)),
            time_support_cycles=float(params.get("time_support_cycles", 4.0)),
            clip_log_ratio=float(params.get("clip_log_ratio", 1.5)),
            top_k=int(params.get("top_k", 3)),
            min_positive_support=float(params.get("min_positive_support", 0.0)),
            support_weighting=str(params.get("support_weighting", "none")),
            calibration_quantile=float(params.get("calibration_quantile", 0.0)),
            period_background_width=int(params.get("period_background_width", 0)),
            map_normalization=str(params.get("map_normalization", "none")),
            score_floor=float(params.get("score_floor", 0.0)),
            max_positive_fraction=float(params.get("max_positive_fraction", 0.0)),
        )
    if method == "horizontal_matched_filter":
        score = _log_floor_ratio(power)
        filtered = ndimage.uniform_filter1d(score, size=max(3, int(params.get("window", 33))), axis=1, mode="nearest")
        filtered = _positive(filtered)
        activity = np.nanmax(filtered, axis=0).astype(np.float32, copy=False)
        return CWTActivityResult(activity=activity, score_map=filtered)
    if method == "row_mad_viterbi_path":
        return _viterbi_score_path(
            _row_mad_z(power),
            radius=int(params.get("radius", 2)),
            penalty=float(params.get("penalty", 0.20)),
        )
    if method == "viterbi_ridge_path":
        return _viterbi_activity(
            power,
            radius=int(params.get("radius", 2)),
            penalty=float(params.get("penalty", 0.20)),
        )
    if method == "spectral_entropy_deficit":
        p = _period_distribution(power)
        entropy = -np.sum(p * np.log(np.maximum(p, MIN_POSITIVE)), axis=0)
        norm = math.log(max(2, p.shape[0]))
        activity = np.clip(1.0 - entropy / max(norm, MIN_POSITIVE), 0.0, 1.0).astype(np.float32, copy=False)
        score_map = (p * activity[None, :]).astype(np.float32, copy=False)
        return CWTActivityResult(activity=activity, score_map=score_map)
    if method == "js_background_divergence":
        p = _period_distribution(power)
        q = np.nanmedian(p, axis=1, keepdims=True)
        q = q / np.maximum(np.sum(q, axis=0, keepdims=True), MIN_POSITIVE)
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * (np.log(np.maximum(p, MIN_POSITIVE)) - np.log(np.maximum(m, MIN_POSITIVE))), axis=0)
        kl_qm = np.sum(q * (np.log(np.maximum(q, MIN_POSITIVE)) - np.log(np.maximum(m, MIN_POSITIVE))), axis=0)
        activity = np.maximum(0.5 * (kl_pm + kl_qm), 0.0).astype(np.float32, copy=False)
        score_map = (np.abs(p - q) * activity[None, :]).astype(np.float32, copy=False)
        return CWTActivityResult(activity=activity, score_map=score_map)
    if method == "period_gini_concentration":
        values = _finite_power(power) + MIN_POSITIVE
        activity = _gini(values)
        p = values / np.maximum(np.sum(values, axis=0, keepdims=True), MIN_POSITIVE)
        score_map = (p * activity[None, :]).astype(np.float32, copy=False)
        return CWTActivityResult(activity=activity, score_map=score_map)
    if method == "row_mad_kurtosis_window":
        score = _window_kurtosis_score(_row_mad_z(power), window=int(params.get("window", 33)))
        activity = np.nanmax(score, axis=0).astype(np.float32, copy=False)
        return CWTActivityResult(activity=activity, score_map=score)
    if method == "spectral_kurtosis_window":
        score = _spectral_kurtosis(power, window=int(params.get("window", 33)))
        activity = np.nanmax(score, axis=0).astype(np.float32, copy=False)
        return CWTActivityResult(activity=activity, score_map=score)
    if method == "row_mad_svd_rank1":
        return _rank1_projection_from_score(_row_mad_z(power))
    if method == "svd_rank1_projection":
        return _rank1_projection(power)
    if method == "row_mad_low_rank_residual":
        return _low_rank_residual_from_score(_row_mad_z(power), rank=int(params.get("rank", 1)))
    if method == "low_rank_residual_max":
        return _low_rank_residual(power, rank=int(params.get("rank", 1)))
    if method == "connected_component_mass":
        return _connected_component_activity(
            power,
            quantile=float(params.get("quantile", 0.975)),
            min_area=int(params.get("min_area", 6)),
        )
    if method == "hough_horizontal_vote":
        return _hough_horizontal_vote(power, window=int(params.get("window", 65)))
    raise ValueError(f"Unknown CWT activity method: {method}")


def smooth_cwt_activity_result(result: CWTActivityResult, smooth_records: int) -> CWTActivityResult:
    width = max(1, int(smooth_records))
    if width <= 1:
        return result
    return CWTActivityResult(
        activity=smooth_activity(result.activity, smooth_records=width),
        score_map=result.score_map,
    )
