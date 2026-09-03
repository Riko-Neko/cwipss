#!/usr/bin/env python3
"""Compare CPRO and direct period reductions on the fixed 1,993-case review set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import signal


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cwipss.config import load_cwt_config  # noqa: E402
from cwipss.signal.activity import robust_standardize  # noqa: E402
from cwipss.signal.cpro import (  # noqa: E402
    CPROParameters,
    cpro_activity,
    cpro_continuity_map,
    cpro_period_mask,
    impulse_cwt_noise_gain,
)
from cwipss.signal.cwt import cwt_power_cube, period_grid_records  # noqa: E402
from cwipss.signal.detection import pelt_windows_from_activity  # noqa: E402
from cwipss.signal.windows import (  # noqa: E402
    active_windows_from_segments,
    merge_close_windows,
    pelt_mean_shift,
)


EXPECTED_CASES = 1993
EPS = float(np.finfo(np.float32).tiny)
RESULT_FIELDS = (
    "review_rank",
    "raw_key",
    "label",
    "algorithm",
    "window_count",
    "best_anchor_iou",
    "truth_available",
    "truth_interval_count",
    "truth_fully_observed",
    "truth_hit",
    "best_truth_iou",
    "truth_coverage",
    "left_error_rec",
    "right_error_rec",
    "priority_truth_available",
    "priority_truth_interval_count",
    "priority_truth_fully_observed",
    "priority_truth_hit",
    "priority_best_truth_iou",
    "priority_truth_coverage",
    "priority_left_error_rec",
    "priority_right_error_rec",
    "fp_interval_count",
    "fp_interval_hit_count",
    "best_t0_rec",
    "best_t1_rec",
    "best_dur_rec",
    "activity_seconds",
    "pelt_seconds",
)
WINDOW_FIELDS = (
    "review_rank",
    "raw_key",
    "algorithm",
    "window_index",
    "t0_rec",
    "t1_rec",
    "dur_rec",
    "activity_mean",
    "activity_max",
    "continuity_mean",
    "continuity_fill",
    "continuity_smooth",
    "continuity_occupancy",
    "continuity_log_mean",
    "period_occupancy",
    "period_log_mean",
    "period_energy_lock",
    "period_profile_coherence",
    "temporal_contrast",
    "anchor_iou",
    "truth_iou",
    "priority_truth_iou",
)


@dataclass(frozen=True)
class PeltSpec:
    penalty: float
    min_size: int
    jump: int
    segment_min_duration: int
    min_duration: int
    min_mean: float
    merge_gap: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=BASE_DIR / "selection.csv")
    parser.add_argument("--labels", type=Path, default=BASE_DIR / "labels.csv")
    parser.add_argument(
        "--archive",
        type=Path,
        default=BASE_DIR / "artifacts/single_channel_slices.npz",
    )
    parser.add_argument("--metadata", type=Path, default=BASE_DIR / "artifacts/metadata.json")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/cwt_default.json")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "artifacts/activity_comparison")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--priority-only",
        action="store_true",
        help="Evaluate every high-confidence Real case and omit the remaining cases",
    )
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--pelt-penalty", type=float, default=None)
    parser.add_argument("--pelt-min-size", type=int, default=None)
    parser.add_argument("--segment-min-duration", type=int, default=None)
    parser.add_argument("--window-min-duration", type=int, default=None)
    parser.add_argument("--merge-gap", type=int, default=None)
    parser.add_argument(
        "--continuity-grid",
        action="store_true",
        help="Rank experimental scale-free CPRO time-continuity weights",
    )
    parser.add_argument(
        "--continuity-gate-grid",
        action="store_true",
        help="Keep raw CPRO PELT boundaries and gate segments by absolute continuity",
    )
    parser.add_argument(
        "--continuity-gate-refine",
        action="store_true",
        help="Refine the selected decay=0.995, exponent=2 continuity gate",
    )
    parser.add_argument(
        "--continuity-gate-alt",
        action="store_true",
        help="Compare weaker continuity exponents around the selected frontier",
    )
    parser.add_argument(
        "--continuity-feature-export",
        action="store_true",
        help="Export ungated PELT segments with scale-free continuity features",
    )
    parser.add_argument(
        "--production-continuity",
        action="store_true",
        help="Replay the exact cutoff-free continuity implementation imported from src",
    )
    parser.add_argument(
        "--assert-production-target",
        action="store_true",
        help="Fail unless the full production replay preserves the selected scientific frontier",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalized_period_integral(
    values: np.ndarray,
    periods: np.ndarray,
    *,
    period_exponent: float = 0.0,
) -> np.ndarray:
    measure = np.power(periods, float(period_exponent), dtype=np.float64)
    return (
        np.trapezoid(values * measure[:, None], x=periods, axis=0)
        / max(float(np.trapezoid(measure, x=periods)), EPS)
    ).astype(np.float32, copy=False)


def _exponential_mean(values: np.ndarray, decay: float) -> np.ndarray:
    """Apply a boundary-initialized first-order IIR along time in O(P*T)."""
    data = np.asarray(values, dtype=np.float32)
    coefficient = 1.0 - float(decay)
    initial = float(decay) * data[:, :1]
    filtered, _state = signal.lfilter(
        [coefficient],
        [1.0, -float(decay)],
        data,
        axis=1,
        zi=initial,
    )
    return filtered.astype(np.float32, copy=False)


def continuity_weighted_map(
    shape_map: np.ndarray,
    *,
    decay: float,
    exponent: float,
) -> np.ndarray:
    """Suppress time-isolated responses while retaining individual period rows."""
    values = np.maximum(np.asarray(shape_map, dtype=np.float32), 0.0)
    forward = _exponential_mean(values, decay)
    backward = _exponential_mean(values[:, ::-1], decay)[:, ::-1]
    support = np.sqrt(np.maximum(forward * backward, 0.0))
    ratio = np.divide(
        np.minimum(support, values),
        values,
        out=np.zeros_like(values),
        where=values > 0.0,
    )
    filtered = values * np.power(ratio, float(exponent), dtype=np.float32)
    return filtered.astype(np.float32, copy=False)


def continuity_weighted_activity(
    shape_map: np.ndarray,
    *,
    decay: float,
    exponent: float,
) -> np.ndarray:
    """Reduce cutoff-free period-row continuity evidence to the time axis."""
    return np.max(
        continuity_weighted_map(shape_map, decay=decay, exponent=exponent),
        axis=0,
    ).astype(np.float32, copy=False)


def pelt_windows(activity: np.ndarray, spec: PeltSpec) -> list[dict]:
    activity_z = robust_standardize(np.asarray(activity, dtype=np.float32))
    segments = pelt_mean_shift(
        activity_z,
        penalty=spec.penalty,
        min_size=spec.min_size,
        jump=spec.jump,
    )
    windows = active_windows_from_segments(
        segments,
        activity_z,
        min_duration=spec.segment_min_duration,
        min_mean=spec.min_mean,
    )
    windows = merge_close_windows(windows, max_gap=spec.merge_gap)
    return [
        row
        for row in windows
        if int(row["record_stop"]) - int(row["record_start"]) >= spec.min_duration
    ]


def continuity_segment_windows(
    boundary_activity: np.ndarray,
    continuity_activity: np.ndarray,
    *,
    calibration_threshold: float,
    spec: PeltSpec,
) -> list[dict]:
    """Describe raw CPRO PELT segments with continuous temporal-shape metrics."""
    boundary = np.asarray(boundary_activity, dtype=np.float32)
    evidence_input = np.asarray(continuity_activity, dtype=np.float32)
    evidence_map = evidence_input if evidence_input.ndim == 2 else None
    evidence = (
        np.max(evidence_input, axis=0).astype(np.float32, copy=False)
        if evidence_map is not None
        else evidence_input
    )
    activity_z = robust_standardize(boundary)
    segments = pelt_mean_shift(
        activity_z,
        penalty=spec.penalty,
        min_size=spec.min_size,
        jump=spec.jump,
    )
    windows = active_windows_from_segments(
        segments,
        activity_z,
        min_duration=spec.segment_min_duration,
        min_mean=spec.min_mean,
    )
    active_indices = [
        index
        for index, segment in enumerate(segments)
        if segment.duration >= spec.segment_min_duration and segment.mean >= spec.min_mean
    ]
    temporal_contrast_by_start: dict[int, float] = {}
    cursor = 0
    while cursor < len(active_indices):
        group = [active_indices[cursor]]
        cursor += 1
        while cursor < len(active_indices) and active_indices[cursor] == group[-1] + 1:
            group.append(active_indices[cursor])
            cursor += 1
        first_index, last_index = group[0], group[-1]
        group_start = int(segments[first_index].start)
        group_stop = int(segments[last_index].stop)
        inside_mean = float(np.mean(boundary[group_start:group_stop]))
        references = []
        if first_index > 0:
            left = segments[first_index - 1]
            references.append(float(np.mean(boundary[left.start:left.stop])))
        if last_index + 1 < len(segments):
            right = segments[last_index + 1]
            references.append(float(np.mean(boundary[right.start:right.stop])))
        reference = max(references, default=inside_mean)
        contrast = inside_mean / max(reference, EPS)
        for segment_index in group:
            temporal_contrast_by_start[int(segments[segment_index].start)] = contrast
    described = []
    denominator = max(float(calibration_threshold), EPS)
    for window in windows:
        start = int(window["record_start"])
        stop = int(window["record_stop"])
        values = np.maximum(evidence[start:stop], 0.0)
        normalized = values / denominator
        if evidence_map is None:
            period_occupancy = float(np.mean(normalized / (1.0 + normalized)))
            period_log_mean = float(np.mean(np.log1p(normalized)))
            period_energy_lock = 1.0
            period_profile_coherence = 1.0
        else:
            period_values = np.maximum(evidence_map[:, start:stop], 0.0)
            period_normalized = period_values / denominator
            period_occupancy = float(
                np.max(np.mean(period_normalized / (1.0 + period_normalized), axis=1))
            )
            period_log_mean = float(np.max(np.mean(np.log1p(period_normalized), axis=1)))
            period_energy_lock = float(
                np.max(np.sum(period_values, axis=1, dtype=np.float64))
                / max(float(np.sum(np.max(period_values, axis=0), dtype=np.float64)), EPS)
            )
            if period_values.shape[1] <= 1:
                period_profile_coherence = 1.0
            else:
                left = period_values[:, :-1]
                right = period_values[:, 1:]
                numerator = float(np.sum(left * right, dtype=np.float64))
                denominator_profile = float(
                    np.sum(
                        np.sqrt(
                            np.sum(np.square(left), axis=0, dtype=np.float64)
                            * np.sum(np.square(right), axis=0, dtype=np.float64)
                        ),
                        dtype=np.float64,
                    )
                )
                period_profile_coherence = numerator / max(denominator_profile, EPS)
        total = float(np.sum(values, dtype=np.float64))
        squared = float(np.sum(np.square(values), dtype=np.float64))
        duration = max(stop - start, 1)
        fill = total * total / max(float(duration) * squared, EPS)
        variation = (
            abs(float(values[0]))
            + float(np.sum(np.abs(np.diff(values)), dtype=np.float64))
            + abs(float(values[-1]))
        )
        smooth = np.clip(1.0 - variation / max(2.0 * total, EPS), 0.0, 1.0)
        described.append(
            {
                **window,
                "shape_activity_mean": float(np.mean(boundary[start:stop])),
                "shape_activity_max": float(np.max(boundary[start:stop])),
                "continuity_mean": float(np.mean(values)) / denominator,
                "continuity_fill": float(fill),
                "continuity_smooth": float(smooth),
                "continuity_occupancy": float(np.mean(normalized / (1.0 + normalized))),
                "continuity_log_mean": float(np.mean(np.log1p(normalized))),
                "period_occupancy": period_occupancy,
                "period_log_mean": period_log_mean,
                "period_energy_lock": period_energy_lock,
                "period_profile_coherence": period_profile_coherence,
                "temporal_contrast": temporal_contrast_by_start.get(start, 1.0),
            }
        )
    return described


def continuity_gated_windows(
    boundary_activity: np.ndarray,
    continuity_activity: np.ndarray,
    *,
    calibration_threshold: float,
    continuity_min: float,
    spec: PeltSpec,
) -> list[dict]:
    """Keep raw CPRO boundaries while selecting segments by absolute continuity."""
    boundary = np.asarray(boundary_activity, dtype=np.float32)
    evidence_input = np.asarray(continuity_activity, dtype=np.float32)
    evidence_map = evidence_input if evidence_input.ndim == 2 else None
    evidence = (
        np.max(evidence_input, axis=0).astype(np.float32, copy=False)
        if evidence_map is not None
        else evidence_input
    )
    denominator = max(float(calibration_threshold), EPS)
    selected = [
        window
        for window in continuity_segment_windows(
            boundary,
            evidence_input,
            calibration_threshold=calibration_threshold,
            spec=spec,
        )
        if float(window["continuity_mean"]) >= float(continuity_min)
    ]
    merged = merge_close_windows(selected, max_gap=spec.merge_gap)
    accepted = []
    for window in merged:
        start = int(window["record_start"])
        stop = int(window["record_stop"])
        if stop - start < spec.min_duration:
            continue
        values = np.maximum(evidence[start:stop], 0.0)
        normalized = values / denominator
        if evidence_map is None:
            period_occupancy = float(np.mean(normalized / (1.0 + normalized)))
            period_log_mean = float(np.mean(np.log1p(normalized)))
            period_energy_lock = 1.0
            period_profile_coherence = 1.0
        else:
            period_values = np.maximum(evidence_map[:, start:stop], 0.0)
            period_normalized = period_values / denominator
            period_occupancy = float(
                np.max(np.mean(period_normalized / (1.0 + period_normalized), axis=1))
            )
            period_log_mean = float(np.max(np.mean(np.log1p(period_normalized), axis=1)))
            period_energy_lock = float(
                np.max(np.sum(period_values, axis=1, dtype=np.float64))
                / max(float(np.sum(np.max(period_values, axis=0), dtype=np.float64)), EPS)
            )
            if period_values.shape[1] <= 1:
                period_profile_coherence = 1.0
            else:
                left = period_values[:, :-1]
                right = period_values[:, 1:]
                numerator = float(np.sum(left * right, dtype=np.float64))
                denominator_profile = float(
                    np.sum(
                        np.sqrt(
                            np.sum(np.square(left), axis=0, dtype=np.float64)
                            * np.sum(np.square(right), axis=0, dtype=np.float64)
                        ),
                        dtype=np.float64,
                    )
                )
                period_profile_coherence = numerator / max(denominator_profile, EPS)
        total = float(np.sum(values, dtype=np.float64))
        squared = float(np.sum(np.square(values), dtype=np.float64))
        duration = max(stop - start, 1)
        variation = (
            abs(float(values[0]))
            + float(np.sum(np.abs(np.diff(values)), dtype=np.float64))
            + abs(float(values[-1]))
        )
        accepted.append(
            {
                **window,
                "shape_activity_mean": float(np.mean(boundary[start:stop])),
                "shape_activity_max": float(np.max(boundary[start:stop])),
                "continuity_mean": float(np.mean(values)) / denominator,
                "continuity_fill": total * total / max(float(duration) * squared, EPS),
                "continuity_smooth": float(
                    np.clip(1.0 - variation / max(2.0 * total, EPS), 0.0, 1.0)
                ),
                "continuity_occupancy": float(np.mean(normalized / (1.0 + normalized))),
                "continuity_log_mean": float(np.mean(np.log1p(normalized))),
                "period_occupancy": period_occupancy,
                "period_log_mean": period_log_mean,
                "period_energy_lock": period_energy_lock,
                "period_profile_coherence": period_profile_coherence,
                "temporal_contrast": float(window.get("temporal_contrast", 1.0)),
            }
        )
    return accepted


def iou(start_a: int, stop_a: int, start_b: int, stop_b: int) -> float:
    overlap = max(0, min(stop_a, stop_b) - max(start_a, start_b))
    union = max(stop_a, stop_b) - min(start_a, start_b)
    return float(overlap) / float(max(union, 1))


def interval_coverage(windows: list[tuple[int, int]], truth_start: int, truth_stop: int) -> float:
    clipped = sorted(
        (max(start, truth_start), min(stop, truth_stop))
        for start, stop in windows
        if min(stop, truth_stop) > max(start, truth_start)
    )
    covered = 0
    cursor_start = cursor_stop = None
    for start, stop in clipped:
        if cursor_start is None:
            cursor_start, cursor_stop = start, stop
        elif start <= cursor_stop:
            cursor_stop = max(cursor_stop, stop)
        else:
            covered += cursor_stop - cursor_start
            cursor_start, cursor_stop = start, stop
    if cursor_start is not None:
        covered += cursor_stop - cursor_start
    return float(covered) / float(max(truth_stop - truth_start, 1))


def label_intervals(row: dict[str, str]) -> list[dict[str, object]]:
    value = str(row.get("intervals", "")).strip()
    if not value:
        return []
    payload = json.loads(value)
    intervals = [
        {
            "t0": int(interval["t0"]),
            "t1": int(interval["t1"]),
            "lc": int(interval.get("lc", 0)),
            "rc": int(interval.get("rc", 0)),
            "label": str(interval.get("label", "")).strip().lower(),
            "conf": str(interval.get("conf", "")).strip().lower(),
        }
        for interval in payload
    ]
    if any(interval["t1"] <= interval["t0"] for interval in intervals):
        return []
    return intervals


def case_label(intervals: list[dict[str, object]]) -> str:
    labels = {str(interval.get("label", "")) for interval in intervals if interval.get("label")}
    if len(labels) == 1:
        return next(iter(labels))
    return "mixed" if labels else ""


def multi_interval_coverage(
    windows: list[tuple[int, int]],
    intervals: list[dict[str, object]],
) -> float:
    total = sum(interval["t1"] - interval["t0"] for interval in intervals)
    if total <= 0:
        return 0.0
    covered = sum(
        interval_coverage(windows, interval["t0"], interval["t1"])
        * (interval["t1"] - interval["t0"])
        for interval in intervals
    )
    return float(covered) / float(total)


def evaluate_truth_windows(
    ranked: list[tuple[float, int, int, int]],
    intervals: list[dict[str, object]],
) -> dict[str, object]:
    has_truth = bool(intervals)
    fully_observed = bool(
        has_truth and all(not interval["lc"] and not interval["rc"] for interval in intervals)
    )
    best = max(ranked, default=None)
    coverage = (
        multi_interval_coverage(
            [(start, stop) for _metric, start, stop, _truth_index in ranked],
            intervals,
        )
        if has_truth
        else 0.0
    )
    matched = intervals[best[3]] if has_truth and best and best[3] >= 0 else None
    return {
        "available": int(has_truth),
        "interval_count": len(intervals),
        "fully_observed": int(fully_observed),
        "hit": int(bool(has_truth and best and best[0] > 0.0)),
        "best_iou": float(best[0]) if has_truth and best else 0.0,
        "coverage": coverage,
        "left_error": (
            int(best[1] - matched["t0"])
            if matched is not None and not matched["lc"]
            else ""
        ),
        "right_error": (
            int(best[2] - matched["t1"])
            if matched is not None and not matched["rc"]
            else ""
        ),
        "best": best,
    }


def target_summary(rows: list[dict[str, object]], field_prefix: str = "") -> dict[str, object]:
    key = lambda name: f"{field_prefix}{name}"
    targets = [row for row in rows if int(row[key("truth_available")])]
    if not targets:
        return {}
    hits = np.asarray([int(row[key("truth_hit")]) for row in targets])
    coverage = np.asarray([float(row[key("truth_coverage")]) for row in targets])
    full_targets = [row for row in targets if int(row[key("truth_fully_observed")])]
    left_error = np.asarray(
        [float(row[key("left_error_rec")]) for row in targets if str(row[key("left_error_rec")]) != ""]
    )
    right_error = np.asarray(
        [float(row[key("right_error_rec")]) for row in targets if str(row[key("right_error_rec")]) != ""]
    )
    metrics: dict[str, object] = {
        "cases": len(targets),
        "hit_count": int(np.sum(hits)),
        "fully_observed_cases": len(full_targets),
        "any_overlap_rate": float(np.mean(hits)),
        "median_coverage": float(np.median(coverage)),
        "coverage_50_rate": float(np.mean(coverage >= 0.50)),
        "left_bias_records": float(np.mean(left_error)) if left_error.size else None,
        "right_bias_records": float(np.mean(right_error)) if right_error.size else None,
        "left_mae_records": float(np.mean(np.abs(left_error))) if left_error.size else None,
        "right_mae_records": float(np.mean(np.abs(right_error))) if right_error.size else None,
        "left_median_ae_records": float(np.median(np.abs(left_error))) if left_error.size else None,
        "right_median_ae_records": float(np.median(np.abs(right_error))) if right_error.size else None,
        "boundary_mae_records": (
            float(np.mean(np.concatenate((np.abs(left_error), np.abs(right_error)))))
            if left_error.size and right_error.size
            else None
        ),
        "boundary_median_ae_records": (
            float(np.median(np.concatenate((np.abs(left_error), np.abs(right_error)))))
            if left_error.size and right_error.size
            else None
        ),
    }
    if full_targets:
        truth_iou = np.asarray([float(row[key("best_truth_iou")]) for row in full_targets])
        metrics.update(
            hit_iou_10_rate=float(np.mean(truth_iou >= 0.10)),
            hit_iou_30_rate=float(np.mean(truth_iou >= 0.30)),
            hit_iou_50_rate=float(np.mean(truth_iou >= 0.50)),
            median_iou=float(np.median(truth_iou)),
        )
    return metrics


def paired_comparisons(
    results: list[dict[str, object]],
    field_prefix: str = "",
) -> dict[str, object]:
    key = lambda name: f"{field_prefix}{name}"
    indexed = {
        (str(row["algorithm"]), str(row["raw_key"])): row
        for row in results
        if int(row[key("truth_available")]) and int(row[key("truth_fully_observed")])
    }
    target_keys = sorted({case for algorithm, case in indexed if algorithm == "cpro"})
    paired: dict[str, object] = {}
    for reference in ("direct_mean", "direct_integral"):
        deltas = []
        wins = ties = losses = 0
        for case in target_keys:
            candidate = indexed.get(("cpro", case))
            baseline = indexed.get((reference, case))
            if candidate is None or baseline is None:
                continue
            delta = float(candidate[key("best_truth_iou")]) - float(
                baseline[key("best_truth_iou")]
            )
            deltas.append(delta)
            if delta > 1e-12:
                wins += 1
            elif delta < -1e-12:
                losses += 1
            else:
                ties += 1
        if deltas:
            paired[f"cpro_vs_{reference}"] = {
                "target_cases": len(deltas),
                "median_iou_delta": float(np.median(deltas)),
                "mean_iou_delta": float(np.mean(deltas)),
                "iou_wins": wins,
                "iou_ties": ties,
                "iou_losses": losses,
            }
    return paired


def summarize(results: list[dict[str, object]], windows: list[dict[str, object]]) -> dict[str, object]:
    by_algorithm: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in results:
        by_algorithm[str(row["algorithm"])].append(row)
    summary: dict[str, object] = {}
    for algorithm, rows in sorted(by_algorithm.items()):
        counts = np.asarray([int(row["window_count"]) for row in rows])
        anchor_iou = np.asarray([float(row["best_anchor_iou"]) for row in rows])
        durations = np.asarray(
            [float(row["best_dur_rec"]) for row in rows if str(row["best_dur_rec"]) != ""],
            dtype=np.float64,
        )
        false_cases = [row for row in rows if str(row["label"]).lower() == "fp"]
        item: dict[str, object] = {
            "cases": len(rows),
            "total_windows": int(np.sum(counts)),
            "cases_with_windows": int(np.count_nonzero(counts)),
            "case_window_rate": float(np.mean(counts > 0)),
            "median_best_anchor_iou": float(np.median(anchor_iou)),
            "median_best_window_duration": float(np.median(durations)) if durations.size else None,
            "activity_seconds": float(sum(float(row["activity_seconds"]) for row in rows)),
            "pelt_seconds": float(sum(float(row["pelt_seconds"]) for row in rows)),
        }
        item.update({f"target_{name}": value for name, value in target_summary(rows).items()})
        item.update(
            {
                f"priority_target_{name}": value
                for name, value in target_summary(rows, "priority_").items()
            }
        )
        if false_cases:
            false_case_window_count = int(
                np.count_nonzero([int(row["window_count"]) > 0 for row in false_cases])
            )
            item.update(
                false_cases=len(false_cases),
                false_case_window_count=false_case_window_count,
                false_case_window_rate=float(
                    np.mean([int(row["window_count"]) > 0 for row in false_cases])
                ),
            )
        fp_interval_count = sum(int(row["fp_interval_count"]) for row in rows)
        fp_interval_hit_count = sum(int(row["fp_interval_hit_count"]) for row in rows)
        item.update(
            fp_interval_count=fp_interval_count,
            fp_interval_hit_count=fp_interval_hit_count,
            fp_interval_hit_rate=(
                float(fp_interval_hit_count / fp_interval_count)
                if fp_interval_count
                else None
            ),
        )
        summary[algorithm] = item

    return {
        "algorithms": summary,
        "paired_target_comparisons": paired_comparisons(results),
        "priority_paired_target_comparisons": paired_comparisons(results, "priority_"),
    }


def rank_parity_robustness(results: list[dict[str, object]]) -> dict[str, object]:
    """Report deterministic half-splits without changing the fixed evaluation set."""
    return {
        f"review_rank_mod2_{parity}": summarize(
            [row for row in results if int(row["review_rank"]) % 2 == parity],
            [],
        )["algorithms"]
        for parity in (0, 1)
    }


def main() -> None:
    args = parse_args()
    if args.assert_production_target and not args.production_continuity:
        raise ValueError("--assert-production-target requires --production-continuity")
    selection = read_rows(args.selection)
    label_rows = read_rows(args.labels)
    labels = {row["raw_key"]: row for row in label_rows}
    metadata = {
        str(row["raw_key"]): row
        for row in json.loads(args.metadata.read_text(encoding="utf-8"))
    }
    selection_keys = [row["raw_key"] for row in selection]
    if (
        len(selection) != EXPECTED_CASES
        or len(label_rows) != EXPECTED_CASES
        or len(labels) != EXPECTED_CASES
        or len(metadata) != EXPECTED_CASES
        or len(set(selection_keys)) != EXPECTED_CASES
        or set(selection_keys) != set(labels)
        or set(selection_keys) != set(metadata)
    ):
        raise RuntimeError("the comparison requires the fixed 1,993-case review set")
    if args.limit > 0:
        selection = selection[: args.limit]
    if args.priority_only:
        selection = [
            row
            for row in selection
            if any(
                interval["label"] == "keep" and interval["conf"] == "high"
                for interval in label_intervals(labels[row["raw_key"]])
            )
        ]

    config = load_cwt_config(args.config, overrides={"cwt_backend": args.backend})
    full_periods = period_grid_records(
        config.period_min_records,
        config.period_max_records,
        config.period_count,
        config.period_spacing,
    )
    mask = cpro_period_mask(
        full_periods,
        config.candidate_period_min_records,
        config.candidate_period_max_records,
    )
    periods = full_periods[mask]
    noise_gain = impulse_cwt_noise_gain(periods, wavelet=config.wavelet, method=config.cwt_method)
    params = CPROParameters(
        threshold_snr=config.cpro_threshold_snr,
        texture_quantile=0.0,
        period_center_bins=config.cpro_period_center_bins,
        period_context_bins=config.cpro_period_context_bins,
        min_period_contrast=config.cpro_min_period_contrast,
        period_support_bins=config.cpro_period_support_bins,
        shape_power_softness=config.cpro_shape_power_softness,
        shape_contrast_softness=config.cpro_shape_contrast_softness,
    )
    old_pelt = PeltSpec(
        config.pelt_penalty,
        config.pelt_min_size_records,
        config.pelt_jump_records,
        96,
        640,
        config.window_min_activity_mean,
        config.window_merge_gap_records,
    )
    shared_pelt = replace(
        old_pelt,
        penalty=old_pelt.penalty if args.pelt_penalty is None else max(0.0, args.pelt_penalty),
        min_size=(old_pelt.min_size if args.pelt_min_size is None else max(1, args.pelt_min_size)),
        segment_min_duration=(
            old_pelt.segment_min_duration
            if args.segment_min_duration is None
            else max(1, args.segment_min_duration)
        ),
        min_duration=(
            old_pelt.min_duration
            if args.window_min_duration is None
            else max(1, args.window_min_duration)
        ),
        merge_gap=(old_pelt.merge_gap if args.merge_gap is None else max(0, args.merge_gap)),
    )

    results: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    cwt_seconds = 0.0
    with np.load(args.archive) as archive:
        if len(archive.files) != EXPECTED_CASES or set(archive.files) != set(selection_keys):
            raise RuntimeError("the raw archive must contain exactly 1,993 slices")
        for case_index, row in enumerate(selection, 1):
            key = row["raw_key"]
            raw = np.asarray(archive[key], dtype=np.float32)
            if raw.ndim != 1 or not np.all(np.isfinite(raw)):
                raise ValueError(f"invalid raw slice: {key}")
            started = perf_counter()
            power = cwt_power_cube(
                raw[:, None],
                periods,
                wavelet=config.wavelet,
                normalize_channels=False,
                method=config.cwt_method,
                backend=args.backend,
                cuda_device=args.cuda_device,
            )[:, :, 0]
            cwt_seconds += perf_counter() - started
            noise_std = float(row["noise_sigma"])
            threshold = float(row["cpro_thr"])
            denominator = np.maximum(noise_std * noise_std * noise_gain[:, None], EPS)
            calibrated = np.maximum(power, 0.0) / denominator

            activity_started = perf_counter()
            cpro_result = cpro_activity(
                power,
                noise_std=noise_std,
                noise_gain=noise_gain,
                params=replace(params, threshold_snr=threshold),
            )
            cpro = cpro_result.shape_activity
            cpro_seconds = perf_counter() - activity_started
            activity_started = perf_counter()
            direct_mean = np.mean(calibrated, axis=0, dtype=np.float32)
            mean_seconds = perf_counter() - activity_started
            activity_started = perf_counter()
            # The logarithmic period grid is nonuniform. Normalize the
            # trapezoidal integral to retain width weighting in power units.
            direct_integral = normalized_period_integral(calibrated, periods)
            integral_seconds = perf_counter() - activity_started
            algorithms = [("cpro", cpro, shared_pelt, cpro_seconds, None, None)]
            if args.production_continuity:
                activity_started = perf_counter()
                evidence = cpro_continuity_map(
                    cpro_result.shape_map,
                    decay=config.cpro_continuity_decay,
                    power=config.cpro_continuity_power,
                )
                activity_seconds = perf_counter() - activity_started
                algorithms.append(
                    (
                        "cpro_continuity_production",
                        cpro,
                        shared_pelt,
                        activity_seconds,
                        evidence,
                        "production",
                    )
                )
            elif args.continuity_grid:
                for decay in (0.95, 0.98, 0.99, 0.995):
                    for exponent in (1.0, 2.0, 4.0):
                        activity_started = perf_counter()
                        activity = continuity_weighted_activity(
                            cpro_result.shape_map,
                            decay=decay,
                            exponent=exponent,
                        )
                        activity_seconds = perf_counter() - activity_started
                        decay_name = int(round(decay * 1000.0))
                        exponent_name = int(round(exponent))
                        algorithms.append(
                            (
                                f"cpro_tc_d{decay_name:03d}_g{exponent_name}",
                                activity,
                                shared_pelt,
                                activity_seconds,
                                None,
                                None,
                            )
                        )
            elif (
                args.continuity_gate_grid
                or args.continuity_gate_refine
                or args.continuity_gate_alt
                or args.continuity_feature_export
            ):
                filter_specs = (
                    ((0.995, 2.0),)
                    if args.continuity_gate_refine or args.continuity_feature_export
                    else ((0.99, 2.0),)
                    if args.continuity_gate_alt
                    else ((0.99, 4.0), (0.995, 2.0), (0.995, 4.0))
                )
                gate_values = (
                    (-1.0,)
                    if args.continuity_feature_export
                    else (0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65)
                    if args.continuity_gate_refine
                    else (0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85)
                    if args.continuity_gate_alt
                    else (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
                )
                for decay, exponent in filter_specs:
                    activity_started = perf_counter()
                    evidence = continuity_weighted_map(
                        cpro_result.shape_map,
                        decay=decay,
                        exponent=exponent,
                    )
                    activity_seconds = perf_counter() - activity_started
                    decay_name = int(round(decay * 1000.0))
                    exponent_name = int(round(exponent))
                    for gate_index, continuity_min in enumerate(gate_values):
                        algorithm_name = (
                            f"cpro_seg_d{decay_name:03d}_g{exponent_name}"
                            if continuity_min < 0.0
                            else f"cpro_tg_d{decay_name:03d}_g{exponent_name}_m"
                            f"{int(round(100 * continuity_min)):02d}"
                        )
                        algorithms.append(
                            (
                                algorithm_name,
                                cpro,
                                shared_pelt,
                                activity_seconds if gate_index == 0 else 0.0,
                                evidence,
                                continuity_min,
                            )
                        )
            else:
                algorithms.extend(
                    (
                        ("direct_mean", direct_mean, shared_pelt, mean_seconds, None, None),
                        ("direct_integral", direct_integral, shared_pelt, integral_seconds, None, None),
                    )
                )
            offset = int(metadata[key]["extract_t0_rec"])
            anchor_start, anchor_stop = int(row["t0_rec"]), int(row["t1_rec"])
            annotations = label_intervals(labels[key])
            truth_intervals = [
                interval for interval in annotations if interval["label"] == "keep"
            ]
            priority_truth_intervals = [
                interval
                for interval in truth_intervals
                if interval["conf"] == "high"
            ]
            fp_intervals = [
                interval for interval in annotations if interval["label"] == "fp"
            ]
            annotation_label = case_label(annotations)
            for name, activity, pelt_spec, activity_seconds, gate_activity, gate_min in algorithms:
                pelt_started = perf_counter()
                if gate_activity is None:
                    windows = pelt_windows(activity, pelt_spec)
                elif gate_min == "production":
                    windows = pelt_windows_from_activity(
                        activity,
                        gate_activity,
                        calibrated_threshold=threshold,
                        penalty=pelt_spec.penalty,
                        min_size=pelt_spec.min_size,
                        jump=pelt_spec.jump,
                        min_mean=pelt_spec.min_mean,
                        min_continuity_mean=config.cpro_min_continuity_mean,
                        min_ridge_lock=config.cpro_min_ridge_lock,
                        merge_gap=pelt_spec.merge_gap,
                    )[0]
                elif float(gate_min) < 0.0:
                    windows = continuity_segment_windows(
                        activity,
                        gate_activity,
                        calibration_threshold=threshold,
                        spec=pelt_spec,
                    )
                else:
                    windows = continuity_gated_windows(
                        activity,
                        gate_activity,
                        calibration_threshold=threshold,
                        continuity_min=float(gate_min),
                        spec=pelt_spec,
                    )
                pelt_seconds = perf_counter() - pelt_started
                ranked = []
                priority_ranked = []
                for window_index, window in enumerate(windows, 1):
                    start = offset + int(window["record_start"])
                    stop = offset + int(window["record_stop"])
                    anchor_iou = iou(start, stop, anchor_start, anchor_stop)
                    truth_scores = [
                        iou(start, stop, interval["t0"], interval["t1"])
                        for interval in truth_intervals
                    ]
                    priority_truth_scores = [
                        iou(start, stop, interval["t0"], interval["t1"])
                        for interval in priority_truth_intervals
                    ]
                    truth_index = int(np.argmax(truth_scores)) if truth_scores else -1
                    priority_truth_index = (
                        int(np.argmax(priority_truth_scores)) if priority_truth_scores else -1
                    )
                    truth_iou = truth_scores[truth_index] if truth_scores else 0.0
                    priority_truth_iou = (
                        priority_truth_scores[priority_truth_index]
                        if priority_truth_scores
                        else 0.0
                    )
                    ranked.append(
                        (truth_iou if truth_intervals else anchor_iou, start, stop, truth_index)
                    )
                    priority_ranked.append(
                        (
                            priority_truth_iou if priority_truth_intervals else anchor_iou,
                            start,
                            stop,
                            priority_truth_index,
                        )
                    )
                    window_rows.append(
                        {
                            "review_rank": int(row["review_rank"]),
                            "raw_key": key,
                            "algorithm": name,
                            "window_index": window_index,
                            "t0_rec": start,
                            "t1_rec": stop,
                            "dur_rec": stop - start,
                            "activity_mean": float(window.get("shape_activity_mean", 0.0)),
                            "activity_max": float(window.get("shape_activity_max", 0.0)),
                            "continuity_mean": float(window.get("continuity_mean", 0.0)),
                            "continuity_fill": float(window.get("continuity_fill", 0.0)),
                            "continuity_smooth": float(window.get("continuity_smooth", 0.0)),
                            "continuity_occupancy": float(
                                window.get("continuity_occupancy", 0.0)
                            ),
                            "continuity_log_mean": float(
                                window.get("continuity_log_mean", 0.0)
                            ),
                            "period_occupancy": float(window.get("period_occupancy", 0.0)),
                            "period_log_mean": float(window.get("period_log_mean", 0.0)),
                            "period_energy_lock": float(
                                window.get("period_energy_lock", window.get("ridge_lock", 0.0))
                            ),
                            "period_profile_coherence": float(
                                window.get("period_profile_coherence", 0.0)
                            ),
                            "temporal_contrast": float(window.get("temporal_contrast", 1.0)),
                            "anchor_iou": anchor_iou,
                            "truth_iou": truth_iou,
                            "priority_truth_iou": priority_truth_iou,
                        }
                    )
                best_anchor_iou = max(
                    (
                        iou(start, stop, anchor_start, anchor_stop)
                        for _metric, start, stop, _truth_index in ranked
                    ),
                    default=0.0,
                )
                truth_metrics = evaluate_truth_windows(ranked, truth_intervals)
                priority_metrics = evaluate_truth_windows(
                    priority_ranked,
                    priority_truth_intervals,
                )
                emitted_spans = [
                    (start, stop) for _metric, start, stop, _truth_index in ranked
                ]
                fp_interval_hit_count = sum(
                    interval_coverage(
                        emitted_spans,
                        int(interval["t0"]),
                        int(interval["t1"]),
                    )
                    > 0.0
                    for interval in fp_intervals
                )
                best = truth_metrics["best"]
                results.append(
                    {
                        "review_rank": int(row["review_rank"]),
                        "raw_key": key,
                        "label": annotation_label,
                        "algorithm": name,
                        "window_count": len(windows),
                        "best_anchor_iou": best_anchor_iou,
                        "truth_available": truth_metrics["available"],
                        "truth_interval_count": truth_metrics["interval_count"],
                        "truth_fully_observed": truth_metrics["fully_observed"],
                        "truth_hit": truth_metrics["hit"],
                        "best_truth_iou": truth_metrics["best_iou"],
                        "truth_coverage": truth_metrics["coverage"],
                        "left_error_rec": truth_metrics["left_error"],
                        "right_error_rec": truth_metrics["right_error"],
                        "priority_truth_available": priority_metrics["available"],
                        "priority_truth_interval_count": priority_metrics["interval_count"],
                        "priority_truth_fully_observed": priority_metrics["fully_observed"],
                        "priority_truth_hit": priority_metrics["hit"],
                        "priority_best_truth_iou": priority_metrics["best_iou"],
                        "priority_truth_coverage": priority_metrics["coverage"],
                        "priority_left_error_rec": priority_metrics["left_error"],
                        "priority_right_error_rec": priority_metrics["right_error"],
                        "fp_interval_count": len(fp_intervals),
                        "fp_interval_hit_count": fp_interval_hit_count,
                        "best_t0_rec": int(best[1]) if best else "",
                        "best_t1_rec": int(best[2]) if best else "",
                        "best_dur_rec": int(best[2] - best[1]) if best else "",
                        "activity_seconds": float(activity_seconds),
                        "pelt_seconds": float(pelt_seconds),
                    }
                )
            if case_index == 1 or case_index % 25 == 0 or case_index == len(selection):
                print(f"[compare] cases={case_index}/{len(selection)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "case_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    with (args.output_dir / "windows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WINDOW_FIELDS)
        writer.writeheader()
        writer.writerows(window_rows)
    label_counts = defaultdict(int)
    invalid_label_intervals = 0
    for row in label_rows:
        annotations = label_intervals(row)
        labels_in_case = [str(interval["label"]) for interval in annotations]
        complete = bool(annotations) and all(labels_in_case)
        label_name = case_label(annotations) if complete else "unlabelled"
        label_counts[label_name] += 1
        if annotations and any(
            label not in {"keep", "fp", "uncertain"} for label in labels_in_case
        ):
            invalid_label_intervals += 1
    labelled_cases = EXPECTED_CASES - label_counts["unlabelled"]
    formal_metrics_ready = (
        len(selection) == EXPECTED_CASES
        and labelled_cases == EXPECTED_CASES
        and not (set(label_counts) - {"keep", "fp", "uncertain", "mixed", "unlabelled"})
        and invalid_label_intervals == 0
    )
    summary = {
        "dataset": "fixed manual CPRF review set",
        "metric_source": "only the fixed 1,993 extracted real single-channel cases",
        "source_cases": EXPECTED_CASES,
        "evaluated_cases": len(selection),
        "full_dataset_evaluation": len(selection) == EXPECTED_CASES,
        "review_labels": {
            "counts": dict(sorted(label_counts.items())),
            "labelled_cases": labelled_cases,
            "invalid_label_intervals": invalid_label_intervals,
            "formal_metrics_ready": formal_metrics_ready,
        },
        "primary_target": {
            "interval_label": "keep",
            "confidence": "high",
            "metric_prefix": "priority_target_",
        },
        "cwt_seconds": cwt_seconds,
        "historical_duration_baseline_pelt": shared_pelt.__dict__,
        "production_continuity": {
            "implementation": "cwipss.signal.cpro + cwipss.signal.detection",
            "decay": config.cpro_continuity_decay,
            "power": config.cpro_continuity_power,
            "min_continuity_mean": config.cpro_min_continuity_mean,
            "min_ridge_lock": config.cpro_min_ridge_lock,
            "pelt_min_size_records": shared_pelt.min_size,
            "pelt_min_activity_mean": shared_pelt.min_mean,
            "merge_gap_records": shared_pelt.merge_gap,
            "independent_duration_gate": None,
        },
        "cpro": {
            "period_center_bins": params.period_center_bins,
            "period_context_bins": params.period_context_bins,
            "min_period_contrast": params.min_period_contrast,
            "period_support_bins": params.period_support_bins,
            "shape_power_softness": params.shape_power_softness,
            "shape_contrast_softness": params.shape_contrast_softness,
            "period_reduction": "maximum",
            "time_smoothing": "none",
        },
        **summarize(results, window_rows),
        "rank_parity_robustness": rank_parity_robustness(results),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    if args.assert_production_target:
        if not formal_metrics_ready:
            raise AssertionError("production acceptance requires the complete labelled 1,993 cases")
        production = summary["algorithms"]["cpro_continuity_production"]
        parity = summary["rank_parity_robustness"]
        checks = {
            "high_confidence_cases": production["priority_target_cases"] == 313,
            "high_confidence_hits": production["priority_target_hit_count"] >= 304,
            "high_confidence_recall": production["priority_target_any_overlap_rate"] >= 0.97,
            "fp_interval_hits": production["fp_interval_hit_count"] <= 68,
            "pure_fp_case_retention": production["false_case_window_count"] <= 61,
            "median_iou": production["priority_target_median_iou"] >= 0.91,
            "even_recall": parity["review_rank_mod2_0"]["cpro_continuity_production"][
                "priority_target_any_overlap_rate"
            ]
            >= 0.96,
            "odd_recall": parity["review_rank_mod2_1"]["cpro_continuity_production"][
                "priority_target_any_overlap_rate"
            ]
            >= 0.96,
        }
        summary["production_acceptance"] = {"passed": all(checks.values()), "checks": checks}
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise AssertionError(f"production continuity acceptance failed: {', '.join(failed)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
