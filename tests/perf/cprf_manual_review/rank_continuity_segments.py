#!/usr/bin/env python3
"""Rank cutoff-free temporal-continuity gates on the fixed manual review set."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows",
        type=Path,
        default=BASE_DIR / "artifacts/cpro_ridge_continuity_features_v1/windows.csv",
    )
    parser.add_argument("--labels", type=Path, default=BASE_DIR / "labels.csv")
    parser.add_argument("--min-recall", type=float, default=0.97)
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args()


def overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return min(a1, b1) > max(a0, b0)


def iou(a0: int, a1: int, b0: int, b1: int) -> float:
    common = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return common / max(union, 1)


def merge_adjacent(rows: list[dict[str, object]]) -> list[tuple[int, int]]:
    spans: list[list[int]] = []
    for row in rows:
        start, stop = int(row["t0"]), int(row["t1"])
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], stop)
        else:
            spans.append([start, stop])
    return [(start, stop) for start, stop in spans]


def evaluate(
    cases: dict[str, list[dict[str, object]]],
    labels: dict[str, list[dict[str, object]]],
    *,
    mean_min: float,
    fill_min: float,
    smooth_min: float,
    occupancy_min: float = 0.0,
    log_mean_min: float = 0.0,
    period_occupancy_min: float = 0.0,
    period_log_mean_min: float = 0.0,
    period_coherence_min: float = 0.0,
    shape_fill_min: float = 0.0,
    period_energy_lock_min: float = 0.0,
    period_profile_coherence_min: float = 0.0,
    score_min: float = 0.0,
    fill_power: float = 0.0,
    smooth_power: float = 0.0,
) -> dict[str, float]:
    selected: dict[str, list[tuple[int, int]]] = {}
    for key, rows in cases.items():
        kept = [
            row
            for row in rows
            if float(row["mean"]) >= mean_min
            and float(row["fill"]) >= fill_min
            and float(row["smooth"]) >= smooth_min
            and float(row["occupancy"]) >= occupancy_min
            and float(row["log_mean"]) >= log_mean_min
            and float(row["period_occupancy"]) >= period_occupancy_min
            and float(row["period_log_mean"]) >= period_log_mean_min
            and float(row["period_coherence"]) >= period_coherence_min
            and float(row["shape_fill"]) >= shape_fill_min
            and float(row["period_energy_lock"]) >= period_energy_lock_min
            and float(row["period_profile_coherence"]) >= period_profile_coherence_min
            and float(row["mean"])
            * float(row["fill"]) ** fill_power
            * float(row["smooth"]) ** smooth_power
            >= score_min
        ]
        selected[key] = merge_adjacent(kept)

    high_hits: list[int] = []
    high_ious: list[float] = []
    boundary_errors: list[float] = []
    fp_hits: list[int] = []
    pure_fp_hits: list[int] = []
    parity_high: dict[int, list[int]] = defaultdict(list)
    parity_fp: dict[int, list[int]] = defaultdict(list)
    parity_pure_fp: dict[int, list[int]] = defaultdict(list)
    for key, intervals in labels.items():
        spans = selected[key]
        rank = int(cases[key][0]["rank"])
        parity = rank % 2
        high = [
            item
            for item in intervals
            if item.get("label") == "keep" and item.get("conf") == "high"
        ]
        false = [item for item in intervals if item.get("label") == "fp"]
        if high:
            hit = int(
                any(
                    overlap(start, stop, int(item["t0"]), int(item["t1"]))
                    for start, stop in spans
                    for item in high
                )
            )
            high_hits.append(hit)
            parity_high[parity].append(hit)
            fully_observed = all(not item.get("lc", 0) and not item.get("rc", 0) for item in high)
            if fully_observed:
                matches = [
                    (iou(start, stop, int(item["t0"]), int(item["t1"])), start, stop, item)
                    for start, stop in spans
                    for item in high
                ]
                best = max(matches, default=None)
                high_ious.append(best[0] if best else 0.0)
                if best:
                    boundary_errors.extend(
                        (abs(best[1] - int(best[3]["t0"])), abs(best[2] - int(best[3]["t1"])))
                    )
        for item in false:
            hit = int(
                any(
                    overlap(start, stop, int(item["t0"]), int(item["t1"]))
                    for start, stop in spans
                )
            )
            fp_hits.append(hit)
            parity_fp[parity].append(hit)
        if false and len(false) == len(intervals):
            hit = int(bool(spans))
            pure_fp_hits.append(hit)
            parity_pure_fp[parity].append(hit)

    result = {
        "mean_min": mean_min,
        "fill_min": fill_min,
        "smooth_min": smooth_min,
        "occupancy_min": occupancy_min,
        "log_mean_min": log_mean_min,
        "period_occupancy_min": period_occupancy_min,
        "period_log_mean_min": period_log_mean_min,
        "period_coherence_min": period_coherence_min,
        "shape_fill_min": shape_fill_min,
        "period_energy_lock_min": period_energy_lock_min,
        "period_profile_coherence_min": period_profile_coherence_min,
        "score_min": score_min,
        "fill_power": fill_power,
        "smooth_power": smooth_power,
        "real_recall": float(np.mean(high_hits)),
        "fp_interval_rate": float(np.mean(fp_hits)),
        "pure_fp_rate": float(np.mean(pure_fp_hits)),
        "median_iou": float(np.median(high_ious)),
        "boundary_median_ae": float(np.median(boundary_errors)),
    }
    for parity in (0, 1):
        result[f"real_recall_p{parity}"] = float(np.mean(parity_high[parity]))
        result[f"fp_interval_rate_p{parity}"] = float(np.mean(parity_fp[parity]))
        result[f"pure_fp_rate_p{parity}"] = float(np.mean(parity_pure_fp[parity]))
    return result


def main() -> None:
    args = parse_args()
    labels = {
        row["raw_key"]: json.loads(row["intervals"])
        for row in csv.DictReader(args.labels.open(newline="", encoding="utf-8"))
    }
    cases: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in csv.DictReader(args.windows.open(newline="", encoding="utf-8")):
        if row["algorithm"] != "cpro_seg_d995_g2":
            continue
        cases[row["raw_key"]].append(
            {
                "rank": int(row["review_rank"]),
                "t0": int(row["t0_rec"]),
                "t1": int(row["t1_rec"]),
                "mean": float(row["continuity_mean"]),
                "fill": float(row["continuity_fill"]),
                "smooth": float(row["continuity_smooth"]),
                "occupancy": float(row["continuity_occupancy"]),
                "log_mean": float(row["continuity_log_mean"]),
                "period_occupancy": float(row["period_occupancy"]),
                "period_log_mean": float(row["period_log_mean"]),
                "period_coherence": float(row["period_occupancy"])
                / max(float(row["continuity_occupancy"]), np.finfo(np.float64).tiny),
                "shape_fill": float(row["activity_mean"])
                / max(float(row["activity_max"]), np.finfo(np.float64).tiny),
                "period_energy_lock": float(row["period_energy_lock"]),
                "period_profile_coherence": float(row["period_profile_coherence"]),
            }
        )
    if set(cases) != set(labels) or len(cases) != 1993:
        raise RuntimeError("ranking requires all 1,993 fixed review cases")
    for rows in cases.values():
        rows.sort(key=lambda item: (int(item["t0"]), int(item["t1"])))

    results = []
    for mean_min in np.arange(0.30, 0.501, 0.01):
        for fill_min in (0.0, 0.80, 0.84, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97):
            for smooth_min in (0.0, 0.75, 0.80, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94):
                result = evaluate(
                    cases,
                    labels,
                    mean_min=float(mean_min),
                    fill_min=fill_min,
                    smooth_min=smooth_min,
                )
                if (
                    result["real_recall"] >= args.min_recall
                    and result["real_recall_p0"] >= args.min_recall - 0.01
                    and result["real_recall_p1"] >= args.min_recall - 0.01
                ):
                    results.append(result)
    for occupancy_min in np.arange(0.15, 0.401, 0.005):
        result = evaluate(
            cases,
            labels,
            mean_min=0.0,
            fill_min=0.0,
            smooth_min=0.0,
            occupancy_min=float(occupancy_min),
        )
        if (
            result["real_recall"] >= args.min_recall
            and result["real_recall_p0"] >= args.min_recall - 0.01
            and result["real_recall_p1"] >= args.min_recall - 0.01
        ):
            results.append(result)
    for log_mean_min in np.arange(0.15, 0.601, 0.005):
        result = evaluate(
            cases,
            labels,
            mean_min=0.0,
            fill_min=0.0,
            smooth_min=0.0,
            log_mean_min=float(log_mean_min),
        )
        if (
            result["real_recall"] >= args.min_recall
            and result["real_recall_p0"] >= args.min_recall - 0.01
            and result["real_recall_p1"] >= args.min_recall - 0.01
        ):
            results.append(result)
    for period_occupancy_min in np.arange(0.10, 0.401, 0.005):
        result = evaluate(
            cases,
            labels,
            mean_min=0.0,
            fill_min=0.0,
            smooth_min=0.0,
            period_occupancy_min=float(period_occupancy_min),
        )
        if (
            result["real_recall"] >= args.min_recall
            and result["real_recall_p0"] >= args.min_recall - 0.01
            and result["real_recall_p1"] >= args.min_recall - 0.01
        ):
            results.append(result)
    for period_log_mean_min in np.arange(0.10, 0.601, 0.005):
        result = evaluate(
            cases,
            labels,
            mean_min=0.0,
            fill_min=0.0,
            smooth_min=0.0,
            period_log_mean_min=float(period_log_mean_min),
        )
        if (
            result["real_recall"] >= args.min_recall
            and result["real_recall_p0"] >= args.min_recall - 0.01
            and result["real_recall_p1"] >= args.min_recall - 0.01
        ):
            results.append(result)
    for mean_min in np.arange(0.30, 0.481, 0.01):
        for period_coherence_min in np.arange(0.50, 0.951, 0.025):
            result = evaluate(
                cases,
                labels,
                mean_min=float(mean_min),
                fill_min=0.0,
                smooth_min=0.0,
                period_coherence_min=float(period_coherence_min),
            )
            if (
                result["real_recall"] >= args.min_recall
                and result["real_recall_p0"] >= args.min_recall - 0.01
                and result["real_recall_p1"] >= args.min_recall - 0.01
            ):
                results.append(result)
    for period_occupancy_min in np.arange(0.10, 0.351, 0.01):
        for period_coherence_min in np.arange(0.50, 0.951, 0.025):
            result = evaluate(
                cases,
                labels,
                mean_min=0.0,
                fill_min=0.0,
                smooth_min=0.0,
                period_occupancy_min=float(period_occupancy_min),
                period_coherence_min=float(period_coherence_min),
            )
            if (
                result["real_recall"] >= args.min_recall
                and result["real_recall_p0"] >= args.min_recall - 0.01
                and result["real_recall_p1"] >= args.min_recall - 0.01
            ):
                results.append(result)
    for mean_min in np.arange(0.30, 0.481, 0.01):
        for period_occupancy_min in np.arange(0.10, 0.351, 0.01):
            result = evaluate(
                cases,
                labels,
                mean_min=float(mean_min),
                fill_min=0.0,
                smooth_min=0.0,
                period_occupancy_min=float(period_occupancy_min),
            )
            if (
                result["real_recall"] >= args.min_recall
                and result["real_recall_p0"] >= args.min_recall - 0.01
                and result["real_recall_p1"] >= args.min_recall - 0.01
            ):
                results.append(result)
    for mean_min in np.arange(0.40, 0.481, 0.01):
        for period_coherence_min in np.arange(0.75, 0.951, 0.025):
            for shape_fill_min in np.arange(0.40, 0.651, 0.025):
                result = evaluate(
                    cases,
                    labels,
                    mean_min=float(mean_min),
                    fill_min=0.0,
                    smooth_min=0.0,
                    period_coherence_min=float(period_coherence_min),
                    shape_fill_min=float(shape_fill_min),
                )
                if (
                    result["real_recall"] >= args.min_recall
                    and result["real_recall_p0"] >= args.min_recall - 0.01
                    and result["real_recall_p1"] >= args.min_recall - 0.01
                ):
                    results.append(result)
    for period_energy_lock_min in np.arange(0.50, 0.951, 0.025):
        result = evaluate(
            cases,
            labels,
            mean_min=0.0,
            fill_min=0.0,
            smooth_min=0.0,
            period_energy_lock_min=float(period_energy_lock_min),
        )
        if (
            result["real_recall"] >= args.min_recall
            and result["real_recall_p0"] >= args.min_recall - 0.01
            and result["real_recall_p1"] >= args.min_recall - 0.01
        ):
            results.append(result)
    for period_profile_coherence_min in np.arange(0.60, 0.981, 0.02):
        result = evaluate(
            cases,
            labels,
            mean_min=0.0,
            fill_min=0.0,
            smooth_min=0.0,
            period_profile_coherence_min=float(period_profile_coherence_min),
        )
        if (
            result["real_recall"] >= args.min_recall
            and result["real_recall_p0"] >= args.min_recall - 0.01
            and result["real_recall_p1"] >= args.min_recall - 0.01
        ):
            results.append(result)
    for mean_min in np.arange(0.30, 0.481, 0.01):
        for period_energy_lock_min in np.arange(0.50, 0.951, 0.025):
            result = evaluate(
                cases,
                labels,
                mean_min=float(mean_min),
                fill_min=0.0,
                smooth_min=0.0,
                period_energy_lock_min=float(period_energy_lock_min),
            )
            if (
                result["real_recall"] >= args.min_recall
                and result["real_recall_p0"] >= args.min_recall - 0.01
                and result["real_recall_p1"] >= args.min_recall - 0.01
            ):
                results.append(result)
    for mean_min in np.arange(0.43, 0.491, 0.005):
        for period_energy_lock_min in np.arange(0.88, 0.961, 0.005):
            for period_profile_coherence_min in np.arange(0.70, 0.901, 0.025):
                result = evaluate(
                    cases,
                    labels,
                    mean_min=float(mean_min),
                    fill_min=0.0,
                    smooth_min=0.0,
                    period_energy_lock_min=float(period_energy_lock_min),
                    period_profile_coherence_min=float(period_profile_coherence_min),
                )
                if (
                    result["real_recall"] >= args.min_recall
                    and result["real_recall_p0"] >= args.min_recall - 0.01
                    and result["real_recall_p1"] >= args.min_recall - 0.01
                ):
                    results.append(result)
    for fill_power in (0.25, 0.5, 1.0, 2.0, 4.0):
        for smooth_power in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
            for score_min in np.arange(0.20, 0.501, 0.005):
                result = evaluate(
                    cases,
                    labels,
                    mean_min=0.0,
                    fill_min=0.0,
                    smooth_min=0.0,
                    score_min=float(score_min),
                    fill_power=fill_power,
                    smooth_power=smooth_power,
                )
                if (
                    result["real_recall"] >= args.min_recall
                    and result["real_recall_p0"] >= args.min_recall - 0.01
                    and result["real_recall_p1"] >= args.min_recall - 0.01
                ):
                    results.append(result)

    results.sort(
        key=lambda row: (
            max(
                row["fp_interval_rate"],
                row["pure_fp_rate"],
                row["fp_interval_rate_p0"],
                row["fp_interval_rate_p1"],
                row["pure_fp_rate_p0"],
                row["pure_fp_rate_p1"],
            ),
            -row["real_recall"],
            -row["median_iou"],
        )
    )
    fields = (
        "mean_min",
        "fill_min",
        "smooth_min",
        "occupancy_min",
        "log_mean_min",
        "period_occupancy_min",
        "period_log_mean_min",
        "period_coherence_min",
        "shape_fill_min",
        "period_energy_lock_min",
        "period_profile_coherence_min",
        "score_min",
        "fill_power",
        "smooth_power",
        "real_recall",
        "real_recall_p0",
        "real_recall_p1",
        "fp_interval_rate",
        "pure_fp_rate",
        "fp_interval_rate_p0",
        "fp_interval_rate_p1",
        "pure_fp_rate_p0",
        "pure_fp_rate_p1",
        "median_iou",
        "boundary_median_ae",
    )
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows({field: row[field] for field in fields} for row in results[: args.top])


if __name__ == "__main__":
    main()
