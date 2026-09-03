#!/usr/bin/env python3
"""Discover compact continuity rules with parity-held-out shallow trees."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text


BASE_DIR = Path(__file__).resolve().parent
WINDOWS = BASE_DIR / "artifacts/cpro_ridge_continuity_features_v1/windows.csv"
FEATURES = (
    "continuity_mean",
    "period_energy_lock",
    "period_profile_coherence",
    "period_occupancy",
    "period_coherence",
    "shape_fill",
    "temporal_contrast",
)


def overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return min(a1, b1) > max(a0, b0)


def load_data():
    labels = {
        row["raw_key"]: json.loads(row["intervals"])
        for row in csv.DictReader((BASE_DIR / "labels.csv").open(newline="", encoding="utf-8"))
    }
    cases = defaultdict(list)
    for row in csv.DictReader(WINDOWS.open(newline="", encoding="utf-8")):
        if row["algorithm"] != "cpro_seg_d995_g2":
            continue
        occupancy = float(row["continuity_occupancy"])
        item = {
            "rank": int(row["review_rank"]),
            "t0": int(row["t0_rec"]),
            "t1": int(row["t1_rec"]),
            "continuity_mean": float(row["continuity_mean"]),
            "period_energy_lock": float(row["period_energy_lock"]),
            "period_profile_coherence": float(row["period_profile_coherence"]),
            "period_occupancy": float(row["period_occupancy"]),
            "period_coherence": float(row["period_occupancy"]) / max(occupancy, 1e-30),
            "shape_fill": float(row["activity_mean"]) / max(float(row["activity_max"]), 1e-30),
            "temporal_contrast": float(row["temporal_contrast"]),
        }
        cases[row["raw_key"]].append(item)
    return labels, cases


def training_rows(labels, cases, parity: int):
    matrix, target = [], []
    for key, rows in cases.items():
        if int(rows[0]["rank"]) % 2 != parity:
            continue
        intervals = labels[key]
        high = [item for item in intervals if item.get("label") == "keep" and item.get("conf") == "high"]
        false = [item for item in intervals if item.get("label") == "fp"]
        pure_false = bool(false) and len(false) == len(intervals)
        for row in rows:
            positive = any(
                overlap(row["t0"], row["t1"], int(item["t0"]), int(item["t1"]))
                for item in high
            )
            negative = pure_false or any(
                overlap(row["t0"], row["t1"], int(item["t0"]), int(item["t1"]))
                for item in false
            )
            if positive == negative:
                continue
            matrix.append([float(row[name]) for name in FEATURES])
            target.append(int(positive))
    return np.asarray(matrix), np.asarray(target)


def evaluate(model, threshold, labels, cases, parity: int):
    high_hits, fp_hits, pure_fp_hits = [], [], []
    for key, rows in cases.items():
        if int(rows[0]["rank"]) % 2 != parity:
            continue
        matrix = np.asarray([[float(row[name]) for name in FEATURES] for row in rows])
        keep = model.predict_proba(matrix)[:, 1] >= threshold
        spans = [(row["t0"], row["t1"]) for row, accepted in zip(rows, keep) if accepted]
        intervals = labels[key]
        high = [item for item in intervals if item.get("label") == "keep" and item.get("conf") == "high"]
        false = [item for item in intervals if item.get("label") == "fp"]
        if high:
            high_hits.append(
                int(
                    any(
                        overlap(a0, a1, int(item["t0"]), int(item["t1"]))
                        for a0, a1 in spans
                        for item in high
                    )
                )
            )
        for item in false:
            fp_hits.append(
                int(any(overlap(a0, a1, int(item["t0"]), int(item["t1"])) for a0, a1 in spans))
            )
        if false and len(false) == len(intervals):
            pure_fp_hits.append(int(bool(spans)))
    return tuple(float(np.mean(values)) for values in (high_hits, fp_hits, pure_fp_hits))


def main() -> None:
    labels, cases = load_data()
    candidates = []
    for depth in (2, 3, 4):
        for leaf in (20, 50, 100):
            for positive_weight in (2.0, 5.0, 10.0, 20.0, 50.0):
                folds = []
                models = []
                for train_parity in (0, 1):
                    matrix, target = training_rows(labels, cases, train_parity)
                    model = DecisionTreeClassifier(
                        max_depth=depth,
                        min_samples_leaf=leaf,
                        class_weight={0: 1.0, 1: positive_weight},
                        random_state=0,
                    ).fit(matrix, target)
                    models.append(model)
                for threshold in np.arange(0.10, 0.901, 0.05):
                    metrics = [
                        evaluate(models[1 - parity], threshold, labels, cases, parity)
                        for parity in (0, 1)
                    ]
                    if min(item[0] for item in metrics) < 0.96:
                        continue
                    candidates.append(
                        (
                            max(item[1] for item in metrics),
                            max(item[2] for item in metrics),
                            -min(item[0] for item in metrics),
                            depth,
                            leaf,
                            positive_weight,
                            threshold,
                            metrics,
                            models,
                        )
                    )
    for candidate in sorted(candidates)[:10]:
        fp, pure_fp, neg_recall, depth, leaf, weight, threshold, metrics, models = candidate
        print(
            f"depth={depth} leaf={leaf} pos_weight={weight:g} threshold={threshold:.2f} "
            f"worst_recall={-neg_recall:.4f} worst_fp={fp:.4f} worst_pure_fp={pure_fp:.4f} "
            f"folds={metrics}"
        )
        print("train_even:\n" + export_text(models[0], feature_names=list(FEATURES)))
        print("train_odd:\n" + export_text(models[1], feature_names=list(FEATURES)))


if __name__ == "__main__":
    main()
