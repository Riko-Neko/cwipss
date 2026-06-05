from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .models import VALIDATION_REVIEWED_FIELDNAMES


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv_rows(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    with Path(path).open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _float(value: object) -> float:
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _rank_value(row: Mapping[str, Any], key: str, default: float) -> float:
    value = _float(row.get(key))
    return value if math.isfinite(value) else default


def valid_pvalue(row: Mapping[str, Any]) -> float:
    if row.get("validation_status") != "evaluated":
        return math.nan
    pvalue = _float(row.get("shuffle_pvalue"))
    if not math.isfinite(pvalue) or pvalue < 0.0 or pvalue > 1.0:
        return math.nan
    return pvalue


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """Return BH adjusted p-values in the original order."""
    if not pvalues:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda item: (item[1], item[0]))
    adjusted = [math.nan] * len(pvalues)
    running_min = 1.0
    m = float(len(pvalues))
    for rank, (original_idx, pvalue) in reversed(list(enumerate(indexed, start=1))):
        qvalue = min(1.0, pvalue * m / float(rank))
        running_min = min(running_min, qvalue)
        adjusted[original_idx] = running_min
    return adjusted


def _apply_qvalues(rows: list[dict[str, Any]], row_indices: list[int], target_key: str) -> None:
    pvalues = [float(rows[idx]["p_value"]) for idx in row_indices]
    qvalues = benjamini_hochberg(pvalues)
    for idx, qvalue in zip(row_indices, qvalues, strict=True):
        rows[idx][target_key] = qvalue


def _run_groups(rows: list[dict[str, Any]], valid_indices: list[int]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx in valid_indices:
        groups[str(rows[idx].get("run_id", ""))].append(idx)
    return groups


def _assign_evidence_ranks(rows: list[dict[str, Any]], valid_indices: list[int]) -> None:
    ordered = sorted(
        valid_indices,
        key=lambda idx: (
            _rank_value(rows[idx], "global_q_value", math.inf),
            _rank_value(rows[idx], "q_value", math.inf),
            _rank_value(rows[idx], "p_value", math.inf),
            -_rank_value(rows[idx], "observed_metric", -math.inf),
            -_rank_value(rows[idx], "fold_profile_snr", -math.inf),
            _rank_value(rows[idx], "candidate_id", math.inf),
        ),
    )
    for rank, idx in enumerate(ordered, start=1):
        rows[idx]["evidence_rank"] = rank


def review_validation_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    reviewed = [dict(row) for row in rows]
    valid_indices: list[int] = []
    for idx, row in enumerate(reviewed):
        pvalue = valid_pvalue(row)
        if math.isfinite(pvalue):
            row["p_value"] = pvalue
            row["stats_status"] = "evaluated"
            valid_indices.append(idx)
        else:
            row["p_value"] = ""
            row["q_value"] = ""
            row["global_q_value"] = ""
            row["evidence_rank"] = ""
            row["stats_status"] = "missing_pvalue"

    for group_indices in _run_groups(reviewed, valid_indices).values():
        _apply_qvalues(reviewed, group_indices, "q_value")
    _apply_qvalues(reviewed, valid_indices, "global_q_value")
    _assign_evidence_ranks(reviewed, valid_indices)
    return reviewed


def run_stats(input_path: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(input_path)
    reviewed = review_validation_rows(rows)
    write_csv_rows(output_path, reviewed, VALIDATION_REVIEWED_FIELDNAMES)
    return reviewed
