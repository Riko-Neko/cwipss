#!/usr/bin/env python3
"""Render a compact scientific review package for period-profile rank runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = (
    PROJECT_DIR / "runs/period_profile_fullband_20211205_v1",
    PROJECT_DIR / "runs/period_profile_fullband_20190830_v1",
)
DEFAULT_RANK_RUN = PROJECT_DIR / "runs/period_profile_rank_full_v3"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/period_profile_review_v2"
DEFAULT_ALGORITHM = "pbsf_focus_joint_c35_r140"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else np.nan


def _render_rank(run_dir: Path, output: Path) -> Path:
    rows = _rows(run_dir / "period_profile_summary.csv")[:20]
    labels = [row["algorithm"].replace("pbsf_", "") for row in rows][::-1]
    scores = [_float(row, "rank_score") for row in rows][::-1]
    colors = ["#1b998b" if int(row["scientific_gate_pass"]) else "#d95d39" for row in rows][::-1]
    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    ax.barh(np.arange(len(rows)), scores, color=colors, alpha=0.90)
    ax.set_yticks(np.arange(len(rows)), labels=labels, fontsize=8)
    ax.set_xlabel("Scientific rank score")
    ax.set_title("Period-profile algorithm rank: green passes the scientific gate")
    ax.grid(axis="x", alpha=0.18)
    path = output / "rank_top20.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _selected_cases(run_dirs: tuple[Path, ...], algorithm: str) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for run_index, run_dir in enumerate(run_dirs, start=1):
        for row in _rows(run_dir / "period_profile_cases.csv"):
            if row["algorithm"] != algorithm:
                continue
            item = dict(row)
            item["run_index"] = str(run_index)
            item["run_name"] = run_dir.name
            selected.append(item)
    return selected


def _render_score_plane(rows: list[dict[str, str]], output: Path, algorithm: dict[str, Any]) -> Path:
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    for kind, color, label in (
        ("positive", "#1b998b", "Injected PELT windows"),
        ("negative", "#d95d39", "Real no-injection PELT windows"),
    ):
        subset = [row for row in rows if row["case_kind"] == kind]
        accepted = [row for row in subset if int(row["accepted"])]
        rejected = [row for row in subset if not int(row["accepted"])]
        ax.scatter(
            [_float(row, "band_concentration") for row in accepted],
            [_float(row, "local_contrast") for row in accepted],
            s=[24.0 + 14.0 * _float(row, "peak_strength") for row in accepted],
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.5,
            label=f"{label}: accepted",
        )
        ax.scatter(
            [_float(row, "band_concentration") for row in rejected],
            [_float(row, "local_contrast") for row in rejected],
            s=38,
            color=color,
            alpha=0.55,
            marker="x",
            linewidth=1.2,
            label=f"{label}: rejected",
        )
    concentration = float(algorithm.get("min_band_concentration", 0.0))
    contrast = float(algorithm.get("min_local_contrast", 0.0))
    if concentration > 0.0:
        ax.axvline(concentration, color="#2d3047", linestyle="--", linewidth=1.1)
    if contrast > 0.0:
        ax.axhline(
            contrast,
            color="#2d3047",
            linestyle="--",
            linewidth=1.1,
            label="Selected-filter hard contrast",
        )
    ax.set_xlabel("Main-band energy concentration")
    ax.set_ylabel("Main-band local sideband contrast")
    ax.set_title("Selected-filter score plane across two independent CE4 observations")
    ax.grid(alpha=0.16)
    ax.legend(fontsize=8, ncol=2)
    path = output / "top1_score_plane.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _render_period_accuracy(rows: list[dict[str, str]], output: Path) -> Path:
    positives = [row for row in rows if row["case_kind"] == "positive"]
    accepted = [row for row in positives if int(row["accepted"])]
    rejected = [row for row in positives if not int(row["accepted"])]
    truth_all = np.asarray([_float(row, "truth_period_records") for row in positives])
    lo = float(np.nanmin(truth_all))
    hi = float(np.nanmax(truth_all))
    axis = np.geomspace(lo, hi, 256)
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.fill_between(axis, 0.9 * axis, 1.1 * axis, color="#f4d35e", alpha=0.20, label="10% error band")
    ax.plot(axis, axis, color="#2d3047", linewidth=1.2)
    ax.scatter(
        [_float(row, "truth_period_records") for row in accepted],
        [_float(row, "peak_period_records") for row in accepted],
        color="#1b998b",
        s=28,
        alpha=0.72,
        label="Accepted",
    )
    ax.scatter(
        [_float(row, "truth_period_records") for row in rejected],
        [_float(row, "peak_period_records") for row in rejected],
        color="#d95d39",
        s=34,
        marker="x",
        alpha=0.65,
        label="Rejected by score filter",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Injected period (records)")
    ax.set_ylabel("Estimated main period (records)")
    ax.set_title("Selected-filter period estimate across two CE4 observations")
    ax.grid(alpha=0.16, which="both")
    ax.legend()
    path = output / "top1_period_accuracy.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _render_harmonics(rows: list[dict[str, str]], output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    for kind, color, label in (
        ("positive", "#1b998b", "Injected"),
        ("negative", "#d95d39", "Real no-injection"),
    ):
        subset = [row for row in rows if row["case_kind"] == kind]
        ax.scatter(
            [_float(row, "harmonic_2_score") for row in subset],
            [_float(row, "harmonic_3_score") for row in subset],
            color=color,
            s=34,
            alpha=0.65,
            label=label,
        )
    ax.set_xlabel("2f0 auxiliary response / main peak")
    ax.set_ylabel("3f0 auxiliary response / main peak")
    ax.set_title("Harmonics are diagnostic, not a hard acceptance condition")
    ax.grid(alpha=0.16)
    ax.legend()
    path = output / "top1_harmonic_diagnostics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _aggregate(rows: list[dict[str, str]]) -> dict[str, Any]:
    positives = [row for row in rows if row["case_kind"] == "positive"]
    negatives = [row for row in rows if row["case_kind"] == "negative"]
    accepted_positives = [row for row in positives if int(row["accepted"])]
    accepted_negatives = [row for row in negatives if int(row["accepted"])]
    errors = [_float(row, "period_error_fraction") for row in accepted_positives]
    return {
        "positive_windows": len(positives),
        "positive_accepted": len(accepted_positives),
        "positive_accept_rate": len(accepted_positives) / max(1, len(positives)),
        "accepted_period_hit_10_rate": float(
            np.mean([_float(row, "period_hit_10") for row in accepted_positives])
        )
        if accepted_positives
        else 0.0,
        "mean_accepted_period_error_fraction": float(np.mean(errors)) if errors else np.nan,
        "negative_windows": len(negatives),
        "negative_accepted": len(accepted_negatives),
        "negative_false_accept_rate": len(accepted_negatives) / max(1, len(negatives)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--rank-run", type=Path, default=DEFAULT_RANK_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--algorithm", default=DEFAULT_ALGORITHM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = tuple(args.run) if args.run else DEFAULT_RUNS
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    algorithm_maps = [json.loads((run / "period_profile_algorithm_map.json").read_text()) for run in run_dirs]
    if any(args.algorithm not in mapping for mapping in algorithm_maps):
        raise KeyError(f"Algorithm {args.algorithm!r} is not present in every run")
    algorithm = algorithm_maps[0][args.algorithm]
    rows = _selected_cases(run_dirs, args.algorithm)
    images = [
        _render_rank(args.rank_run, output),
        _render_score_plane(rows, output, algorithm),
        _render_period_accuracy(rows, output),
        _render_harmonics(rows, output),
    ]
    summary = {
        "schema_version": 1,
        "algorithm": args.algorithm,
        "algorithm_parameters": algorithm,
        "runs": [str(run) for run in run_dirs],
        "rank_run": str(args.rank_run),
        "aggregate": _aggregate(rows),
        "images": [path.name for path in images],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True))
    aggregate = summary["aggregate"]
    (output / "RESULT.md").write_text(
        "\n".join(
            [
                "# Period-Profile Rank Review",
                "",
                f"Selected algorithm: `{args.algorithm}`",
                "",
                f"- Positive PELT windows accepted: {aggregate['positive_accepted']}/{aggregate['positive_windows']} ",
                f"  ({aggregate['positive_accept_rate']:.3%})",
                f"- Accepted 10% period accuracy: {aggregate['accepted_period_hit_10_rate']:.3%}",
                f"- Mean accepted period error: {aggregate['mean_accepted_period_error_fraction']:.3%}",
                "- Real no-injection PELT windows retained: "
                f"{aggregate['negative_accepted']}/{aggregate['negative_windows']} ",
                f"  ({aggregate['negative_false_accept_rate']:.3%})",
                "",
                *[f"![{path.stem}]({path.name})" for path in images],
                "",
            ]
        )
    )
    print(f"Review: {output / 'RESULT.md'}")


if __name__ == "__main__":
    main()
