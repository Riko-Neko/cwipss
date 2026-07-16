"""Dataset-specific Top1 simplification and adaptation benchmark.

The default phase removes one Top1 mechanism at a time. Adaptation additionally
scans frequency-reference geometry and coherent-cycle count. The quick pass uses
the existing threshold negative control; rerun Pareto finalists with --strict-pelt.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cwt_activity_algorithms import (  # noqa: E402
    HIGH_RANK_MECHANISM_CONTROLS,
    TOP1_CONTINUOUS_SUPPORT_ALGORITHMS,
    TOP1_ADAPTATION_ALGORITHMS,
    TOP1_SIMPLIFICATION_ALGORITHMS,
    cwt_activity_algorithm_map,
)
from cwt_activity_rank import (  # noqa: E402
    CWTActivityRun,
    largest_complete_2c,
    run_cwt_activity_rank,
)


BASELINE = "post_freq_max8_center_8cycle_s70_f20"
REPORT_FIELDS = [
    "algorithm",
    "algorithm_family",
    "reference_channels",
    "guard_channels",
    "coherent_cycles",
    "persistence_support",
    "support_weighting",
    "score_floor",
    "optional_gate_count",
    "rank_score",
    "paired_activity_detection_rate",
    "paired_activity_loss",
    "paired_band_detection_rate",
    "paired_band_loss",
    "truth_window_hit_rate",
    "truth_window_loss",
    "peak_in_truth_rate",
    "peak_in_truth_loss",
    "negative_score_positive_fraction",
    "false_windows_per_channel",
    "mean_algorithm_seconds",
    "runtime_ratio",
    "within_loss_budget",
]


def _unique(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _reference_geometry(mode: str) -> tuple[int, int]:
    match = re.fullmatch(r"post_cwt_neighbor(\d+)(?:_guard(\d+))?", mode)
    if match is None:
        return 0, 0
    return int(match.group(1)), int(match.group(2) or 0)


def _loss_report(
    result: dict[str, Any],
    *,
    max_recall_loss: float,
    max_positive_fraction: float,
    max_false_windows_per_channel: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_by_name = {str(row["algorithm"]): row for row in result["summary_rows"]}
    baseline = rows_by_name[BASELINE]
    catalog = cwt_activity_algorithm_map()
    report: list[dict[str, Any]] = []
    for name in result["config"]["algorithms"]:
        row = rows_by_name[str(name)]
        definition = catalog[str(name)]
        params = definition["params"]
        references, guard = _reference_geometry(str(definition["input_denoiser"]))
        activity_loss = float(baseline["paired_activity_detection_rate"]) - float(row["paired_activity_detection_rate"])
        band_loss = float(baseline["paired_band_detection_rate"]) - float(row["paired_band_detection_rate"])
        window_loss = float(baseline["truth_window_hit_rate"]) - float(row["truth_window_hit_rate"])
        peak_loss = float(baseline["peak_in_truth_rate"]) - float(row["peak_in_truth_rate"])
        positive_fraction = float(row["negative_score_positive_fraction"])
        false_windows = float(row["false_windows_per_channel"])
        baseline_seconds = max(float(baseline["mean_algorithm_seconds"]), 1e-12)
        support = float(params.get("min_positive_support", 0.0))
        floor = float(params.get("score_floor", 0.0))
        within_budget = (
            max(activity_loss, band_loss, window_loss, peak_loss) <= max_recall_loss + 1e-12
            and positive_fraction <= max_positive_fraction
            and false_windows <= max_false_windows_per_channel
        )
        report.append(
            {
                "algorithm": name,
                "algorithm_family": definition["family"],
                "reference_channels": references,
                "guard_channels": guard,
                "coherent_cycles": float(params.get("time_support_cycles", 0.0)),
                "persistence_support": support,
                "support_weighting": str(params.get("support_weighting", "none")),
                "score_floor": floor,
                "optional_gate_count": int(support > 0.0) + int(floor > 0.0),
                "rank_score": float(row["rank_score"]),
                "paired_activity_detection_rate": float(row["paired_activity_detection_rate"]),
                "paired_activity_loss": activity_loss,
                "paired_band_detection_rate": float(row["paired_band_detection_rate"]),
                "paired_band_loss": band_loss,
                "truth_window_hit_rate": float(row["truth_window_hit_rate"]),
                "truth_window_loss": window_loss,
                "peak_in_truth_rate": float(row["peak_in_truth_rate"]),
                "peak_in_truth_loss": peak_loss,
                "negative_score_positive_fraction": positive_fraction,
                "false_windows_per_channel": false_windows,
                "mean_algorithm_seconds": float(row["mean_algorithm_seconds"]),
                "runtime_ratio": float(row["mean_algorithm_seconds"]) / baseline_seconds,
                "within_loss_budget": int(within_budget),
            }
        )

    eligible_simplifications = [
        row
        for row in report
        if (
            row["algorithm"] in (*TOP1_SIMPLIFICATION_ALGORITHMS, *TOP1_CONTINUOUS_SUPPORT_ALGORITHMS)
            and row["within_loss_budget"]
        )
    ]
    eligible_simplifications.sort(
        key=lambda row: (
            int(row["optional_gate_count"]),
            float(row["runtime_ratio"]),
            -float(row["rank_score"]),
        )
    )
    eligible_adaptations = [
        row
        for row in report
        if row["algorithm"] in TOP1_ADAPTATION_ALGORITHMS and row["within_loss_budget"]
    ]
    eligible_adaptations.sort(key=lambda row: -float(row["rank_score"]))
    decision = {
        "baseline": BASELINE,
        "loss_budget": {
            "max_recall_loss": max_recall_loss,
            "max_negative_score_positive_fraction": max_positive_fraction,
            "max_false_windows_per_channel": max_false_windows_per_channel,
        },
        "recommended_simplification": (
            eligible_simplifications[0]["algorithm"] if eligible_simplifications else BASELINE
        ),
        "best_adapted_parameters": (
            eligible_adaptations[0]["algorithm"] if eligible_adaptations else None
        ),
        "selection_rule": "fewest optional gates, then runtime, then rank; adaptation uses highest rank within the same absolute loss budget",
    }
    return report, decision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Top1 parameters on a new dataset.")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--injections", type=Path, default=PROJECT_DIR / "configs/injection_lowfreq_random_100.json")
    parser.add_argument("--cwt-config", type=Path, default=PROJECT_DIR / "configs/cwt_default.json")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "runs/top1_parameter_validation")
    parser.add_argument(
        "--phase",
        choices=("simplification", "comparison", "adaptation", "all"),
        default="simplification",
    )
    parser.add_argument("--algorithms", help="Comma-separated catalog names; overrides --phase.")
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--candidate-period-max", type=float, default=1000.0)
    parser.add_argument("--negative-f-start", type=float, default=0.15)
    parser.add_argument("--negative-f-stop", type=float, default=1.90)
    parser.add_argument("--negative-max-channels", type=int, default=0)
    parser.add_argument("--strict-pelt", action="store_true")
    parser.add_argument("--max-recall-loss", type=float, default=0.03)
    parser.add_argument("--max-positive-fraction", type=float, default=0.001)
    parser.add_argument("--max-false-windows-per-channel", type=float, default=0.10)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = args.input or largest_complete_2c(PROJECT_DIR / "data" / "CE4")
    if args.algorithms:
        requested = [item.strip() for item in args.algorithms.split(",") if item.strip()]
        algorithms = _unique([BASELINE, *requested])
    elif args.phase == "simplification":
        algorithms = TOP1_SIMPLIFICATION_ALGORITHMS
    elif args.phase == "comparison":
        algorithms = _unique([*TOP1_SIMPLIFICATION_ALGORITHMS, *HIGH_RANK_MECHANISM_CONTROLS])
    elif args.phase == "adaptation":
        algorithms = _unique([BASELINE, *TOP1_ADAPTATION_ALGORITHMS])
    else:
        algorithms = _unique(
            [
                *TOP1_SIMPLIFICATION_ALGORITHMS,
                *TOP1_ADAPTATION_ALGORITHMS,
                *HIGH_RANK_MECHANISM_CONTROLS,
            ]
        )

    result = run_cwt_activity_rank(
        CWTActivityRun(
            output_dir=args.output,
            input_path=input_path,
            injection_config=args.injections,
            cwt_config=args.cwt_config,
            algorithms=algorithms,
            cwt_backend=args.backend,
            candidate_period_max_records=args.candidate_period_max,
            progress_every=args.progress_every,
            negative_control=True,
            negative_f_start_mhz=args.negative_f_start,
            negative_f_stop_mhz=args.negative_f_stop,
            negative_max_channels=args.negative_max_channels,
            negative_window_method="pelt" if args.strict_pelt else "threshold",
        )
    )
    report, decision = _loss_report(
        result,
        max_recall_loss=args.max_recall_loss,
        max_positive_fraction=args.max_positive_fraction,
        max_false_windows_per_channel=args.max_false_windows_per_channel,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "top1_performance_loss.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(report)
    (args.output / "top1_validation_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=True)
    )
    print(json.dumps(decision, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
