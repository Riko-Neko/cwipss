#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ce4_period_search.config import SWTScanConfig, load_swt_config
from ce4_period_search.validation import (
    ValidationConfig,
    read_csv_rows,
    validate_candidate_rows,
    validation_config_from_scan_config,
    write_validation_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SWT period candidates in the original time series.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing candidate CSV files.")
    parser.add_argument("--config", type=Path, default=None, help="Optional config JSON. Defaults to run config.")
    parser.add_argument("--candidates", type=Path, default=None, help="Candidate CSV. Defaults to candidates_reviewed.csv.")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV. Defaults to validation_summary.csv.")
    parser.add_argument(
        "--include-vetoed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Validate candidates already marked vetoed.",
    )
    parser.add_argument("--max-candidates", type=int, default=None, help="Maximum candidates to validate.")
    parser.add_argument("--window-periods", type=int, default=None, help="Validation window size in approximate periods.")
    parser.add_argument("--min-window-records", type=int, default=None, help="Minimum validation window records.")
    parser.add_argument("--max-window-records", type=int, default=None, help="Maximum validation window records.")
    parser.add_argument("--period-search-radius", type=float, default=None, help="Search +/- factor around SWT scale.")
    parser.add_argument("--fold-bins", type=int, default=None, help="Maximum phase bins used during folding.")
    parser.add_argument("--shuffle-trials", type=int, default=None, help="Shuffle/null trials per candidate.")
    parser.add_argument("--random-seed", type=int, default=None, help="Base random seed for shuffle/null tests.")
    return parser.parse_args()


def _default_candidate_path(run_dir: Path) -> Path:
    reviewed = run_dir / "candidates_reviewed.csv"
    if reviewed.exists():
        return reviewed
    return run_dir / "candidates_raw.csv"


def _load_scan_config(args: argparse.Namespace) -> SWTScanConfig:
    config_path = args.config
    if config_path is None:
        default_path = args.run_dir / "config.resolved.json"
        config_path = default_path if default_path.exists() else None
    return load_swt_config(config_path)


def _resolve_validation_config(args: argparse.Namespace) -> ValidationConfig:
    scan_config = _load_scan_config(args)
    config = validation_config_from_scan_config(scan_config)
    overrides = {}
    for field_name in (
        "include_vetoed",
        "max_candidates",
        "window_periods",
        "min_window_records",
        "max_window_records",
        "period_search_radius",
        "fold_bins",
        "shuffle_trials",
        "random_seed",
    ):
        value = getattr(args, field_name)
        if value is not None:
            overrides[field_name] = value
    return replace(config, **overrides)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    candidates_path = args.candidates or _default_candidate_path(run_dir)
    output_path = args.output or (run_dir / "validation_summary.csv")
    detail_dir = run_dir / "validation"
    config = _resolve_validation_config(args)

    rows = read_csv_rows(candidates_path)
    results = validate_candidate_rows(rows, config=config, project_dir=PROJECT_DIR)
    write_validation_outputs(output_path, detail_dir, results, config)

    evaluated = sum(1 for row in results if row.get("validation_status") == "evaluated")
    errors = sum(1 for row in results if row.get("validation_status") == "error")
    print(f"Candidates read: {len(rows)}")
    print(f"Validation rows: {len(results)}")
    print(f"Evaluated: {evaluated}")
    print(f"Errors: {errors}")
    print(f"Summary: {output_path}")
    print(f"Details: {detail_dir}")


if __name__ == "__main__":
    main()
