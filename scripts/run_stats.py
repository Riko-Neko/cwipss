#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cwipss.stats import run_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review validation evidence with Benjamini-Hochberg FDR.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory containing validation_summary.csv.")
    parser.add_argument("--input", type=Path, default=None, help="Input validation summary CSV.")
    parser.add_argument("--output", type=Path, default=None, help="Output reviewed validation CSV.")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.input is None and args.run_dir is None:
        raise SystemExit("Provide --run-dir or --input")
    input_path = args.input or (args.run_dir / "validation_summary.csv")
    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name("validation_reviewed.csv")
    return input_path, output_path


def main() -> None:
    input_path, output_path = resolve_paths(parse_args())
    rows = run_stats(input_path, output_path)
    valid = sum(1 for row in rows if row.get("stats_status") == "evaluated")
    print(f"Rows read: {len(rows)}")
    print(f"Rows with p-values: {valid}")
    print(f"Reviewed validation: {output_path}")


if __name__ == "__main__":
    main()
