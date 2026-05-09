#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ce4_period_search.reporting import write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Markdown CWT period-candidate review report.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Single-run or batch directory.")
    parser.add_argument("--output", type=Path, default=None, help="Output Markdown path. Defaults to report.md.")
    parser.add_argument("--top-n", type=int, default=10, help="Rows to show in top tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = write_report(args.run_dir, output_path=args.output, top_n=args.top_n)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
