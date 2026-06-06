#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cwipss.reporting.gallery import CandidateGalleryConfig, visualize_candidate_gallery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render per-candidate raw time-frequency and CWT gallery figures from an existing run."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Single-run or batch result directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <run-dir>/candidate_gallery.")
    parser.add_argument("--top", type=int, default=100, help="Number of candidates; 0 means all.")
    parser.add_argument(
        "--sort-by",
        choices=["auto", "evidence_rank", "global_q_value", "integrated_score"],
        default="auto",
        help="Candidate ordering. Auto prefers evidence_rank when validation statistics exist.",
    )
    parser.add_argument("--include-vetoed", action="store_true", help="Include vetoed candidates.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Directory containing source data when recorded source_file paths are unavailable.",
    )
    parser.add_argument("--context-periods", type=float, default=16.0, help="Target time context in candidate periods.")
    parser.add_argument("--min-window-records", type=int, default=256, help="Minimum plotted time window.")
    parser.add_argument("--max-window-records", type=int, default=4096, help="Maximum plotted time window.")
    parser.add_argument(
        "--freq-context-channels",
        type=int,
        default=8,
        help="Raw-frequency channels shown on each side of the candidate channel.",
    )
    parser.add_argument("--period-radius", type=float, default=2.0, help="CWT y-axis factor around the candidate period.")
    parser.add_argument("--cwt-backend", choices=["cpu", "cuda", "auto"], default=None, help="Override saved backend.")
    parser.add_argument("--cuda-device", type=int, default=None, help="Override saved CUDA device.")
    parser.add_argument("--dpi", type=int, default=140, help="Output PNG DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = visualize_candidate_gallery(
        args.run_dir,
        args.output_dir,
        source_root=args.source_root,
        project_dir=PROJECT_DIR,
        config=CandidateGalleryConfig(
            top_n=args.top,
            sort_by=args.sort_by,
            include_vetoed=args.include_vetoed,
            context_periods=args.context_periods,
            min_window_records=args.min_window_records,
            max_window_records=args.max_window_records,
            freq_context_channels=args.freq_context_channels,
            period_radius=args.period_radius,
            dpi=args.dpi,
            cwt_backend=args.cwt_backend,
            cuda_device=args.cuda_device,
        ),
    )
    print(f"Candidate gallery: {index}")


if __name__ == "__main__":
    main()
