#!/usr/bin/env python3
"""Extract selected single-channel CE4 2C slices for manual CPRF review."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = Path("/data/Raid0/obs_data/CE4_LFRS_2C")
DTYPE = np.dtype(
    [
        ("frame_id", ("u1", 4)),
        ("version", "u1"),
        ("work_param", ("u1", 71)),
        ("solar_el", ">f4"),
        ("solar_az", ">f4"),
        ("cancel_period", ">u4"),
        ("cancel_num", ">u2"),
        ("accum_times", ">u2"),
        ("rec_len", ">u2"),
        ("spec", (">f4", 2048)),
        ("quality", "u1"),
    ],
    align=False,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=BASE_DIR / "selection.csv")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if DTYPE.itemsize != 8287:
        raise RuntimeError(f"unexpected CE4 2C record size: {DTYPE.itemsize}")

    with args.selection.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"selection is empty: {args.selection}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[Path(row["source_file"]).name].append(row)

    arrays: dict[str, np.ndarray] = {}
    metadata: list[dict[str, int | str]] = []
    total_files = len(grouped)
    for file_index, (source_name, source_rows) in enumerate(grouped.items(), 1):
        source = args.source_root / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        records = np.memmap(source, dtype=DTYPE, mode="r")
        n_records = len(records)
        for row in source_rows:
            channel = int(row["channel"])
            start = max(0, min(int(row["extract_t0_rec"]), n_records))
            stop = max(start, min(int(row["extract_t1_rec"]), n_records))
            key = row["raw_key"]
            arrays[key] = np.asarray(records["spec"][start:stop, channel], dtype=np.float32)
            metadata.append(
                {
                    "raw_key": key,
                    "review_rank": int(row["review_rank"]),
                    "source_file": str(source),
                    "channel": channel,
                    "extract_t0_rec": start,
                    "extract_t1_rec": stop,
                    "candidate_t0_local": int(row["t0_rec"]) - start,
                    "candidate_t1_local": int(row["t1_rec"]) - start,
                    "n_records": stop - start,
                }
            )
        del records
        if file_index == 1 or file_index % 10 == 0 or file_index == total_files:
            print(f"[extract] files={file_index}/{total_files} slices={len(arrays)}/{len(rows)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "single_channel_slices.npz"
    metadata_path = args.output_dir / "metadata.json"
    np.savez_compressed(archive, **arrays)
    metadata_path.write_text(
        json.dumps(sorted(metadata, key=lambda item: int(item["review_rank"])), indent=2),
        encoding="utf-8",
    )
    print(f"[extract] complete slices={len(arrays)} archive={archive}")


if __name__ == "__main__":
    main()
