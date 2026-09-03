#!/usr/bin/env python3
"""Render one raw-series plus CWT diagnostic image per CPRF review candidate."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cwipss-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cwipss.signal.cwt import cwt_power_cube, period_grid_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=BASE_DIR / "selection.csv")
    parser.add_argument("--labels", type=Path, default=BASE_DIR / "labels.csv")
    parser.add_argument("--metadata", type=Path, default=BASE_DIR / "artifacts/metadata.json")
    parser.add_argument("--archive", type=Path, default=BASE_DIR / "artifacts/single_channel_slices.npz")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "artifacts/review")
    parser.add_argument("--start-rank", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Zero renders every remaining candidate")
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--comparison-windows", type=Path, default=None)
    parser.add_argument(
        "--confidence",
        choices=("all", "low", "medium", "high"),
        default="all",
        help="Render cases containing a Real interval at this confidence",
    )
    return parser.parse_args()


def finite_limits(values: np.ndarray, low: float, high: float) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.quantile(finite, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def span_iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    overlap = max(0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return float(overlap / union) if union > 0 else 0.0


def render_case(
    row: dict[str, str],
    meta: dict[str, int | str],
    raw: np.ndarray,
    periods: np.ndarray,
    output: Path,
    backend: str,
    cuda_device: int,
    comparison: dict[str, list[tuple[int, int]]] | None = None,
    truth_spans: list[tuple[int, int]] | None = None,
) -> None:
    offset = int(meta["extract_t0_rec"])
    t0 = int(row["t0_rec"])
    t1 = int(row["t1_rec"])
    period = float(row["period_rec"])
    power = cwt_power_cube(
        raw[:, None],
        periods,
        wavelet="cmor1.5-1.0",
        normalize_channels=False,
        method="fft",
        backend=backend,
        cuda_device=cuda_device,
    )[:, :, 0]
    positive = power[power > 0]
    floor = max(float(np.median(positive)) * 1e-8, np.finfo(np.float32).tiny) if positive.size else 1e-30
    shown = np.log10(np.maximum(power, floor))
    lo, hi = finite_limits(shown, 0.01, 0.995)
    records = np.arange(len(raw), dtype=np.float64) + offset + 0.5
    record_start = offset
    record_stop = offset + len(raw)

    fig, (ax_raw, ax_strip, ax_cwt) = plt.subplots(
        3,
        1,
        figsize=(14, 8.5),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (0.75, 0.16, 2.2)},
    )
    ax_raw.plot(records, raw, color="#1d3557", linewidth=0.7)
    ax_raw.axvspan(t0, t1, color="#6c757d", alpha=0.14)
    ax_raw.set_ylabel("Raw amplitude")
    ax_raw.grid(alpha=0.18)

    raw_lo, raw_hi = finite_limits(raw, 0.01, 0.99)
    ax_strip.imshow(
        raw[None, :],
        origin="lower",
        aspect="auto",
        extent=(record_start, record_stop, -0.5, 0.5),
        cmap="cividis",
        vmin=raw_lo,
        vmax=raw_hi,
        interpolation="nearest",
    )
    ax_strip.axvspan(t0, t1, color="#6c757d", alpha=0.12)
    ax_strip.set_yticks([0.0], labels=[f"ch {row['channel']}"])
    ax_strip.set_ylabel("Raw 1-bin")

    styles = {
        "cpro": ("#0077b6", "Edge-preserving CPRO"),
        "direct_mean": ("#e76f51", "Direct mean"),
        "direct_integral": ("#bc6c25", "Direct integral"),
    }
    if truth_spans:
        comparison = {
            algorithm: list(
                dict.fromkeys(
                    max(spans, key=lambda span: span_iou(span, truth))
                    for truth in truth_spans
                )
            )
            for algorithm, spans in (comparison or {}).items()
            if spans
        }
    has_comparison_lines = False
    for span_index, (start, stop) in enumerate(truth_spans or []):
        for ax in (ax_raw, ax_strip, ax_cwt):
            ax.axvline(start, color="#2a9d8f", linewidth=1.5)
            ax.axvline(stop, color="#2a9d8f", linewidth=1.5)
        if span_index == 0:
            ax_raw.plot([], [], color="#2a9d8f", label="Manual Real boundary")
        has_comparison_lines = True
    for algorithm, spans in (comparison or {}).items():
        if algorithm not in styles:
            continue
        color, label = styles[algorithm]
        for span_index, (start, stop) in enumerate(spans):
            has_comparison_lines = True
            for ax in (ax_raw, ax_strip, ax_cwt):
                ax.axvline(start, color=color, linewidth=1.1, linestyle="--")
                ax.axvline(stop, color=color, linewidth=1.1, linestyle="--")
            if span_index == 0:
                ax_raw.plot([], [], color=color, linestyle="--", label=label)

    image = ax_cwt.imshow(
        shown,
        origin="lower",
        aspect="auto",
        extent=(record_start, record_stop, periods[0], periods[-1]),
        cmap="magma",
        vmin=lo,
        vmax=hi,
        interpolation="nearest",
    )
    ax_cwt.axvspan(t0, t1, color="#6c757d", alpha=0.10)
    ax_cwt.axhline(period, color="#f4d35e", linestyle="--", linewidth=1.2)
    ax_cwt.set_yscale("log")
    ax_cwt.set_ylim(max(periods[0], period / 2.0), min(periods[-1], period * 2.0))
    ax_cwt.set_xlim(record_start, record_stop)
    ax_cwt.set_xlabel("Record")
    ax_cwt.set_ylabel("Period (records)")
    if has_comparison_lines:
        ax_raw.legend(loc="upper right")
    fig.colorbar(image, ax=ax_cwt, pad=0.01, label="log10 CWT power")
    audit_category = str(row.get("_audit_category", "")).strip()
    category_text = f" | audit={audit_category}" if audit_category else ""
    fig.suptitle(
        f"Rank {row['review_rank']} | {row['run_id']} | ch={row['channel']} | "
        f"conc={float(row['band_conc']):.3f} contrast={float(row['local_contrast']):.3f} "
        f"strength={float(row['ridge_int']):.3f}{category_text}",
        fontsize=10,
    )
    filename = f"{int(row['review_rank']):04d}_{row['raw_key']}.png"
    fig.savefig(output / filename, dpi=125)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    with args.selection.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with args.labels.open(newline="", encoding="utf-8") as handle:
        label_rows = {row["raw_key"]: row for row in csv.DictReader(handle)}
    metadata = {
        str(item["raw_key"]): item
        for item in json.loads(args.metadata.read_text(encoding="utf-8"))
    }
    comparison: dict[str, dict[str, list[tuple[int, int]]]] = {}
    if args.comparison_windows is not None:
        with args.comparison_windows.open(newline="", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                key = item["raw_key"]
                algorithm = item["algorithm"]
                comparison.setdefault(key, {}).setdefault(algorithm, []).append(
                    (int(item["t0_rec"]), int(item["t1_rec"]))
                )
    rows = [row for row in rows if int(row["review_rank"]) >= args.start_rank]
    truth: dict[str, list[tuple[int, int]]] = {}
    for key, label_row in label_rows.items():
        intervals = json.loads(label_row.get("intervals") or "[]")
        truth[key] = [
            (int(interval["t0"]), int(interval["t1"]))
            for interval in intervals
            if str(interval.get("label", "")).lower() in {"keep", "real"}
            and (
                args.confidence == "all"
                or str(interval.get("conf", "")).lower() == args.confidence
            )
        ]
    if args.confidence != "all":
        rows = [row for row in rows if truth.get(row["raw_key"])]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("no candidates selected for rendering")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    periods = period_grid_records(2.0, 512.0, 96, "log")
    with np.load(args.archive) as archive:
        total = len(rows)
        for index, row in enumerate(rows, 1):
            key = row["raw_key"]
            render_case(
                row,
                metadata[key],
                np.asarray(archive[key], dtype=np.float32),
                periods,
                args.output_dir,
                args.backend,
                args.cuda_device,
                comparison.get(key),
                truth.get(key),
            )
            if index == 1 or index % 25 == 0 or index == total:
                print(f"[render] images={index}/{total}", flush=True)
    print(f"[render] complete output={args.output_dir}")


if __name__ == "__main__":
    main()
