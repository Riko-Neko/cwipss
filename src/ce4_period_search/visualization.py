from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "cwt_period_search_matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from .cwt import aggregate_cwt_time, cwt_power_cube
from .detection import channel_period_peak_score


@dataclass(frozen=True)
class CWTVisualizationConfig:
    enabled: bool = False
    max_blocks: int = 2
    max_channels: int = 4
    top_candidates: int = 50
    dpi: int = 140


@dataclass(frozen=True)
class SearchVisualizationConfig:
    wavelet: str
    periods: np.ndarray
    cwt_method: str = "fft"
    block_channels: int = 128
    threshold: float = 2.5
    min_prominence: float = 2.5
    dog_sigma_peak: float = 1.0
    dog_sigma_background: float = 10.0
    time_aggregation: str = "p95"
    aggregation_percentile: float = 95.0


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def _float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _limits(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None, None
    lo, hi = np.nanpercentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None, None
    return float(lo), float(hi)


def _freq_step(freqs_mhz: np.ndarray) -> float:
    if freqs_mhz.size < 2:
        return 1.0
    diffs = np.diff(np.asarray(freqs_mhz, dtype=np.float64))
    diffs = np.abs(diffs[np.isfinite(diffs) & (np.abs(diffs) > 0)])
    return float(np.nanmedian(diffs)) if diffs.size else 1.0


def _matrix_extent(freqs_mhz: np.ndarray, records: int, record_offset: int) -> list[float]:
    step = _freq_step(freqs_mhz)
    if freqs_mhz.size == 0:
        x0, x1 = 0.0, 1.0
    elif freqs_mhz.size == 1:
        x0, x1 = float(freqs_mhz[0]) - 0.5 * step, float(freqs_mhz[0]) + 0.5 * step
    else:
        x0, x1 = float(freqs_mhz[0]) - 0.5 * step, float(freqs_mhz[-1]) + 0.5 * step
    return [x0, x1, float(record_offset), float(record_offset + records)]


def _response_extent(freqs_mhz: np.ndarray, periods: np.ndarray) -> list[float]:
    step = _freq_step(freqs_mhz)
    if freqs_mhz.size == 0:
        x0, x1 = 0.0, 1.0
    elif freqs_mhz.size == 1:
        x0, x1 = float(freqs_mhz[0]) - 0.5 * step, float(freqs_mhz[0]) + 0.5 * step
    else:
        x0, x1 = float(freqs_mhz[0]) - 0.5 * step, float(freqs_mhz[-1]) + 0.5 * step
    return [x0, x1, float(np.nanmin(periods)), float(np.nanmax(periods))]


def _scalogram_extent(records: int, record_offset: int, periods: np.ndarray) -> list[float]:
    return [
        float(record_offset),
        float(record_offset + records),
        float(np.nanmin(periods)),
        float(np.nanmax(periods)),
    ]


def _linear_edges_from_centers(values: np.ndarray) -> np.ndarray:
    centers = np.asarray(values, dtype=np.float64)
    if centers.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)
    if centers.size == 1:
        width = max(1.0, abs(float(centers[0])) * 0.01)
        return np.array([centers[0] - 0.5 * width, centers[0] + 0.5 * width], dtype=np.float64)
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges


def _period_edges(periods: np.ndarray) -> np.ndarray:
    centers = np.asarray(periods, dtype=np.float64)
    if centers.size == 0:
        return np.array([1.0, 2.0], dtype=np.float64)
    if centers.size == 1:
        width = max(1e-6, abs(float(centers[0])) * 0.1)
        return np.array([max(1e-12, centers[0] - 0.5 * width), centers[0] + 0.5 * width], dtype=np.float64)
    if np.all(centers > 0):
        edges = np.empty(centers.size + 1, dtype=np.float64)
        edges[1:-1] = np.sqrt(centers[:-1] * centers[1:])
        edges[0] = centers[0] * centers[0] / edges[1]
        edges[-1] = centers[-1] * centers[-1] / edges[-2]
        return edges
    return _linear_edges_from_centers(centers)


def _record_edges(records: int, record_offset: int) -> np.ndarray:
    return np.arange(int(record_offset), int(record_offset) + int(records) + 1, dtype=np.float64)


def _new_figure(width: float = 9.0, height: float = 5.4) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    return fig, ax


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=max(72, int(dpi)))
    plt.close(fig)


def _draw_period_rows(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    color: str,
    label: str,
    linewidth: float = 1.2,
    max_rows: int = 100,
) -> None:
    drawn = 0
    for row in rows[:max_rows]:
        f0 = _float(row, "freq_start_mhz")
        f1 = _float(row, "freq_stop_mhz", f0)
        p0 = _float(row, "period_start_records", _float(row, "period_records", _float(row, "peak_period_records")))
        p1 = _float(row, "period_stop_records", p0)
        if not all(math.isfinite(value) for value in [f0, f1, p0, p1]):
            continue
        f0, f1 = sorted([f0, f1])
        p0, p1 = sorted([p0, p1])
        if f1 <= f0:
            x0, x1 = ax.get_xlim()
            width = max(1e-9, abs(x1 - x0) * 0.01)
            f0 -= 0.5 * width
            f1 += 0.5 * width
        if p1 <= p0:
            y0, y1 = ax.get_ylim()
            height = max(1e-9, abs(y1 - y0) * 0.01)
            p0 -= 0.5 * height
            p1 += 0.5 * height
        ax.add_patch(
            Rectangle(
                (f0, p0),
                f1 - f0,
                p1 - p0,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                alpha=0.9,
            )
        )
        drawn += 1
    if drawn:
        ax.plot([], [], color=color, linewidth=linewidth, label=label)


def _draw_time_truth(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    color: str,
    label: str,
    linewidth: float = 1.2,
) -> None:
    drawn = 0
    for row in rows:
        r0 = _float(row, "record_start")
        r1 = _float(row, "record_stop", r0)
        p = _float(row, "period_records", _float(row, "peak_period_records"))
        if not all(math.isfinite(value) for value in [r0, r1, p]):
            continue
        ax.hlines(p, r0, r1, colors=color, linewidth=linewidth, alpha=0.9)
        drawn += 1
    if drawn:
        ax.plot([], [], color=color, linewidth=linewidth, label=label)


def _imshow(
    ax: plt.Axes,
    image: np.ndarray,
    extent: list[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str,
    cbar_label: str,
    yscale: str | None = None,
) -> None:
    vmin, vmax = _limits(image)
    im = ax.imshow(
        image,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if yscale:
        ax.set_yscale(yscale)
    plt.colorbar(im, ax=ax, label=cbar_label)


def _pcolormesh(
    ax: plt.Axes,
    image: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str,
    cbar_label: str,
    yscale: str | None = None,
) -> None:
    vmin, vmax = _limits(image)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        image,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if yscale:
        ax.set_yscale(yscale)
    plt.colorbar(mesh, ax=ax, label=cbar_label)


def _candidate_status_colors(rows: list[dict[str, Any]]) -> list[str]:
    colors = []
    for row in rows:
        status = str(row.get("candidate_status", "raw"))
        if status == "vetoed":
            colors.append("#b23b2e")
        elif status == "needs_validation":
            colors.append("#1f77b4")
        else:
            colors.append("#5c677d")
    return colors


def _sort_candidates(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: _float(row, "peak_score", -math.inf), reverse=True)[: max(0, limit)]


def _representative_channels(
    block_start: int,
    block_stop: int,
    block_freqs: np.ndarray,
    candidates: list[dict[str, Any]],
    max_channels: int,
) -> list[int]:
    channels: list[int] = []
    freqs = np.asarray(block_freqs, dtype=np.float64)
    for row in _sort_candidates(candidates, max_channels * 4):
        peak_freq = _float(row, "peak_freq_mhz")
        if not math.isfinite(peak_freq) or freqs.size == 0:
            continue
        local_channel = int(np.nanargmin(np.abs(freqs - peak_freq)))
        channels.append(block_start + local_channel)
    if not channels:
        span = max(1, block_stop - block_start)
        channels = [block_start + int((idx + 1) * span / (max_channels + 1)) for idx in range(max_channels)]
    unique: list[int] = []
    for channel in channels:
        clamped = min(max(int(channel), block_start), block_stop - 1)
        if clamped not in unique:
            unique.append(clamped)
        if len(unique) >= max(1, max_channels):
            break
    return unique


def visualize_cwt_stages(
    data: np.ndarray,
    freqs_mhz: np.ndarray,
    output_dir: str | Path,
    search_config: SearchVisualizationConfig,
    raw_candidates: list[dict[str, Any]],
    reviewed_candidates: list[dict[str, Any]] | None = None,
    *,
    truths: list[dict[str, Any]] | None = None,
    validation_rows: list[dict[str, Any]] | None = None,
    injection_results: list[dict[str, Any]] | None = None,
    run_id: str = "",
    source_name: str = "",
    record_offset: int = 0,
    config: CWTVisualizationConfig | None = None,
) -> Path:
    config = config or CWTVisualizationConfig(enabled=True)
    matrix = np.asarray(data, dtype=np.float32)
    freqs = np.asarray(freqs_mhz, dtype=np.float64)
    periods = np.asarray(search_config.periods, dtype=np.float64)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reviewed = reviewed_candidates or raw_candidates
    truths = truths or []
    validation_rows = validation_rows or []
    injection_results = injection_results or []
    images: list[tuple[str, Path, str]] = []

    fig, ax = _new_figure()
    _imshow(
        ax,
        matrix,
        _matrix_extent(freqs, matrix.shape[0], record_offset),
        title=f"Stage 01 input matrix: {source_name or run_id}",
        xlabel="Frequency / channel coordinate",
        ylabel="Record",
        cmap="magma",
        cbar_label="amplitude",
    )
    for truth in truths:
        f0 = _float(truth, "freq_start_mhz")
        f1 = _float(truth, "freq_stop_mhz", f0)
        r0 = _float(truth, "record_start")
        r1 = _float(truth, "record_stop", r0)
        if all(math.isfinite(value) for value in [f0, f1, r0, r1]):
            if f1 <= f0:
                f0 -= 0.25
                f1 += 0.25
            ax.add_patch(Rectangle((f0, r0), f1 - f0, r1 - r0, fill=False, edgecolor="#00e5ff", linewidth=1.5))
    if truths:
        ax.plot([], [], color="#00e5ff", linewidth=1.5, label="injection truth")
        ax.legend(loc="upper right")
    path = output_dir / "stage_01_input_matrix.png"
    _save(fig, path, config.dpi)
    images.append(("Stage 01 Input Matrix", path, "Raw time-channel matrix. Cyan boxes mark injected truth spans when available."))

    max_blocks = math.inf if int(config.max_blocks) <= 0 else int(config.max_blocks)
    max_channels = max(1, int(config.max_channels))
    block_count = 0
    for block_index, block_start in enumerate(range(0, matrix.shape[1], search_config.block_channels), start=1):
        if block_count >= max_blocks:
            break
        block_count += 1
        block_stop = min(block_start + int(search_config.block_channels), matrix.shape[1])
        block_data = matrix[:, block_start:block_stop]
        block_freqs = freqs[block_start:block_stop]
        block_id = f"block_{block_index:04d}"
        power = cwt_power_cube(
            block_data,
            periods,
            wavelet=search_config.wavelet,
            method=search_config.cwt_method,
            normalize_channels=True,
        )
        response = aggregate_cwt_time(
            power,
            method=search_config.time_aggregation,
            percentile=search_config.aggregation_percentile,
        )
        score = channel_period_peak_score(
            np.log10(response + 1e-12),
            sigma_peak=search_config.dog_sigma_peak,
            sigma_background=search_config.dog_sigma_background,
        )
        block_rows = [row for row in raw_candidates if str(row.get("block_id", "")) == block_id]
        representative = _representative_channels(block_start, block_stop, block_freqs, block_rows, max_channels)

        for global_channel in representative:
            local_channel = global_channel - block_start
            if local_channel < 0 or local_channel >= block_freqs.size:
                continue
            scalogram = np.log10(power[:, :, local_channel] + 1e-12)
            fig, ax = _new_figure()
            _pcolormesh(
                ax,
                scalogram,
                _record_edges(matrix.shape[0], record_offset),
                _period_edges(periods),
                title=f"Stage 02 CWT scalogram: {block_id}, channel {global_channel}",
                xlabel="Record",
                ylabel="Period / records",
                cmap="inferno",
                cbar_label="log10(CWT power)",
                yscale="log",
            )
            channel_truths = [
                row for row in truths
                if _float(row, "channel_start", -1) <= global_channel < _float(row, "channel_stop", -1)
            ]
            _draw_time_truth(ax, channel_truths, color="#00e5ff", label="truth period", linewidth=1.5)
            if channel_truths:
                ax.legend(loc="best")
            path = output_dir / f"stage_02_{block_id}_channel_{global_channel:04d}_scalogram.png"
            _save(fig, path, config.dpi)
            images.append((f"Stage 02 CWT Scalogram {block_id} Ch {global_channel}", path, "Full period-time CWT power for one representative frequency channel before time aggregation."))

        fig, ax = _new_figure()
        _pcolormesh(
            ax,
            np.log10(response + 1e-12),
            _linear_edges_from_centers(block_freqs),
            _period_edges(periods),
            title=f"Stage 03 period-channel response: {block_id}",
            xlabel="Frequency / channel coordinate",
            ylabel="Period / records",
            cmap="magma",
            cbar_label=f"log10({search_config.time_aggregation} CWT power)",
            yscale="log",
        )
        _draw_period_rows(ax, truths, color="#00e5ff", label="truth", linewidth=1.5)
        if truths:
            ax.legend(loc="best")
        path = output_dir / f"stage_03_{block_id}_period_channel_response.png"
        _save(fig, path, config.dpi)
        images.append((f"Stage 03 Period-Channel Response {block_id}", path, "CWT power after time aggregation, forming the period-channel candidate map."))

        fig, ax = _new_figure()
        _pcolormesh(
            ax,
            score,
            _linear_edges_from_centers(block_freqs),
            _period_edges(periods),
                title=f"Stage 04 channel period-peak score: {block_id}",
            xlabel="Frequency / channel coordinate",
            ylabel="Period / records",
            cmap="viridis",
                cbar_label="channel DoG robust score",
            yscale="log",
        )
        ax.contour(
            block_freqs,
            periods,
            score,
            levels=[float(search_config.threshold)],
            colors="white",
            linewidths=0.7,
        )
        _draw_period_rows(ax, _sort_candidates(block_rows, config.top_candidates), color="#ffdf4d", label="candidate")
        _draw_period_rows(ax, truths, color="#00e5ff", label="truth", linewidth=1.5)
        if block_rows or truths:
            ax.legend(loc="best")
        path = output_dir / f"stage_04_{block_id}_period_channel_candidates.png"
        _save(fig, path, config.dpi)
        images.append((f"Stage 04 Period-Channel Candidate Overlay {block_id}", path, "Channel-wise period-peak score map with threshold contour, candidates, and optional truth."))

    if reviewed:
        top_rows = _sort_candidates(reviewed, config.top_candidates)
        x = [_float(row, "peak_freq_mhz") for row in top_rows]
        y = [_float(row, "peak_period_records") for row in top_rows]
        size = [max(15.0, 12.0 * _float(row, "peak_score", 1.0)) for row in top_rows]
        colors = _candidate_status_colors(top_rows)
        fig, ax = _new_figure()
        ax.scatter(x, y, s=size, c=colors, alpha=0.75, edgecolors="black", linewidths=0.3)
        _draw_period_rows(ax, truths, color="#00a6c8", label="truth", linewidth=1.4)
        ax.set_title("Stage 05 candidate review overview")
        ax.set_xlabel("Frequency / channel coordinate")
        ax.set_ylabel("Period / records")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.plot([], [], "o", color="#1f77b4", label="needs_validation")
        ax.plot([], [], "o", color="#b23b2e", label="vetoed")
        if truths:
            ax.plot([], [], color="#00a6c8", label="truth")
        ax.legend(loc="best")
        path = output_dir / "stage_05_candidate_review_overview.png"
        _save(fig, path, config.dpi)
        images.append(("Stage 05 Candidate Review Overview", path, "Top period-channel candidates after veto review, colored by candidate status and scaled by peak score."))

    if validation_rows:
        rows = sorted(validation_rows, key=lambda row: _float(row, "evidence_rank", math.inf))
        candidate_ids = [_float(row, "candidate_id") for row in rows]
        qvalues = [_float(row, "global_q_value") for row in rows]
        periods_refined = [_float(row, "refined_period_records") for row in rows]
        fig, ax1 = _new_figure()
        ax1.scatter(candidate_ids, qvalues, c="#1f77b4", label="global q-value")
        ax1.set_yscale("log")
        ax1.set_xlabel("candidate_id")
        ax1.set_ylabel("global q-value")
        ax1.grid(alpha=0.25)
        ax2 = ax1.twinx()
        ax2.scatter(candidate_ids, periods_refined, c="#e07a2f", marker="x", label="refined period")
        ax2.set_ylabel("refined period / records")
        ax1.set_title("Stage 06 validation/statistics overview")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")
        path = output_dir / "stage_06_validation_overview.png"
        _save(fig, path, config.dpi)
        images.append(("Stage 06 Validation Overview", path, "Global q-values and refined periods for reviewed validation rows."))

    if injection_results:
        detected_raw = sum(1 for row in injection_results if _bool_value(row.get("detected_raw")))
        detected_after_veto = sum(1 for row in injection_results if _bool_value(row.get("detected_after_veto")))
        validated = sum(1 for row in injection_results if _bool_value(row.get("validated")))
        total = max(1, len(injection_results))
        labels = ["raw", "after veto", "validated"]
        rates = [detected_raw / total, detected_after_veto / total, validated / total]
        fig, ax = _new_figure()
        ax.bar(labels, rates, color=["#5b8bd9", "#49a078", "#d97941"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("recovery rate")
        ax.set_title("Stage 07 injection recovery")
        for idx, rate in enumerate(rates):
            ax.text(idx, rate + 0.025, f"{rate:.2f}", ha="center", va="bottom")
        path = output_dir / "stage_07_injection_recovery.png"
        _save(fig, path, config.dpi)
        images.append(("Stage 07 Injection Recovery", path, "Detection, after-veto, and validation recovery rates for the injection run."))

        truth_periods = [_float(row, "period_records") for row in injection_results]
        refined_periods = [_float(row, "refined_period_records") for row in injection_results]
        stages = [str(row.get("failure_stage", "")) for row in injection_results]
        stage_colors = ["#2f855a" if stage == "validated" else "#c2410c" for stage in stages]
        fig, ax = _new_figure()
        ax.scatter(truth_periods, refined_periods, c=stage_colors, s=55, edgecolors="black", linewidths=0.4)
        finite = [value for value in truth_periods + refined_periods if math.isfinite(value)]
        if finite:
            lo, hi = min(finite), max(finite)
            ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("injected period / records")
        ax.set_ylabel("refined period / records")
        ax.set_title("Stage 08 injection period recovery")
        ax.grid(alpha=0.25)
        path = output_dir / "stage_08_injection_period_recovery.png"
        _save(fig, path, config.dpi)
        images.append(("Stage 08 Injection Period Recovery", path, "Injected period versus validation-refined period; dashed line is perfect recovery."))

    index_path = output_dir / "index.md"
    lines = [
        f"# Stage Visualization: {run_id or source_name or output_dir.parent.name}",
        "",
        "These figures are diagnostic review products. They do not claim a confirmed periodic signal.",
        "",
        "## Files",
        "",
    ]
    for title, path, note in images:
        rel = path.name
        lines.extend([f"### {title}", "", note, "", f"![{title}]({rel})", ""])
    lines.append("## Metadata")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(
            {
                "run_id": run_id,
                "source_name": source_name,
                "record_offset": record_offset,
                "matrix_shape": list(matrix.shape),
                "visualization_config": config.__dict__,
                "search_config": {
                    "wavelet": search_config.wavelet,
                    "cwt_method": search_config.cwt_method,
                    "period_count": int(periods.size),
                    "period_min_records": float(np.nanmin(periods)),
                    "period_max_records": float(np.nanmax(periods)),
                    "block_channels": search_config.block_channels,
                    "threshold": search_config.threshold,
                    "min_prominence": search_config.min_prominence,
                    "dog_sigma_peak": search_config.dog_sigma_peak,
                    "dog_sigma_background": search_config.dog_sigma_background,
                    "time_aggregation": search_config.time_aggregation,
                },
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    lines.append("```")
    index_path.write_text("\n".join(lines) + "\n")
    return index_path
