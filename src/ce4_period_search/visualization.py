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

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "swt_period_search_matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from .detection import robust_score_2d
from .swt import swt_detail_power_matrix


@dataclass(frozen=True)
class VisualizationConfig:
    enabled: bool = False
    max_blocks: int = 2
    max_levels: int = 3
    top_candidates: int = 50
    dpi: int = 140


@dataclass(frozen=True)
class SearchVisualizationConfig:
    wavelet: str = "db4"
    levels: int = 5
    block_channels: int = 128
    threshold: float = 5.0
    local_time: int = 513
    local_freq: int = 9


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


def _extent(freqs_mhz: np.ndarray, records: int, record_offset: int) -> list[float]:
    step = _freq_step(freqs_mhz)
    if freqs_mhz.size == 0:
        x0, x1 = 0.0, 1.0
    elif freqs_mhz.size == 1:
        x0, x1 = float(freqs_mhz[0]) - 0.5 * step, float(freqs_mhz[0]) + 0.5 * step
    else:
        x0, x1 = float(freqs_mhz[0]) - 0.5 * step, float(freqs_mhz[-1]) + 0.5 * step
    return [x0, x1, float(record_offset), float(record_offset + records)]


def _new_figure(width: float = 9.0, height: float = 5.4) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    return fig, ax


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=max(72, int(dpi)))
    plt.close(fig)


def _draw_rows(
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
        r0 = _float(row, "record_start")
        r1 = _float(row, "record_stop", r0 + 1)
        if not all(math.isfinite(value) for value in [f0, f1, r0, r1]):
            continue
        f0, f1 = sorted([f0, f1])
        r0, r1 = sorted([r0, r1])
        if f1 <= f0:
            f1 = f0 + 1e-9
        if r1 <= r0:
            r1 = r0 + 1.0
        ax.add_patch(
            Rectangle(
                (f0, r0),
                f1 - f0,
                r1 - r0,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                alpha=0.9,
            )
        )
        drawn += 1
    if drawn:
        ax.plot([], [], color=color, linewidth=linewidth, label=label)


def _imshow(
    ax: plt.Axes,
    image: np.ndarray,
    freqs_mhz: np.ndarray,
    record_offset: int,
    *,
    title: str,
    cmap: str,
    cbar_label: str,
) -> None:
    vmin, vmax = _limits(image)
    im = ax.imshow(
        image,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=_extent(freqs_mhz, image.shape[0], record_offset),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Frequency / channel coordinate")
    ax.set_ylabel("Record")
    plt.colorbar(im, ax=ax, label=cbar_label)


def _candidate_status_colors(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    colors = []
    labels = []
    for row in rows:
        status = str(row.get("candidate_status", "raw"))
        if status == "vetoed":
            colors.append("#b23b2e")
        elif status == "needs_validation":
            colors.append("#1f77b4")
        else:
            colors.append("#5c677d")
        labels.append(status)
    return colors, labels


def _sort_candidates(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: _float(row, "peak_score", -math.inf), reverse=True)[: max(0, limit)]


def visualize_matrix_stages(
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
    config: VisualizationConfig | None = None,
) -> Path:
    config = config or VisualizationConfig(enabled=True)
    matrix = np.asarray(data, dtype=np.float32)
    freqs = np.asarray(freqs_mhz, dtype=np.float64)
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
        freqs,
        record_offset,
        title=f"Stage 01 input matrix: {source_name or run_id}",
        cmap="magma",
        cbar_label="amplitude",
    )
    _draw_rows(ax, truths, color="#00e5ff", label="injection truth", linewidth=1.5)
    if truths:
        ax.legend(loc="upper right")
    path = output_dir / "stage_01_input_matrix.png"
    _save(fig, path, config.dpi)
    images.append(("Stage 01 Input Matrix", path, "Raw time-channel matrix. Cyan boxes mark injected truth spans when available."))

    max_blocks = math.inf if int(config.max_blocks) <= 0 else int(config.max_blocks)
    max_levels = math.inf if int(config.max_levels) <= 0 else int(config.max_levels)
    block_count = 0
    for block_index, block_start in enumerate(range(0, matrix.shape[1], search_config.block_channels), start=1):
        if block_count >= max_blocks:
            break
        block_count += 1
        block_stop = min(block_start + int(search_config.block_channels), matrix.shape[1])
        block_data = matrix[:, block_start:block_stop]
        block_freqs = freqs[block_start:block_stop]
        block_id = f"block_{block_index:04d}"
        powers, level_numbers = swt_detail_power_matrix(
            block_data,
            wavelet=search_config.wavelet,
            levels=search_config.levels,
            normalize_channels=True,
        )
        level_count = 0
        for level_idx, level_number in enumerate(level_numbers):
            if level_count >= max_levels:
                break
            level_count += 1
            log_power = np.log10(powers[level_idx] + 1e-12)
            score = robust_score_2d(
                log_power,
                local_time=search_config.local_time,
                local_freq=min(search_config.local_freq, max(3, block_freqs.size | 1)),
            )
            level_rows = [
                row for row in raw_candidates
                if str(row.get("block_id", "")) == block_id
                and int(_float(row, "swt_level", -1)) == int(level_number)
            ]

            fig, ax = _new_figure()
            _imshow(
                ax,
                log_power,
                block_freqs,
                record_offset,
                title=f"Stage 02 SWT log-power: {block_id}, level {int(level_number)}",
                cmap="inferno",
                cbar_label="log10(detail power)",
            )
            path = output_dir / f"stage_02_{block_id}_level_{int(level_number)}_power.png"
            _save(fig, path, config.dpi)
            images.append((f"Stage 02 SWT Power {block_id} L{int(level_number)}", path, "SWT detail-power map before local robust S/N."))

            fig, ax = _new_figure()
            _imshow(
                ax,
                score,
                block_freqs,
                record_offset,
                title=f"Stage 03 local robust S/N: {block_id}, level {int(level_number)}",
                cmap="viridis",
                cbar_label="local robust S/N",
            )
            ax.contour(
                block_freqs,
                np.arange(record_offset, record_offset + score.shape[0]),
                score,
                levels=[float(search_config.threshold)],
                colors="white",
                linewidths=0.7,
            )
            path = output_dir / f"stage_03_{block_id}_level_{int(level_number)}_snr.png"
            _save(fig, path, config.dpi)
            images.append((f"Stage 03 Local S/N {block_id} L{int(level_number)}", path, "Local robust S/N map; white contour is the detection threshold."))

            fig, ax = _new_figure()
            _imshow(
                ax,
                score,
                block_freqs,
                record_offset,
                title=f"Stage 04 candidates: {block_id}, level {int(level_number)}",
                cmap="viridis",
                cbar_label="local robust S/N",
            )
            _draw_rows(ax, _sort_candidates(level_rows, config.top_candidates), color="#ffdf4d", label="candidate")
            _draw_rows(ax, truths, color="#00e5ff", label="truth", linewidth=1.5)
            if level_rows or truths:
                ax.legend(loc="upper right")
            path = output_dir / f"stage_04_{block_id}_level_{int(level_number)}_candidates.png"
            _save(fig, path, config.dpi)
            images.append((f"Stage 04 Candidate Overlay {block_id} L{int(level_number)}", path, "Candidate boxes over the local S/N map; truth boxes are cyan when available."))

    if reviewed:
        top_rows = _sort_candidates(reviewed, config.top_candidates)
        x = [_float(row, "peak_freq_mhz") for row in top_rows]
        y = [_float(row, "peak_record") for row in top_rows]
        size = [max(15.0, 12.0 * _float(row, "peak_score", 1.0)) for row in top_rows]
        colors, _labels = _candidate_status_colors(top_rows)
        fig, ax = _new_figure()
        ax.scatter(x, y, s=size, c=colors, alpha=0.75, edgecolors="black", linewidths=0.3)
        _draw_rows(ax, truths, color="#00a6c8", label="truth", linewidth=1.4)
        ax.set_title("Stage 05 candidate review overview")
        ax.set_xlabel("Frequency / channel coordinate")
        ax.set_ylabel("Record")
        ax.grid(alpha=0.25)
        ax.plot([], [], "o", color="#1f77b4", label="needs_validation")
        ax.plot([], [], "o", color="#b23b2e", label="vetoed")
        if truths:
            ax.plot([], [], color="#00a6c8", label="truth")
        ax.legend(loc="best")
        path = output_dir / "stage_05_candidate_review_overview.png"
        _save(fig, path, config.dpi)
        images.append(("Stage 05 Candidate Review Overview", path, "Top candidates after veto review, colored by candidate status and scaled by peak score."))

    if validation_rows:
        rows = sorted(validation_rows, key=lambda row: _float(row, "evidence_rank", math.inf))
        candidate_ids = [_float(row, "candidate_id") for row in rows]
        qvalues = [_float(row, "global_q_value") for row in rows]
        periods = [_float(row, "refined_period_records") for row in rows]
        fig, ax1 = _new_figure()
        ax1.scatter(candidate_ids, qvalues, c="#1f77b4", label="global q-value")
        ax1.set_yscale("log")
        ax1.set_xlabel("candidate_id")
        ax1.set_ylabel("global q-value")
        ax1.grid(alpha=0.25)
        ax2 = ax1.twinx()
        ax2.scatter(candidate_ids, periods, c="#e07a2f", marker="x", label="refined period")
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
                "search_config": search_config.__dict__,
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    lines.append("```")
    index_path.write_text("\n".join(lines) + "\n")
    return index_path

