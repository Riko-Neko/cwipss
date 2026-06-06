"""Staged search and benchmark visualizations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from matplotlib import pyplot as plt

from ..signal.activity import (
    coherent_structure_map,
    crop_valid_periods,
    low_fraction_noise_floor,
    relative_excess,
    robust_standardize,
    signed_trimmed_period_activity,
    smooth_activity,
)
from ..signal.cwt import aggregate_cwt_time, cwt_power_cube
from ..signal.profile import windowed_period_profile
from .plotting import (
    CANDIDATE_COLOR,
    TRUTH_COLOR,
    VETO_COLOR,
    ImageIndex,
    cwt_view,
    edges,
    heatmap,
    number,
    raw_view,
    row_boxes,
    save_figure,
)


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
    cwt_backend: str = "cpu"
    cuda_device: int = 0
    block_channels: int = 128
    candidate_period_min_records: float | None = 10.0
    candidate_period_max_records: float | None = 200.0
    time_aggregation: str = "p95"
    aggregation_percentile: float = 95.0
    noise_floor_fraction: float = 0.20
    excess_eps_fraction: float = 1e-6
    structure_baseline_quantile: float = 0.10
    structure_scale_quantile: float = 0.20
    structure_z_threshold: float = 1.0
    structure_time_support_records: int = 64
    structure_period_support_bins: int = 3
    structure_min_support_fraction: float = 0.10
    activity_trim_low: float = 0.05
    activity_trim_high: float = 0.95
    activity_smooth_records: int = 16


def _top(rows: Iterable[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (number(row, "integrated_score", -math.inf), number(row, "peak_score", -math.inf)),
        reverse=True,
    )[:count]


def _channels(start: int, stop: int, freqs: np.ndarray, rows: list[Mapping[str, Any]], count: int) -> list[int]:
    if count <= 0:
        return list(range(start, stop))
    selected = [
        start + int(np.nanargmin(abs(freqs - peak)))
        for row in _top(rows, count * 4)
        if math.isfinite(peak := number(row, "peak_freq_mhz"))
    ]
    if not selected:
        selected = [start + (index + 1) * (stop - start) // (count + 1) for index in range(count)]
    return list(dict.fromkeys(min(max(channel, start), stop - 1) for channel in selected))[:count]


def _structure(power: np.ndarray, periods: np.ndarray, cfg: SearchVisualizationConfig):
    power, valid, _ = crop_valid_periods(
        power, periods, cfg.candidate_period_min_records, cfg.candidate_period_max_records
    )
    floor = low_fraction_noise_floor(power, fraction=cfg.noise_floor_fraction)
    structured = coherent_structure_map(
        relative_excess(power, floor, eps_fraction=cfg.excess_eps_fraction),
        baseline_quantile=cfg.structure_baseline_quantile,
        scale_quantile=cfg.structure_scale_quantile,
        z_threshold=cfg.structure_z_threshold,
        time_support_records=cfg.structure_time_support_records,
        period_support_bins=cfg.structure_period_support_bins,
        min_support_fraction=cfg.structure_min_support_fraction,
    )
    activity = signed_trimmed_period_activity(
        structured, trim_low=cfg.activity_trim_low, trim_high=cfg.activity_trim_high
    )
    return valid, structured, robust_standardize(smooth_activity(activity, cfg.activity_smooth_records)), floor


def _simple_plot(path: Path, dpi: int, draw) -> None:
    def wrapped(ax):
        draw(ax)
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize="small")

    save_figure(path, dpi, wrapped)


def _activity_plot(path, activity, offset, windows, truths, title, dpi):
    def draw(ax):
        ax.plot(np.arange(offset, offset + len(activity)), activity, color="#0f172a", label="activity")
        ax.axhline(0, color="#6b7280", linestyle="--", linewidth=0.8)
        for rows, color, label, alpha in (
            (windows, CANDIDATE_COLOR, "PELT window", 0.18),
            (truths, TRUTH_COLOR, "truth span", 0.12),
        ):
            spans = [(number(row, "record_start"), number(row, "record_stop")) for row in rows]
            for start, stop in spans:
                if math.isfinite(start) and math.isfinite(stop) and stop > start:
                    ax.axvspan(start, stop, color=color, alpha=alpha)
            if spans:
                ax.plot([], [], color=color, linewidth=4, alpha=0.6, label=label)
        ax.set(title=title, xlabel="Record", ylabel="Activity / robust z")

    _simple_plot(path, dpi, draw)


def _profile_plot(path, structured, periods, rows, truths, offset, title, limit, dpi):
    def draw(ax):
        for row in _top(rows, min(5, limit)):
            start = max(0, int(number(row, "record_start", offset) - offset))
            stop = min(structured.shape[1], int(number(row, "record_stop", offset) - offset))
            if stop > start:
                ax.plot(periods, windowed_period_profile(structured, start, stop), label=f"cand {row.get('candidate_id')}")
        for row in truths:
            if math.isfinite(period := number(row, "period_records")):
                ax.axvline(period, color=TRUTH_COLOR, linestyle="--")
        ax.set(title=title, xlabel="Period / records", ylabel="Windowed structure profile", xscale="log")

    _simple_plot(path, dpi, draw)


def _review_plot(path, rows, dpi):
    def draw(ax):
        scores = np.array([max(0, number(row, "integrated_score", number(row, "peak_score", 0))) for row in rows])
        sizes = 20 + 180 * np.sqrt(scores / scores.max()) if scores.size and scores.max() else np.full(len(rows), 20)
        colors = [VETO_COLOR if row.get("candidate_status") == "vetoed" else CANDIDATE_COLOR for row in rows]
        ax.scatter(
            [number(row, "peak_freq_mhz") for row in rows],
            [number(row, "peak_period_records") for row in rows],
            s=sizes, c=colors, edgecolors="black", linewidths=0.3,
        )
        ax.plot([], [], "o", color=CANDIDATE_COLOR, label="needs_validation")
        ax.plot([], [], "o", color=VETO_COLOR, label="vetoed")
        ax.set(title="Stage 08 candidate review overview", xlabel="Frequency / MHz", ylabel="Period / records", yscale="log")

    _simple_plot(path, dpi, draw)


def _validation_plot(path, rows, dpi):
    rows = sorted(rows, key=lambda row: number(row, "evidence_rank", math.inf))

    def draw(ax):
        ids = [number(row, "candidate_id") for row in rows]
        ax.scatter(ids, [number(row, "global_q_value") for row in rows], label="global q-value")
        ax.set(title="Stage 09 validation/statistics overview", xlabel="candidate_id", ylabel="global q-value", yscale="log")
        twin = ax.twinx()
        twin.scatter(ids, [number(row, "refined_period_records") for row in rows], color="#e07a2f", marker="x", label="refined period")
        twin.set_ylabel("refined period / records")

    _simple_plot(path, dpi, draw)


def _injection_plots(output: Path, rows, index: ImageIndex, dpi: int):
    rates = [
        sum(str(row.get(key)).lower() in {"1", "true", "yes"} for row in rows) / max(1, len(rows))
        for key in ("detected_raw", "detected_after_veto", "validated")
    ]
    path = output / "stage_10_injection_recovery.png"

    def recovery(ax):
        ax.bar(["raw", "after veto", "validated"], rates)
        ax.set(title="Stage 10 injection recovery", ylabel="recovery rate", ylim=(0, 1.05))

    _simple_plot(path, dpi, recovery)
    index.add("Stage 10 Injection Recovery", path, "Injection recovery rates.")
    path = output / "stage_11_injection_period_recovery.png"

    def periods(ax):
        truth = [number(row, "period_records") for row in rows]
        refined = [number(row, "refined_period_records") for row in rows]
        ax.scatter(truth, refined)
        finite = [value for value in [*truth, *refined] if math.isfinite(value)]
        if finite:
            ax.plot([min(finite), max(finite)], [min(finite), max(finite)], "k--")
        ax.set(title="Stage 11 injection period recovery", xlabel="injected period / records", ylabel="refined period / records")

    _simple_plot(path, dpi, periods)
    index.add("Stage 11 Injection Period Recovery", path, "Injected versus refined period.")


def _add(index, title, path, note):
    index.add(title, path, note)
    return path


def visualize_cwt_stages(
    data: np.ndarray,
    freqs_mhz: np.ndarray,
    output_dir: str | Path,
    search_config: SearchVisualizationConfig,
    raw_candidates: list[dict[str, Any]],
    reviewed_candidates: list[dict[str, Any]] | None = None,
    time_windows: list[dict[str, Any]] | None = None,
    *,
    truths: list[dict[str, Any]] | None = None,
    validation_rows: list[dict[str, Any]] | None = None,
    injection_results: list[dict[str, Any]] | None = None,
    run_id: str = "",
    source_name: str = "",
    record_offset: int = 0,
    config: CWTVisualizationConfig | None = None,
) -> Path:
    cfg = config or CWTVisualizationConfig(enabled=True)
    matrix, freqs, periods = np.asarray(data), np.asarray(freqs_mhz), np.asarray(search_config.periods)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for path in output.glob("stage_*.png"):
        path.unlink()
    truths, windows = truths or [], time_windows or []
    index = ImageIndex(output, f"Stage Visualization: {run_id or source_name or output.parent.name}")

    path = output / "stage_01_input_matrix.png"
    raw_view(path, matrix, freqs, offset=record_offset, title=f"Stage 01 input matrix: {source_name or run_id}", truths=truths, dpi=cfg.dpi)
    _add(index, "Stage 01 Input Matrix", path, "Raw time-channel matrix.")

    block_limit = math.inf if cfg.max_blocks <= 0 else cfg.max_blocks
    for block_no, start in enumerate(range(0, matrix.shape[1], search_config.block_channels), 1):
        if block_no > block_limit:
            break
        stop, block_id = min(start + search_config.block_channels, matrix.shape[1]), f"block_{block_no:04d}"
        block, block_freqs = matrix[:, start:stop], freqs[start:stop]
        power = cwt_power_cube(
            block, periods, wavelet=search_config.wavelet, method=search_config.cwt_method,
            backend=search_config.cwt_backend, cuda_device=search_config.cuda_device, normalize_channels=True,
        )
        response = aggregate_cwt_time(power, search_config.time_aggregation, search_config.aggregation_percentile)
        block_rows = [row for row in raw_candidates if row.get("block_id") == block_id]

        for channel in _channels(start, stop, block_freqs, block_rows, cfg.max_channels):
            local = channel - start
            rows = [row for row in block_rows if int(number(row, "channel_index", -1)) == local]
            channel_windows = [
                row for row in windows
                if row.get("block_id") == block_id and int(number(row, "channel_index", -1)) == local
            ]
            channel_truths = [
                row for row in truths
                if number(row, "channel_start", -1) <= channel < number(row, "channel_stop", -1)
            ]
            prefix = f"{block_id}_channel_{channel:04d}"
            path = output / f"stage_02_{prefix}_scalogram.png"
            cwt_view(path, power[:, :, local], periods, offset=record_offset, title=f"Stage 02 CWT scalogram: {block_id}, channel {channel}", candidates=rows, truths=channel_truths, dpi=cfg.dpi)
            _add(index, f"Stage 02 CWT Scalogram {block_id} Ch {channel}", path, "Full period-time CWT power.")

            valid, structured, activity, floor = _structure(power[:, :, local], periods, search_config)
            path = output / f"stage_03_{prefix}_structure_map.png"
            cwt_view(
                path, structured, valid, offset=record_offset,
                title=f"Stage 03 structure-gated CWT map: {block_id}, channel {channel}",
                candidates=rows, truths=channel_truths, cmap="viridis",
                colorbar=f"positive period-z x support (floor={floor:.4g})", log_power=False, dpi=cfg.dpi,
            )
            _add(index, f"Stage 03 Structure-Gated CWT Map {block_id} Ch {channel}", path, "Structure-gated CWT.")
            path = output / f"stage_04_{prefix}_activity_windows.png"
            _activity_plot(path, activity, record_offset, channel_windows, channel_truths, f"Stage 04 activity windows: {block_id}, channel {channel}", cfg.dpi)
            _add(index, f"Stage 04 Activity Windows {block_id} Ch {channel}", path, "PELT activity curve.")
            if rows:
                path = output / f"stage_05_{prefix}_period_profiles.png"
                _profile_plot(path, structured, valid, rows, channel_truths, record_offset, f"Stage 05 period profiles: {block_id}, channel {channel}", cfg.top_candidates, cfg.dpi)
                _add(index, f"Stage 05 Windowed Period Profiles {block_id} Ch {channel}", path, "Windowed period profiles.")

        shaded = []
        if search_config.candidate_period_min_records is not None:
            shaded.append((periods.min(), search_config.candidate_period_min_records))
        if search_config.candidate_period_max_records is not None:
            shaded.append((search_config.candidate_period_max_records, periods.max()))
        axes = ("freq_start_mhz", "freq_stop_mhz", "peak_freq_mhz"), ("period_start_records", "period_stop_records", "period_records", "peak_period_records")
        truth_boxes = row_boxes(truths, *axes, color=TRUTH_COLOR, label="truth", fallback=(1e-6, 1e-6))
        for stage, values, cmap, boxes, note in (
            (6, np.log10(response + 1e-12), "magma", truth_boxes, "Time-aggregated CWT overview."),
            (
                7,
                np.where(
                    ((periods >= (search_config.candidate_period_min_records or -math.inf)) &
                     (periods <= (search_config.candidate_period_max_records or math.inf)))[:, None],
                    np.log10(response + 1e-12), np.nan,
                ),
                "viridis",
                row_boxes(_top(block_rows, cfg.top_candidates), *axes, color=CANDIDATE_COLOR, label="candidate", fallback=(1e-6, 1e-6)) + truth_boxes,
                "Candidate-period CWT overview.",
            ),
        ):
            path = output / f"stage_{stage:02d}_{block_id}_period_channel_{'response' if stage == 6 else 'candidates'}.png"
            heatmap(
                path, values, edges(block_freqs), edges(periods, True),
                title=f"Stage {stage:02d} period-channel overview: {block_id}",
                xlabel="Frequency / MHz", ylabel="Period / records",
                colorbar=f"log10({search_config.time_aggregation} CWT power)",
                cmap=cmap, yscale="log", boxes=boxes, shaded=shaded, dpi=cfg.dpi,
            )
            _add(index, f"Stage {stage:02d} Period-Channel Overview {block_id}", path, note)

    reviewed = reviewed_candidates or raw_candidates
    if reviewed:
        path = output / "stage_08_candidate_review_overview.png"
        _review_plot(path, _top(reviewed, cfg.top_candidates), cfg.dpi)
        _add(index, "Stage 08 Candidate Review Overview", path, "Candidates after veto review.")
    if validation_rows:
        path = output / "stage_09_validation_overview.png"
        _validation_plot(path, validation_rows, cfg.dpi)
        _add(index, "Stage 09 Validation Overview", path, "Validation statistics.")
    if injection_results:
        _injection_plots(output, injection_results, index, cfg.dpi)
    index.metadata = {
        "run_id": run_id, "source_name": source_name, "matrix_shape": list(matrix.shape),
        "visualization_config": cfg.__dict__,
    }
    return index.write()
