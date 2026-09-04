"""Staged search and benchmark visualizations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from matplotlib import pyplot as plt

from ..signal.cpro import (
    CPROParameters,
    cpro_activity,
    cpro_period_mask,
    difference_noise_std,
    impulse_cwt_noise_gain,
)
from ..signal.cprf import CPRFParameters, evaluate_cprf, normalize_cwt_power
from ..signal.cwt import aggregate_cwt_time, cwt_power_cube
from .plotting import (
    CANDIDATE_COLOR,
    CWT_POWER_CMAP,
    TRUTH_COLOR,
    VETO_COLOR,
    ImageIndex,
    cwt_power_display_values,
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
    cpro_threshold_snr: float = 32.0
    cpro_texture_quantile: float = 0.9375
    cpro_period_center_bins: int = 3
    cpro_period_context_bins: int = 15
    cpro_min_period_contrast: float = 1.5
    cpro_period_support_bins: int = 3
    cpro_shape_power_softness: float = 1.0
    cpro_shape_contrast_softness: float = 0.10
    cpro_continuity_decay: float = 0.995
    cpro_continuity_power: float = 2.0
    cpro_min_continuity_mean: float = 0.47
    cpro_min_ridge_lock: float = 0.94
    cprf_threshold_snr: float = 32.0
    cprf_texture_quantile: float = 0.9375
    cprf_smooth_bins: int = 3
    cprf_peak_band_fraction: float = 0.50
    cprf_min_width_bins: int = 3
    cprf_min_peak_strength: float = 1.25
    cprf_min_integrated_strength: float = 0.0
    cprf_min_band_persistence: float = 0.40
    cprf_min_band_concentration: float = 0.50
    cprf_min_local_contrast: float = 1.20
    cprf_harmonic_weight: float = 0.20
    cprf_harmonic_min_relative: float = 0.12
    cprf_harmonic_window_scale: float = 1.25
    cprf_max_peak_hypotheses: int = 8


def _cprf_parameters(cfg: SearchVisualizationConfig) -> CPRFParameters:
    return CPRFParameters(
        threshold_snr=cfg.cprf_threshold_snr,
        texture_quantile=cfg.cprf_texture_quantile,
        smooth_bins=cfg.cprf_smooth_bins,
        peak_band_fraction=cfg.cprf_peak_band_fraction,
        min_width_bins=cfg.cprf_min_width_bins,
        min_peak_strength=cfg.cprf_min_peak_strength,
        min_integrated_strength=cfg.cprf_min_integrated_strength,
        min_band_persistence=cfg.cprf_min_band_persistence,
        min_band_concentration=cfg.cprf_min_band_concentration,
        min_local_contrast=cfg.cprf_min_local_contrast,
        harmonic_weight=cfg.cprf_harmonic_weight,
        harmonic_min_relative=cfg.cprf_harmonic_min_relative,
        harmonic_window_scale=cfg.cprf_harmonic_window_scale,
        max_peak_hypotheses=cfg.cprf_max_peak_hypotheses,
    )


def _top(rows: Iterable[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: number(row, "score", -math.inf),
        reverse=True,
    )[:count]


def _channels(start: int, stop: int, freqs: np.ndarray, rows: list[Mapping[str, Any]], count: int) -> list[int]:
    if count <= 0:
        return list(range(start, stop))
    selected = [
        start + int(np.nanargmin(abs(freqs - peak)))
        for row in _top(rows, count * 4)
        if math.isfinite(peak := number(row, "freq_mhz"))
    ]
    if not selected:
        selected = [start + (index + 1) * (stop - start) // (count + 1) for index in range(count)]
    return list(dict.fromkeys(min(max(channel, start), stop - 1) for channel in selected))[:count]


def _cpro_products(
    power: np.ndarray,
    raw_series: np.ndarray,
    periods: np.ndarray,
    target: int,
    noise_gain: np.ndarray,
    cfg: SearchVisualizationConfig,
):
    valid = cpro_period_mask(
        periods,
        cfg.candidate_period_min_records,
        cfg.candidate_period_max_records,
    )
    params = CPROParameters(
        threshold_snr=cfg.cpro_threshold_snr,
        texture_quantile=cfg.cpro_texture_quantile,
        period_center_bins=cfg.cpro_period_center_bins,
        period_context_bins=cfg.cpro_period_context_bins,
        min_period_contrast=cfg.cpro_min_period_contrast,
        period_support_bins=cfg.cpro_period_support_bins,
        shape_power_softness=cfg.cpro_shape_power_softness,
        shape_contrast_softness=cfg.cpro_shape_contrast_softness,
        continuity_decay=cfg.cpro_continuity_decay,
        continuity_power=cfg.cpro_continuity_power,
        min_continuity_mean=cfg.cpro_min_continuity_mean,
        min_ridge_lock=cfg.cpro_min_ridge_lock,
    )
    noise_std = difference_noise_std(raw_series)
    result = cpro_activity(
        power[:, :, target],
        noise_std=noise_std,
        noise_gain=noise_gain,
        target_period_mask=valid,
        params=params,
    )
    cprf_params = _cprf_parameters(cfg)
    normalized_cwt, threshold = normalize_cwt_power(
        power[valid, :, target],
        noise_std=noise_std,
        noise_gain=noise_gain[valid],
        params=cprf_params,
    )
    return periods[valid], result, normalized_cwt, threshold, cprf_params


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
        for rows, color, label, alpha, keys in (
            (windows, CANDIDATE_COLOR, "PELT window", 0.18, ("t0_rec", "t1_rec")),
            (truths, TRUTH_COLOR, "truth span", 0.12, ("record_start", "record_stop")),
        ):
            spans = [(number(row, keys[0]), number(row, keys[1])) for row in rows]
            for start, stop in spans:
                if math.isfinite(start) and math.isfinite(stop) and stop > start:
                    ax.axvspan(start, stop, color=color, alpha=alpha)
            if spans:
                ax.plot([], [], color=color, linewidth=4, alpha=0.6, label=label)
        ax.set(title=title, xlabel="Record", ylabel="CPRO continuous shape evidence")

    _simple_plot(path, dpi, draw)


def _profile_plot(path, normalized_cwt, periods, threshold, params, rows, truths, offset, title, limit, dpi):
    def draw(ax):
        for row in _top(rows, min(5, limit)):
            start = max(0, int(number(row, "t0_rec", offset) - offset))
            stop = min(normalized_cwt.shape[1], int(number(row, "t1_rec", offset) - offset))
            if stop > start:
                result = evaluate_cprf(
                    normalized_cwt[:, start:stop],
                    periods,
                    normalization_threshold=threshold,
                    params=params,
                )
                ax.plot(periods, result.profile, label=f"cand {row.get('candidate_id')}")
        for row in truths:
            if math.isfinite(period := number(row, "period_records")):
                ax.axvline(period, color=TRUTH_COLOR, linestyle="--")
        ax.set(title=title, xlabel="Period / records", ylabel="CPRF period profile", xscale="log")

    _simple_plot(path, dpi, draw)


def _review_plot(path, rows, dpi):
    def draw(ax):
        scores = np.array([max(0, number(row, "score", 0)) for row in rows])
        sizes = 20 + 180 * np.sqrt(scores / scores.max()) if scores.size and scores.max() else np.full(len(rows), 20)
        colors = [VETO_COLOR if row.get("candidate_status") == "vetoed" else CANDIDATE_COLOR for row in rows]
        ax.scatter(
            [number(row, "freq_mhz") for row in rows],
            [number(row, "period_rec") for row in rows],
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
    noise_gain = impulse_cwt_noise_gain(
        periods,
        wavelet=search_config.wavelet,
        method=search_config.cwt_method,
    )
    index = ImageIndex(output, f"Stage Visualization: {run_id or source_name or output.parent.name}")

    path = output / "stage_01_input_matrix.png"
    raw_view(path, matrix, freqs, offset=record_offset, title=f"Stage 01 input matrix: {source_name or run_id}", truths=truths, dpi=cfg.dpi)
    _add(index, "Stage 01 Input Matrix", path, "Raw time-channel matrix.")

    block_limit = math.inf if cfg.max_blocks <= 0 else cfg.max_blocks
    for block_no, start in enumerate(range(0, matrix.shape[1], search_config.block_channels), 1):
        if block_no > block_limit:
            break
        stop, block_id = min(start + search_config.block_channels, matrix.shape[1]), f"block_{block_no:04d}"
        halo = slice(start, stop)
        block, block_freqs = matrix[:, halo], freqs[halo]
        target_freqs = freqs[start:stop]
        power = cwt_power_cube(
            block, periods, wavelet=search_config.wavelet, method=search_config.cwt_method,
            backend=search_config.cwt_backend, cuda_device=search_config.cuda_device, normalize_channels=False,
        )
        target_start, target_stop = start - halo.start, stop - halo.start
        response = aggregate_cwt_time(
            power[:, :, target_start:target_stop], search_config.time_aggregation, search_config.aggregation_percentile
        )
        block_rows = [row for row in raw_candidates if row.get("block_id") == block_id]

        for channel in _channels(start, stop, target_freqs, block_rows, cfg.max_channels):
            local = channel - start
            power_channel = target_start + local
            rows = [row for row in block_rows if int(number(row, "channel", -1)) == channel]
            channel_windows = [
                row for row in windows
                if row.get("block_id") == block_id and int(number(row, "channel", -1)) == channel
            ]
            channel_truths = [
                row for row in truths
                if number(row, "channel_start", -1) <= channel < number(row, "channel_stop", -1)
            ]
            prefix = f"{block_id}_channel_{channel:04d}"
            path = output / f"stage_02_{prefix}_scalogram.png"
            cwt_view(path, power[:, :, power_channel], periods, offset=record_offset, title=f"Stage 02 CWT scalogram: {block_id}, channel {channel}", candidates=rows, truths=channel_truths, dpi=cfg.dpi)
            _add(index, f"Stage 02 CWT Scalogram {block_id} Ch {channel}", path, "Full period-time CWT power.")

            valid_periods, result, normalized_cwt, cprf_threshold, cprf_params = _cpro_products(
                power,
                block[:, power_channel],
                periods,
                power_channel,
                noise_gain,
                search_config,
            )
            path = output / f"stage_03_{prefix}_cpro_score_map.png"
            cwt_view(
                path,
                result.shape_map[
                    cpro_period_mask(
                        periods,
                        search_config.candidate_period_min_records,
                        search_config.candidate_period_max_records,
                    )
                ],
                valid_periods,
                offset=record_offset,
                title=f"Stage 03 CPRO score map: {block_id}, channel {channel}",
                candidates=rows,
                truths=channel_truths,
                cmap="viridis",
                colorbar="Continuous ridge-shape response",
                log_power=True,
                dpi=cfg.dpi,
            )
            _add(index, f"Stage 03 CPRO Shape Map {block_id} Ch {channel}", path, "Continuous CPRO proposal response.")
            path = output / f"stage_04_{prefix}_activity_windows.png"
            _activity_plot(
                path,
                result.shape_activity,
                record_offset,
                channel_windows,
                channel_truths,
                f"Stage 04 CPRO activity + PELT windows: {block_id}, channel {channel}",
                cfg.dpi,
            )
            _add(index, f"Stage 04 Activity Windows {block_id} Ch {channel}", path, "CPRO activity and accepted PELT windows.")
            if rows:
                path = output / f"stage_05_{prefix}_period_profiles.png"
                _profile_plot(
                    path,
                    normalized_cwt,
                    valid_periods,
                    cprf_threshold,
                    cprf_params,
                    rows,
                    channel_truths,
                    record_offset,
                    f"Stage 05 CPRF period profiles: {block_id}, channel {channel}",
                    cfg.top_candidates,
                    cfg.dpi,
                )
                _add(index, f"Stage 05 CPRF Period Profiles {block_id} Ch {channel}", path, "PELT-windowed CPRF period profiles.")

        shaded = []
        if search_config.candidate_period_min_records is not None:
            shaded.append((periods.min(), search_config.candidate_period_min_records))
        if search_config.candidate_period_max_records is not None:
            shaded.append((search_config.candidate_period_max_records, periods.max()))
        truth_boxes = row_boxes(
            truths,
            ("freq_start_mhz", "freq_stop_mhz"),
            ("period_records",),
            color=TRUTH_COLOR,
            label="truth",
            min_span=(1e-6, 1e-6),
        )
        response_values = cwt_power_display_values(response)
        for stage, values, cmap, boxes, note in (
            (6, response_values, CWT_POWER_CMAP, truth_boxes, "Time-aggregated CWT overview."),
            (
                7,
                np.where(
                    ((periods >= (search_config.candidate_period_min_records or -math.inf)) &
                     (periods <= (search_config.candidate_period_max_records or math.inf)))[:, None],
                    response_values, np.nan,
                ),
                CWT_POWER_CMAP,
                row_boxes(
                    _top(block_rows, cfg.top_candidates),
                    ("freq_mhz",),
                    ("p0_rec", "p1_rec"),
                    color=CANDIDATE_COLOR,
                    label="candidate",
                    min_span=(1e-6, 1e-6),
                ) + truth_boxes,
                "Candidate-period CWT overview.",
            ),
        ):
            path = output / f"stage_{stage:02d}_{block_id}_period_channel_{'response' if stage == 6 else 'candidates'}.png"
            heatmap(
                path, values, edges(target_freqs), edges(periods, True),
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
