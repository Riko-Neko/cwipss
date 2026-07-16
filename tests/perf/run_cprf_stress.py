#!/usr/bin/env python3
"""Paired CPRF stress test on historical negative windows and PELT positives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.stats import beta, binomtest


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
PERF_DIR = Path(__file__).resolve().parent
for search_path in (SRC_DIR, PERF_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from period_profile_algorithms import PeriodProfileAlgorithm, period_profile_catalog  # noqa: E402
from period_profile_benchmark import (  # noqa: E402
    CASE_FIELDNAMES,
    MODEL_SUMMARY_FIELDNAMES,
    STAGE1_FIELDNAMES,
    SUMMARY_FIELDNAMES,
    _WindowCase,
    _evaluate_case,
    _model_summaries,
    _normalized_cwt_power,
    _prepare_positive_case,
    _profile_normalization_threshold,
    _resolve_activity_algorithm,
    _summaries,
    _write_csv,
)
from stage_boundaries import pelt_parameters_from_config  # noqa: E402
from persistent_occupancy import mask_windows, regularize_time_mask  # noqa: E402
from cwipss.analysis.injection_config import load_injection_config, make_injections_from_config  # noqa: E402
from cwipss.config import load_cwt_config  # noqa: E402
from cwipss.data.readers import open_spectrum_reader  # noqa: E402
from cwipss.signal.cpro import (  # noqa: E402
    CPROParameters,
    cpro_activity,
    cpro_period_mask,
    difference_noise_std,
    impulse_cwt_noise_gain,
)
from cwipss.signal.cwt import cwt_power_cube, period_grid_records  # noqa: E402


DEFAULT_INPUTS = (
    PROJECT_DIR / "data/CE4/CE4_GRAS_LFRS-TR_SCI_P_20211205160000_20211206040000_0211_B.2C",
    PROJECT_DIR / "data/CE4/CE4_GRAS_LFRS-TR_SCI_P_20190830160000_20190831040000_0056_B.2C",
)
DEFAULT_ALGORITHMS = (
    "cprf_concentrated_ridge_c45",
    "cprf_absolute_ridge_c35_r140",
    "pbsf_focus_concentration_c30",
    "pbsf_persistence_sqrt_gate_e150_b4",
)
NEW_CPRF = "cprf_concentrated_ridge_c45"
OLD_CPRF = "cprf_absolute_ridge_c35_r140"
TRIAL_CPRF = "cprf_contrast_balanced_c40_r150_i300"
SWEEP_DIRNAME = "threshold_sweep_integrated0_fine_v1"
HISTORICAL_NEGATIVE_COUNTS = {
    "20211205": 1051,
    "20190830": 3253,
}


@dataclass(frozen=True)
class StressConfig:
    output_dir: Path
    inputs: tuple[Path, ...] = DEFAULT_INPUTS
    injection_config: Path = PERF_DIR / "configs/cprf_stress_positive_216.json"
    cwt_config: Path = PROJECT_DIR / "configs/cwt_default.json"
    algorithms: tuple[str, ...] = DEFAULT_ALGORITHMS
    positive_activity_algorithm: str = "sm_cpro_w769"
    negative_channel_start: int = 0
    negative_channel_stop: int = 2048
    negative_block_channels: int = 16
    max_positive_per_observation: int = 0
    positive_period_max_records: float = 1000.0
    historical_negative_period_max_records: float = 200.0
    stage3_min_window_records: int = 96
    profile_threshold_snr: float = 32.0
    profile_texture_quantile: float = 0.9375
    progress_every: int = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_algorithms(names: tuple[str, ...]) -> tuple[PeriodProfileAlgorithm, ...]:
    catalog = {item.name: item for item in period_profile_catalog()}
    missing = sorted(set(names) - set(catalog))
    if missing:
        raise ValueError(f"Unknown period-profile algorithms: {', '.join(missing)}")
    return tuple(catalog[name] for name in names)


def _observation_id(path: Path) -> str:
    return "20211205" if "20211205" in path.name else "20190830" if "20190830" in path.name else path.stem


def _historical_negative_cases(
    *,
    reader: Any,
    observation: str,
    periods: np.ndarray,
    noise_gain: np.ndarray,
    historical_period_mask: np.ndarray,
    wavelet: str,
    method: str,
    channel_start: int,
    channel_stop: int,
    block_channels: int,
    progress_every: int,
) -> list[_WindowCase]:
    """Rebuild the historical w385_v40_d096 negative-window corpus."""
    parameters = CPROParameters(window_support_records=385, min_window_occupancy=0.40)
    cases: list[_WindowCase] = []
    first = max(0, int(channel_start))
    last = min(int(reader.n_channels), int(channel_stop))
    channels = list(range(first, last))
    width = max(1, int(block_channels))
    for block_index, offset in enumerate(range(0, len(channels), width), start=1):
        selected = channels[offset : offset + width]
        block = np.asarray(
            reader.read_block(
                slice(0, reader.n_records),
                slice(selected[0], selected[-1] + 1),
            ).data,
            dtype=np.float32,
        )
        power_cube = cwt_power_cube(
            block,
            periods,
            wavelet=wavelet,
            normalize_channels=False,
            method=method,
            backend="cpu",
        )
        for local_index, channel in enumerate(selected):
            series = block[:, local_index]
            power = np.asarray(power_cube[:, :, local_index], dtype=np.float32)
            noise_std = difference_noise_std(series)
            activity_result = cpro_activity(
                power[historical_period_mask],
                noise_std=noise_std,
                noise_gain=noise_gain[historical_period_mask],
                params=parameters,
            )
            legacy_mask = regularize_time_mask(
                activity_result.active_mask,
                max_gap=64,
                min_duration=96,
            )
            legacy_activity = np.where(legacy_mask, activity_result.activity, 0.0)
            legacy_windows = mask_windows(legacy_mask, legacy_activity)
            threshold = _profile_normalization_threshold(
                power,
                noise_std=noise_std,
                noise_gain=noise_gain,
                threshold_snr=parameters.threshold_snr,
                texture_quantile=parameters.texture_quantile,
            )
            normalized = _normalized_cwt_power(
                power,
                noise_std=noise_std,
                noise_gain=noise_gain,
                calibrated_threshold=threshold,
            )
            for window_index, window in enumerate(legacy_windows, start=1):
                start = int(window["record_start"])
                stop = int(window["record_stop"])
                cases.append(
                    _WindowCase(
                        case_id=f"{observation}_legacy_negative_ch{channel:04d}_w{window_index:04d}",
                        case_kind="negative",
                        signal_model="historical_window_index_raw_cwt_profile",
                        channel_index=channel,
                        frequency_mhz=float(reader.freqs_mhz[channel]),
                        record_start=start,
                        record_stop=stop,
                        truth_period_records=math.nan,
                        periods=periods,
                        normalized_score_map=np.asarray(normalized[:, start:stop], dtype=np.float32).copy(),
                    )
                )
        del power_cube
        completed = min(offset + width, len(channels))
        if progress_every > 0 and (block_index % progress_every == 0 or completed == len(channels)):
            print(
                f"[cprf-stress] {observation} historical negatives "
                f"{completed}/{len(channels)} channels: {len(cases)} windows",
                flush=True,
            )
    return cases


def _positive_cases(
    *,
    reader: Any,
    observation: str,
    specs: list[Any],
    periods: np.ndarray,
    noise_gain: np.ndarray,
    wavelet: str,
    method: str,
    activity_algorithm: Any,
    pelt_parameters: Any,
    config: StressConfig,
) -> tuple[list[_WindowCase], list[dict[str, Any]]]:
    cases: list[_WindowCase] = []
    stage1_rows: list[dict[str, Any]] = []
    for index, original in enumerate(specs, start=1):
        spec = replace(original, injection_id=f"{observation}_{original.injection_id}")
        try:
            case, row = _prepare_positive_case(
                reader=reader,
                spec=spec,
                periods=periods,
                noise_gain=noise_gain,
                wavelet=wavelet,
                method=method,
                backend="cpu",
                cuda_device=0,
                activity_algorithm=activity_algorithm,
                pelt_parameters=pelt_parameters,
                stage3_min_window_records=config.stage3_min_window_records,
                profile_threshold_snr=config.profile_threshold_snr,
                profile_texture_quantile=config.profile_texture_quantile,
            )
        except Exception as exc:
            case = None
            row = {
                "case_id": spec.injection_id,
                "signal_model": spec.signal_model,
                "channel_index": int(round(float(spec.channel_center))),
                "frequency_mhz": "",
                "truth_period_records": float(spec.period_records),
                "truth_record_start": int(spec.record_start),
                "truth_record_stop": int(spec.record_start + int(spec.duration_records or 0)),
                "stage1_status": "error",
                "pelt_window_count": 0,
                "stage3_window_count": 0,
                "stage1_window_count": 0,
                "matched_record_start": "",
                "matched_record_stop": "",
                "matched_overlap_fraction": 0.0,
                "preprocess_seconds": 0.0,
                "error": str(exc),
            }
        row["observation"] = observation
        stage1_rows.append(row)
        if case is not None:
            cases.append(case)
        if config.progress_every > 0 and (index % config.progress_every == 0 or index == len(specs)):
            print(
                f"[cprf-stress] {observation} positives {index}/{len(specs)}: "
                f"{len(cases)} PELT windows",
                flush=True,
            )
    return cases, stage1_rows


def _evaluate(cases: list[_WindowCase], algorithms: tuple[PeriodProfileAlgorithm, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        offset = case_index % len(algorithms)
        order = algorithms[offset:] + algorithms[:offset]
        rows.extend(_evaluate_case(case, algorithm) for algorithm in order)
        if (case_index + 1) % 500 == 0 or case_index + 1 == len(cases):
            print(f"[cprf-stress] evaluated {case_index + 1}/{len(cases)} windows", flush=True)
    return rows


def _proportion_interval(successes: int, total: int, alpha: float = 0.05) -> list[float]:
    if total <= 0:
        return [math.nan, math.nan]
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, total - successes + 1))
    high = 1.0 if successes == total else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes))
    return [low, high]


def _paired_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_algorithm = {
        name: {str(row["case_id"]): row for row in rows if row["algorithm"] == name}
        for name in (NEW_CPRF, OLD_CPRF)
    }
    shared = sorted(set(by_algorithm[NEW_CPRF]) & set(by_algorithm[OLD_CPRF]))
    payload: dict[str, Any] = {}
    for kind in ("positive", "negative"):
        ids = [case_id for case_id in shared if by_algorithm[NEW_CPRF][case_id]["case_kind"] == kind]
        old_only = sum(
            int(by_algorithm[OLD_CPRF][case_id]["accepted"])
            and not int(by_algorithm[NEW_CPRF][case_id]["accepted"])
            for case_id in ids
        )
        new_only = sum(
            int(by_algorithm[NEW_CPRF][case_id]["accepted"])
            and not int(by_algorithm[OLD_CPRF][case_id]["accepted"])
            for case_id in ids
        )
        both = sum(
            int(by_algorithm[NEW_CPRF][case_id]["accepted"])
            and int(by_algorithm[OLD_CPRF][case_id]["accepted"])
            for case_id in ids
        )
        neither = len(ids) - old_only - new_only - both
        discordant = old_only + new_only
        payload[kind] = {
            "case_count": len(ids),
            "both_accept": both,
            "old_only_accept": old_only,
            "new_only_accept": new_only,
            "neither_accept": neither,
            "exact_p_new_accepts_less": (
                float(binomtest(new_only, discordant, 0.5, alternative="less").pvalue)
                if discordant
                else 1.0
            ),
        }
    return payload


def _summary_with_intervals(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in summary_rows:
        positive_total = int(row["positive_case_count"])
        negative_total = int(row["negative_case_count"])
        positive_successes = int(round(float(row["positive_accept_rate"]) * positive_total))
        negative_successes = int(round(float(row["negative_false_accept_rate"]) * negative_total))
        enriched.append(
            {
                **row,
                "positive_accept_count": positive_successes,
                "positive_accept_ci95": _proportion_interval(positive_successes, positive_total),
                "negative_false_accept_count": negative_successes,
                "negative_false_accept_ci95": _proportion_interval(negative_successes, negative_total),
            }
        )
    return enriched


def _plot_summary(summary: list[dict[str, Any]], output: Path) -> Path:
    labels = [str(row["algorithm"]).replace("cprf_", "") for row in summary]
    positive = [100.0 * float(row["positive_accept_rate"]) for row in summary]
    negative = [int(row["negative_false_accept_count"]) for row in summary]
    colors = ["#0b6e69" if row["algorithm"] == NEW_CPRF else "#c84b31" if row["algorithm"] == OLD_CPRF else "#d5a021" for row in summary]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    axes[0].barh(labels[::-1], positive[::-1], color=colors[::-1])
    axes[0].set_xlabel("Positive PELT-window acceptance (%)")
    axes[0].set_xlim(0, 100)
    axes[0].grid(axis="x", alpha=0.18)
    axes[1].barh(labels[::-1], negative[::-1], color=colors[::-1])
    axes[1].set_xlabel("Accepted historical negative windows (count)")
    axes[1].grid(axis="x", alpha=0.18)
    fig.suptitle("CPRF stress trade-off on shared windows")
    path = output / "stress_acceptance_tradeoff.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_paired(paired: dict[str, Any], output: Path) -> Path:
    categories = ["Both", "Old only", "New only", "Neither"]
    keys = ["both_accept", "old_only_accept", "new_only_accept", "neither_accept"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, kind, color in zip(axes, ("positive", "negative"), ("#0b6e69", "#c84b31"), strict=True):
        values = [int(paired[kind][key]) for key in keys]
        ax.bar(categories, values, color=[color, "#c84b31", "#0b6e69", "#8d99ae"])
        ax.set_title(f"{kind.capitalize()} paired decisions")
        ax.set_ylabel("Window count")
        ax.grid(axis="y", alpha=0.18)
    path = output / "paired_cprf_decisions.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_period_errors(rows: list[dict[str, Any]], output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for algorithm, color in ((NEW_CPRF, "#0b6e69"), (OLD_CPRF, "#c84b31")):
        errors = sorted(
            float(row["period_error_fraction"])
            for row in rows
            if row["algorithm"] == algorithm
            and row["case_kind"] == "positive"
            and int(row["accepted"])
        )
        if not errors:
            continue
        y = np.arange(1, len(errors) + 1, dtype=np.float64) / len(errors)
        ax.step(100.0 * np.asarray(errors), y, where="post", label=algorithm, color=color)
    ax.axvline(10.0, color="#2d3047", linestyle="--", linewidth=1.0, label="10% error")
    ax.set_xlabel("Accepted period error (%)")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.set_xscale("log")
    ax.grid(alpha=0.18, which="both")
    ax.legend(fontsize=8)
    path = output / "accepted_period_error_ecdf.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_negative_score_plane(rows: list[dict[str, Any]], output: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, algorithm, title in zip(
        axes,
        (OLD_CPRF, NEW_CPRF),
        ("Old CPRF", "New CPRF"),
        strict=True,
    ):
        selected = [
            row for row in rows
            if row["algorithm"] == algorithm and row["case_kind"] == "negative"
        ]
        accepted = np.asarray([int(row["accepted"]) for row in selected], dtype=bool)
        concentration = np.asarray([float(row["band_concentration"]) for row in selected])
        contrast = np.asarray([float(row["local_contrast"]) for row in selected])
        ax.hexbin(concentration[~accepted], contrast[~accepted], gridsize=45, mincnt=1, cmap="Blues", bins="log")
        if np.any(accepted):
            ax.scatter(concentration[accepted], contrast[accepted], color="#d1495b", s=18, alpha=0.75, label="Accepted")
        ax.set_title(f"{title}: accepted {int(np.count_nonzero(accepted))}/{len(selected)}")
        ax.set_xlabel("Band concentration")
        ax.set_ylabel("Local contrast")
        ax.grid(alpha=0.12)
        if np.any(accepted):
            ax.legend()
    path = output / "historical_negative_score_plane.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_new_cprf_plane(
    rows: list[dict[str, Any]],
    output: Path,
    algorithm: PeriodProfileAlgorithm,
    *,
    y_key: str,
    y_label: str,
    filename: str,
    gate_y: float = 0.0,
    log_y: bool = False,
) -> Path:
    selected = [row for row in rows if row["algorithm"] == algorithm.name]
    fig, ax = plt.subplots(figsize=(15, 10), constrained_layout=True)
    styles = (
        ("negative", 0, "#d85d3f", "x", 18, 0.34, 0.9, "Historical negative windows: rejected"),
        ("positive", 0, "#168f83", "x", 44, 0.90, 1.4, "Injected PELT windows: rejected"),
        ("negative", 1, "#d85d3f", "o", 20, 0.70, 0.8, "Historical negative windows: accepted"),
        ("positive", 1, "#168f83", "o", 58, 0.82, 0.8, "Injected PELT windows: accepted"),
    )
    for kind, accepted, color, marker, size, alpha, linewidth, label in styles:
        subset = [
            row for row in selected
            if row["case_kind"] == kind and int(row["accepted"]) == accepted
        ]
        ax.scatter(
            [float(row["band_concentration"]) for row in subset],
            [float(row[y_key]) for row in subset],
            color=color,
            marker=marker,
            s=size,
            alpha=alpha,
            linewidth=linewidth,
            edgecolor="white" if marker == "o" else None,
            label=f"{label} ({len(subset)})",
            rasterized=True,
        )
    ax.axvline(
        algorithm.min_band_concentration,
        color="#25324a",
        linestyle="--",
        linewidth=1.4,
        label=f"Concentration >= {algorithm.min_band_concentration:.2f}",
    )
    if gate_y > 0.0:
        ax.axhline(
            gate_y,
            color="#6b4f3b",
            linestyle="--",
            linewidth=1.4,
            label=f"{y_label} >= {gate_y:g}",
        )
    if log_y:
        ax.set_yscale("log")
    if y_key == "local_contrast":
        clipped = sum(float(row[y_key]) > 20.0 for row in selected)
        ax.set_ylim(0.0, 20.0)
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        if clipped:
            ax.text(
                0.99,
                0.985,
                f"{clipped} window above y=20 clipped",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color="#5f4b3b",
                fontsize=9,
            )
    ax.set_xlabel("Main-band energy concentration")
    ax.set_ylabel(y_label)
    ax.set_title(
        "New CPRF stress score plane: 420 injected PELT windows + "
        "4,304 historical negative windows"
    )
    ax.grid(alpha=0.18, which="major")
    ax.grid(alpha=0.08, which="minor")
    ax.legend(fontsize=9, ncol=2)
    path = output / filename
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_new_cprf_score_planes(
    rows: list[dict[str, Any]],
    output: Path,
    algorithm: PeriodProfileAlgorithm,
) -> list[Path]:
    return [
        _plot_new_cprf_plane(
            rows,
            output,
            algorithm,
            y_key="local_contrast",
            y_label="Main-band local sideband contrast (diagnostic; no hard gate)",
            filename="new_cprf_score_plane.png",
            gate_y=algorithm.min_local_contrast,
        ),
        _plot_new_cprf_plane(
            rows,
            output,
            algorithm,
            y_key="integrated_strength",
            y_label="Integrated band strength",
            filename="new_cprf_gate_plane.png",
            gate_y=algorithm.min_integrated_strength,
            log_y=True,
        ),
    ]


def _rethreshold_row(
    row: dict[str, Any],
    algorithm: PeriodProfileAlgorithm,
) -> dict[str, Any]:
    accepted = int(
        int(float(row["width_bins"])) >= algorithm.min_width_bins
        and float(row["peak_strength"]) >= algorithm.min_peak_strength
        and float(row["integrated_strength"]) >= algorithm.min_integrated_strength
        and float(row["band_persistence"]) >= algorithm.min_band_persistence
        and float(row["band_concentration"]) >= algorithm.min_band_concentration
        and float(row["local_contrast"]) >= algorithm.min_local_contrast
        and float(row["total_score"]) >= algorithm.min_total_score
    )
    replay = dict(row)
    replay["algorithm"] = algorithm.name
    replay["accepted"] = accepted
    if row["case_kind"] == "positive":
        error = float(row["period_error_fraction"])
        truth = float(row["truth_period_records"])
        band_lo, band_hi = sorted(
            (float(row["period_start_records"]), float(row["period_stop_records"]))
        )
        replay["period_hit_05"] = int(accepted and error <= 0.05)
        replay["period_hit_10"] = int(accepted and error <= 0.10)
        replay["period_hit_20"] = int(accepted and error <= 0.20)
        replay["truth_inside_peak_band"] = int(accepted and band_lo <= truth <= band_hi)
        aliases = (truth / 2.0, truth / 3.0, truth * 2.0, truth * 3.0)
        estimate = float(row["peak_period_records"])
        replay["harmonic_confusion"] = int(
            accepted
            and error > 0.20
            and any(abs(estimate - alias) / max(abs(alias), 1e-12) <= 0.10 for alias in aliases)
        )
    return replay


def rethreshold_existing_stress(run_dir: Path) -> Path:
    """Replay gate-only CPRF changes without recomputing CWT or peak metrics."""
    with (run_dir / "cprf_stress_cases.csv").open(newline="") as stream:
        source_rows = [
            row for row in csv.DictReader(stream)
            if row["algorithm"] == NEW_CPRF
        ]
    base = _resolve_algorithms((NEW_CPRF,))[0]
    trial = replace(
        base,
        name=TRIAL_CPRF,
        min_band_concentration=0.40,
        min_local_contrast=1.50,
        min_integrated_strength=3.00,
    )
    rows = [_rethreshold_row(row, trial) for row in source_rows]
    output = run_dir / "threshold_trial_c40_r150_i300"
    output.mkdir(parents=True, exist_ok=True)
    summary = _summary_with_intervals(_summaries(rows, (trial,)))[0]
    _write_csv(output / "cases.csv", CASE_FIELDNAMES, rows)
    _write_csv(output / "summary.csv", SUMMARY_FIELDNAMES, [summary])
    score_plane = _plot_new_cprf_plane(
        rows,
        output,
        trial,
        y_key="local_contrast",
        y_label="Main-band local sideband contrast",
        filename="trial_score_plane.png",
        gate_y=trial.min_local_contrast,
    )
    gate_plane = _plot_new_cprf_plane(
        rows,
        output,
        trial,
        y_key="integrated_strength",
        y_label="Integrated band strength",
        filename="trial_gate_plane.png",
        gate_y=trial.min_integrated_strength,
        log_y=True,
    )
    baseline = {str(row["case_id"]): int(row["accepted"]) for row in source_rows}
    paired: dict[str, dict[str, int]] = {}
    for kind in ("positive", "negative"):
        subset = [row for row in rows if row["case_kind"] == kind]
        both = sum(baseline[str(row["case_id"])] and int(row["accepted"]) for row in subset)
        old_only = sum(baseline[str(row["case_id"])] and not int(row["accepted"]) for row in subset)
        new_only = sum(not baseline[str(row["case_id"])] and int(row["accepted"]) for row in subset)
        paired[kind] = {
            "both_accept": both,
            "current_only_accept": old_only,
            "trial_only_accept": new_only,
            "neither_accept": len(subset) - both - old_only - new_only,
            "exact_p_trial_accepts_more": (
                float(binomtest(new_only, old_only + new_only, 0.5, alternative="greater").pvalue)
                if old_only + new_only
                else 1.0
            ),
            "exact_p_trial_accepts_less": (
                float(binomtest(new_only, old_only + new_only, 0.5, alternative="less").pvalue)
                if old_only + new_only
                else 1.0
            ),
        }
    payload = {
        "schema_version": 1,
        "method": "gate_only_replay_on_fixed_cprf_peak_metrics",
        "source_run": str(run_dir),
        "source_algorithm": NEW_CPRF,
        "trial_algorithm": trial.to_dict(),
        "summary": summary,
        "paired_current_vs_trial": paired,
        "images": [score_plane.name, gate_plane.name],
    }
    path = output / "summary.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    positive_count = int(summary["positive_accept_count"])
    negative_count = int(summary["negative_false_accept_count"])
    (output / "RESULT.md").write_text(
        "\n".join(
            (
                "# CPRF Threshold Trial c40/r150/i300",
                "",
                f"- Conditional positive acceptance: `{positive_count}/420` "
                f"(`{100 * float(summary['positive_accept_rate']):.2f}%`)",
                f"- End-to-end positive acceptance: `{positive_count}/432` "
                f"(`{100 * positive_count / 432:.2f}%`)",
                f"- Accepted 10% period accuracy: "
                f"`{100 * float(summary['accepted_period_hit_10_rate']):.2f}%`",
                f"- Historical negative retention: `{negative_count}/4304` "
                f"(`{100 * float(summary['negative_false_accept_rate']):.2f}%`)",
                "- Paired positives: 341 both, 0 current-only, 12 trial-only, 67 neither.",
                "- Paired negatives: 122 both, 102 current-only, 90 trial-only, 3990 neither.",
                "- Positive gain exact p-value: `0.000244` (one-sided).",
                "- Negative reduction exact p-value: `0.214` (one-sided; not significant).",
                "",
                "`min_integrated_strength=3.0` is retained as a protection gate, but it has "
                "no independent marginal decision effect on this fixed corpus at c40/r150.",
                "",
                "![trial score plane](trial_score_plane.png)",
                "![trial gate plane](trial_gate_plane.png)",
                "",
            )
        )
    )
    return path


def sweep_existing_stress(run_dir: Path) -> Path:
    """Find the exact two-gate Pareto frontier on fixed CPRF peak metrics."""
    with (run_dir / "cprf_stress_cases.csv").open(newline="") as stream:
        source_rows = [
            row for row in csv.DictReader(stream)
            if row["algorithm"] == NEW_CPRF
        ]

    kind = np.asarray([row["case_kind"] for row in source_rows])
    concentration = np.asarray(
        [float(row["band_concentration"]) for row in source_rows]
    )
    contrast = np.asarray([float(row["local_contrast"]) for row in source_rows])
    base_eligible = np.asarray(
        [
            int(float(row["width_bins"])) >= 3
            and float(row["peak_strength"]) >= 1.25
            and float(row["band_persistence"]) >= 0.35
            for row in source_rows
        ],
        dtype=bool,
    )
    positive = kind == "positive"
    negative = kind == "negative"
    positive_attempt_count = 432
    negative_count = int(np.count_nonzero(negative))

    # Threshold decisions only change at observed metric values. Enumerating
    # these breakpoints is finer and more complete than an arbitrary grid.
    best_by_positive: dict[int, tuple[int, float, float]] = {}
    for concentration_gate in np.unique(concentration[base_eligible]):
        eligible = base_eligible & (concentration >= concentration_gate)
        indices = np.flatnonzero(eligible)
        order = indices[np.argsort(-contrast[indices], kind="stable")]
        ordered_contrast = contrast[order]
        positive_cumulative = np.cumsum(positive[order])
        negative_cumulative = np.cumsum(negative[order])
        group_ends = np.flatnonzero(
            np.r_[ordered_contrast[:-1] != ordered_contrast[1:], True]
        )
        for end in group_ends:
            positive_accept_count = int(positive_cumulative[end])
            candidate = (
                int(negative_cumulative[end]),
                float(concentration_gate),
                float(ordered_contrast[end]),
            )
            current = best_by_positive.get(positive_accept_count)
            if current is None or candidate < current:
                best_by_positive[positive_accept_count] = candidate

    best_rows: list[dict[str, Any]] = []
    for positive_accept_count, candidate in sorted(best_by_positive.items()):
        negative_accept_count, concentration_gate, contrast_gate = candidate
        best_rows.append(
            {
                "positive_accept_count": positive_accept_count,
                "positive_attempt_count": positive_attempt_count,
                "end_to_end_recall": positive_accept_count / positive_attempt_count,
                "negative_accept_count": negative_accept_count,
                "negative_count": negative_count,
                "negative_retention_rate": negative_accept_count / negative_count,
                "min_band_concentration": concentration_gate,
                "min_local_contrast": contrast_gate,
                "min_integrated_strength": 0.0,
            }
        )

    pareto_rows: list[dict[str, Any]] = []
    lowest_negative_count = math.inf
    for row in reversed(best_rows):
        negative_accept_count = int(row["negative_accept_count"])
        if negative_accept_count < lowest_negative_count:
            pareto_rows.append(row)
            lowest_negative_count = negative_accept_count
    pareto_rows.reverse()

    output = run_dir / SWEEP_DIRNAME
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(best_rows[0])
    _write_csv(output / "best_by_positive_count.csv", fieldnames, best_rows)
    _write_csv(output / "pareto_frontier.csv", fieldnames, pareto_rows)

    slice_rows: list[dict[str, Any]] = []
    for concentration_gate in (0.50, 0.55):
        for contrast_gate in np.linspace(0.0, 20.0, 2001):
            accepted = (
                base_eligible
                & (concentration >= concentration_gate)
                & (contrast >= contrast_gate)
            )
            positive_accept_count = int(np.count_nonzero(accepted & positive))
            negative_accept_count = int(np.count_nonzero(accepted & negative))
            slice_rows.append(
                {
                    "min_band_concentration": concentration_gate,
                    "min_local_contrast": float(contrast_gate),
                    "min_integrated_strength": 0.0,
                    "positive_accept_count": positive_accept_count,
                    "positive_attempt_count": positive_attempt_count,
                    "end_to_end_recall": positive_accept_count / positive_attempt_count,
                    "negative_accept_count": negative_accept_count,
                    "negative_count": negative_count,
                    "negative_retention_rate": negative_accept_count / negative_count,
                }
            )
    _write_csv(
        output / "contrast_slice_c50_c55_step001.csv",
        tuple(slice_rows[0]),
        slice_rows,
    )

    payload = {
        "schema_version": 1,
        "method": "exact_observed_breakpoint_sweep_on_fixed_cprf_peak_metrics",
        "source_run": str(run_dir),
        "source_algorithm": NEW_CPRF,
        "fixed_parameters": {
            "min_width_bins": 3,
            "min_peak_strength": 1.25,
            "min_band_persistence": 0.35,
            "min_integrated_strength": 0.0,
        },
        "positive_attempt_count": positive_attempt_count,
        "positive_pelt_window_count": int(np.count_nonzero(positive)),
        "negative_window_count": negative_count,
        "attainable_positive_counts": len(best_rows),
        "pareto_point_count": len(pareto_rows),
        "minimum_negative_count_with_nonzero_recall": min(
            int(row["negative_accept_count"]) for row in best_rows
        ),
        "contrast_slice": {
            "concentration_values": [0.50, 0.55],
            "contrast_start": 0.0,
            "contrast_stop": 20.0,
            "contrast_step": 0.01,
            "high_sensitivity_review_range": [0.0, 3.0],
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    return summary_path


def _plot_new_cprf_period_accuracy(rows: list[dict[str, Any]], output: Path) -> Path:
    positives = [
        row for row in rows
        if row["algorithm"] == NEW_CPRF and row["case_kind"] == "positive"
    ]
    accepted = [row for row in positives if int(row["accepted"])]
    rejected = [row for row in positives if not int(row["accepted"])]
    truth = np.asarray([float(row["truth_period_records"]) for row in positives])
    axis = np.geomspace(float(np.min(truth)), float(np.max(truth)), 256)
    fig, ax = plt.subplots(figsize=(11, 10), constrained_layout=True)
    ax.fill_between(
        axis,
        0.9 * axis,
        1.1 * axis,
        color="#e9c46a",
        alpha=0.20,
        label="10% error band",
    )
    ax.plot(axis, axis, color="#25324a", linewidth=1.3)
    ax.scatter(
        [float(row["truth_period_records"]) for row in accepted],
        [float(row["peak_period_records"]) for row in accepted],
        color="#2a9d8f",
        s=38,
        alpha=0.72,
        label=f"Accepted ({len(accepted)})",
        rasterized=True,
    )
    ax.scatter(
        [float(row["truth_period_records"]) for row in rejected],
        [float(row["peak_period_records"]) for row in rejected],
        color="#e07a5f",
        s=50,
        marker="x",
        alpha=0.92,
        linewidth=1.5,
        label=f"Rejected by new CPRF ({len(rejected)})",
        rasterized=True,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Injected period (records)")
    ax.set_ylabel("Estimated main period (records)")
    ax.set_title("New CPRF period estimates on 420 stress-test PELT windows")
    ax.grid(alpha=0.16, which="both")
    ax.legend()
    path = output / "new_cprf_period_accuracy.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_new_cprf_harmonics(rows: list[dict[str, Any]], output: Path) -> Path:
    selected = [row for row in rows if row["algorithm"] == NEW_CPRF]
    fig, ax = plt.subplots(figsize=(11, 10), constrained_layout=True)
    styles = (
        ("negative", 0, "#d85d3f", "x", 17, 0.32, 0.9, "Historical negatives: rejected"),
        ("positive", 0, "#168f83", "x", 42, 0.88, 1.4, "Injected: rejected"),
        ("negative", 1, "#d85d3f", "o", 24, 0.70, 0.8, "Historical negatives: accepted"),
        ("positive", 1, "#168f83", "o", 42, 0.80, 0.8, "Injected: accepted"),
    )
    for kind, accepted, color, marker, size, alpha, linewidth, label in styles:
        subset = [
            row for row in selected
            if row["case_kind"] == kind and int(row["accepted"]) == accepted
        ]
        ax.scatter(
            [float(row["harmonic_2_score"]) for row in subset],
            [float(row["harmonic_3_score"]) for row in subset],
            color=color,
            marker=marker,
            s=size,
            alpha=alpha,
            linewidth=linewidth,
            edgecolor="white" if marker == "o" else None,
            label=f"{label} ({len(subset)})",
            rasterized=True,
        )
    ax.set_xlabel("2f0 auxiliary response / main peak")
    ax.set_ylabel("3f0 auxiliary response / main peak")
    ax.set_title("New CPRF harmonic responses are diagnostic, not a hard gate")
    ax.grid(alpha=0.16)
    ax.legend(fontsize=9, ncol=2)
    path = output / "new_cprf_harmonic_diagnostics.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def run(config: StressConfig) -> Path:
    started = perf_counter()
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    algorithms = _resolve_algorithms(config.algorithms)
    activity_algorithm = _resolve_activity_algorithm(config.positive_activity_algorithm)
    search = load_cwt_config(config.cwt_config, overrides={"cwt_backend": "cpu"})
    full_periods = period_grid_records(
        search.period_min_records,
        search.period_max_records,
        search.period_count,
        search.period_spacing,
    )
    positive_periods = full_periods[
        cpro_period_mask(
            full_periods,
            search.candidate_period_min_records,
            config.positive_period_max_records,
        )
    ]
    historical_period_mask = cpro_period_mask(
        positive_periods,
        search.candidate_period_min_records,
        config.historical_negative_period_max_records,
    )
    positive_noise_gain = impulse_cwt_noise_gain(
        positive_periods,
        wavelet=search.wavelet,
        method=search.cwt_method,
    )
    pelt_parameters = pelt_parameters_from_config(search)
    injection_payload = load_injection_config(config.injection_config)

    cases: list[_WindowCase] = []
    stage1_rows: list[dict[str, Any]] = []
    corpus_counts: dict[str, dict[str, int]] = {}
    for input_path in config.inputs:
        reader = open_spectrum_reader(input_path)
        observation = _observation_id(input_path)
        specs = make_injections_from_config(
            injection_payload,
            records=reader.n_records,
            channels=reader.n_channels,
            freqs_mhz=reader.freqs_mhz,
        )
        if config.max_positive_per_observation > 0:
            specs = specs[: config.max_positive_per_observation]
        positive_cases, positive_rows = _positive_cases(
            reader=reader,
            observation=observation,
            specs=specs,
            periods=positive_periods,
            noise_gain=positive_noise_gain,
            wavelet=search.wavelet,
            method=search.cwt_method,
            activity_algorithm=activity_algorithm,
            pelt_parameters=pelt_parameters,
            config=config,
        )
        negative_cases = _historical_negative_cases(
            reader=reader,
            observation=observation,
            periods=positive_periods,
            noise_gain=positive_noise_gain,
            historical_period_mask=historical_period_mask,
            wavelet=search.wavelet,
            method=search.cwt_method,
            channel_start=config.negative_channel_start,
            channel_stop=config.negative_channel_stop,
            block_channels=config.negative_block_channels,
            progress_every=config.progress_every,
        )
        expected_negative_count = HISTORICAL_NEGATIVE_COUNTS.get(observation)
        if expected_negative_count is not None and len(negative_cases) != expected_negative_count:
            raise RuntimeError(
                f"Historical negative corpus mismatch for {observation}: "
                f"expected {expected_negative_count}, rebuilt {len(negative_cases)}"
            )
        cases.extend(positive_cases)
        cases.extend(negative_cases)
        stage1_rows.extend(positive_rows)
        corpus_counts[observation] = {
            "positive_attempts": len(specs),
            "positive_pelt_windows": len(positive_cases),
            "historical_negative_windows": len(negative_cases),
        }

    case_rows = _evaluate(cases, algorithms)
    summaries = _summary_with_intervals(_summaries(case_rows, algorithms))
    paired = _paired_comparison(case_rows)
    _write_csv(output / "cprf_stress_cases.csv", CASE_FIELDNAMES, case_rows)
    _write_csv(output / "cprf_stress_summary.csv", SUMMARY_FIELDNAMES, summaries)
    _write_csv(output / "cprf_stress_model_summary.csv", MODEL_SUMMARY_FIELDNAMES, _model_summaries(case_rows, algorithms))
    stage1_fields = STAGE1_FIELDNAMES + ["observation"]
    _write_csv(output / "cprf_stress_stage1.csv", stage1_fields, stage1_rows)
    (output / "cprf_stress_algorithm_map.json").write_text(
        json.dumps({algorithm.name: algorithm.to_dict() for algorithm in algorithms}, indent=2)
    )
    images = [
        _plot_summary(summaries, output),
        _plot_paired(paired, output),
        _plot_period_errors(case_rows, output),
        _plot_negative_score_plane(case_rows, output),
    ]
    images.extend(
        _plot_new_cprf_score_planes(
            case_rows,
            output,
            next(algorithm for algorithm in algorithms if algorithm.name == NEW_CPRF),
        )
    )
    images.extend(
        (
            _plot_new_cprf_period_accuracy(case_rows, output),
            _plot_new_cprf_harmonics(case_rows, output),
        )
    )
    positive_attempt_count = sum(int(counts["positive_attempts"]) for counts in corpus_counts.values())
    positive_window_count = sum(case.case_kind == "positive" for case in cases)
    negative_window_count = sum(case.case_kind == "negative" for case in cases)
    payload = {
        "schema_version": 1,
        "benchmark": "paired_cprf_historical_negative_stress",
        "config": {**asdict(config), "output_dir": str(output)},
        "reproducibility": {
            "input_sha256": {str(path): _sha256(path) for path in config.inputs},
            "injection_config_sha256": _sha256(config.injection_config),
            "cwt_config_sha256": _sha256(config.cwt_config),
        },
        "scientific_boundaries": {
            "positive_windows": "current sm_cpro_w769 activity -> native C++ PELT -> 96-record whole-window gate",
            "negative_windows": "historical CPRO w385_v40_d096 window indices reconstructed only for fixed CPRF stress",
            "positive_period_input": "original unmasked absolute CWT sliced by current native-PELT windows and independently calibrated",
            "negative_period_input": "original unmasked absolute CWT sliced by the reconstructed historical window indices and independently calibrated",
            "positive_period_domain_records": [
                float(positive_periods[0]),
                float(positive_periods[-1]),
            ],
            "historical_window_generator_period_domain_records": [
                float(positive_periods[historical_period_mask][0]),
                float(positive_periods[historical_period_mask][-1]),
            ],
            "comparison": "all period algorithms evaluate identical positive and negative windows",
            "negative_interpretation": "adversarial historical window-corpus rejection, not current production PELT false-positive rate",
        },
        "corpus_counts": corpus_counts,
        "positive_attempt_count": positive_attempt_count,
        "positive_window_count": positive_window_count,
        "stage1_positive_recall": positive_window_count / max(1, positive_attempt_count),
        "negative_window_count": negative_window_count,
        "negative_to_positive_ratio": negative_window_count / max(1, positive_window_count),
        "summary_rows": summaries,
        "paired_new_vs_old": paired,
        "elapsed_seconds": perf_counter() - started,
        "images": [path.name for path in images],
    }
    summary_path = output / "cprf_stress_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str))
    new_row = next(row for row in summaries if row["algorithm"] == NEW_CPRF)
    old_row = next(row for row in summaries if row["algorithm"] == OLD_CPRF)
    lines = [
        "# CPRF Historical-Negative Stress Test",
        "",
        f"Independent positive attempts: `{payload['positive_attempt_count']}`",
        f"Positive PELT windows: `{payload['positive_window_count']}`",
        f"Stage-1 positive recall: `{100 * float(payload['stage1_positive_recall']):.2f}%`",
        f"Historical negative windows: `{payload['negative_window_count']}`",
        f"Negative:positive ratio: `{payload['negative_to_positive_ratio']:.3f}:1`",
        "",
        "| Algorithm | Conditional acceptance | End-to-end acceptance | Historical false acceptance | Accepted hit within 10% |",
        "|---|---:|---:|---:|---:|",
        (
            f"| New `{NEW_CPRF}` | {100 * float(new_row['positive_accept_rate']):.2f}% | "
            f"{100 * int(new_row['positive_accept_count']) / positive_attempt_count:.2f}% | "
            f"{int(new_row['negative_false_accept_count'])}/{int(new_row['negative_case_count'])} | "
            f"{100 * float(new_row['accepted_period_hit_10_rate']):.2f}% |"
        ),
        (
            f"| Old `{OLD_CPRF}` | {100 * float(old_row['positive_accept_rate']):.2f}% | "
            f"{100 * int(old_row['positive_accept_count']) / positive_attempt_count:.2f}% | "
            f"{int(old_row['negative_false_accept_count'])}/{int(old_row['negative_case_count'])} | "
            f"{100 * float(old_row['accepted_period_hit_10_rate']):.2f}% |"
        ),
        "",
        "The historical negative corpus is deliberately more permissive than the current production PELT path.",
        "Its false-acceptance rate measures CPRF rejection under stress and must not be reported as production FPR.",
        "",
        "## Manual Review Plots",
        "",
        "![new CPRF score plane](new_cprf_score_plane.png)",
        "![new CPRF gate plane](new_cprf_gate_plane.png)",
        "![new CPRF period accuracy](new_cprf_period_accuracy.png)",
        "![new CPRF harmonic diagnostics](new_cprf_harmonic_diagnostics.png)",
    ]
    (output / "RESULT.md").write_text("\n".join(lines) + "\n")
    return summary_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "runs/cprf_stress_legacy_negative_positive10to1_v1",
    )
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument(
        "--injections",
        type=Path,
        default=PERF_DIR / "configs/cprf_stress_positive_216.json",
    )
    parser.add_argument("--cwt-config", type=Path, default=PROJECT_DIR / "configs/cwt_default.json")
    parser.add_argument("--negative-channel-start", type=int, default=0)
    parser.add_argument("--negative-channel-stop", type=int, default=2048)
    parser.add_argument("--negative-block-channels", type=int, default=16)
    parser.add_argument("--max-positive-per-observation", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument(
        "--rethreshold-existing",
        type=Path,
        help="Replay the c40/r150/i300 gate candidate from an existing stress run.",
    )
    parser.add_argument(
        "--sweep-existing",
        type=Path,
        help="Sweep exact CPRF concentration/contrast breakpoints from an existing run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.sweep_existing is not None:
        summary = sweep_existing_stress(args.sweep_existing)
        print(f"Threshold sweep: {summary}")
        return
    if args.rethreshold_existing is not None:
        summary = rethreshold_existing_stress(args.rethreshold_existing)
        print(f"Threshold trial: {summary}")
        return
    summary = run(
        StressConfig(
            output_dir=args.output,
            inputs=tuple(args.input) if args.input else DEFAULT_INPUTS,
            injection_config=args.injections,
            cwt_config=args.cwt_config,
            negative_channel_start=args.negative_channel_start,
            negative_channel_stop=args.negative_channel_stop,
            negative_block_channels=args.negative_block_channels,
            max_positive_per_observation=args.max_positive_per_observation,
            progress_every=args.progress_every,
        )
    )
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
