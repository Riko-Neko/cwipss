"""Render representative Top1 CE4 detections for rapid visual review."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
PERF_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
for path in (PERF_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cwt_activity_algorithms import compute_cwt_activity, resolve_cwt_activity_algorithms  # noqa: E402
from cwt_activity_rank import (  # noqa: E402
    CWTActivityRun,
    _activity_windows,
    activity_config_from_cwt,
    prepare_activity_component,
)
from cwipss.analysis.injection_config import load_injection_config, make_injections_from_config  # noqa: E402
from cwipss.data.readers import open_spectrum_reader  # noqa: E402
from cwipss.signal.activity import robust_standardize  # noqa: E402
from cwipss.signal.activity import crop_valid_periods  # noqa: E402
from cwipss.signal.cwt import cwt_power_cube, period_grid_records  # noqa: E402


DEFAULT_ALGORITHM = "post_freq_max8_center_8cycle_s70_f20"
DEFAULT_RUN_DIR = PROJECT_DIR / "runs/compression_final/denoising_rank"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "runs/compression_final/top1_review_visualizations"

REPRESENTATIVES = (
    ("clean_strong", "inj_0028_weak_family_a_lowfreq_028_b001_c01", "Clean high-gain single-frequency hit"),
    ("weak_hit", "inj_0032_weak_family_a_lowfreq_032_b001_c01", "Weakest recovered single-frequency case"),
    ("multifreq_copy_a", "inj_0077_weak_family_b_multifreq_lowfreq_022_b001_c01", "Multi-frequency group, copy A"),
    ("multifreq_copy_b", "inj_0078_weak_family_b_multifreq_lowfreq_022_b001_c02", "Multi-frequency group, copy B"),
    ("borderline_hit", "inj_0091_weak_family_b_multifreq_lowfreq_029_b001_c01", "Borderline multi-frequency copy recovered"),
    ("borderline_miss", "inj_0092_weak_family_b_multifreq_lowfreq_029_b001_c02", "Same borderline group, second copy missed"),
    ("long_period", "inj_0004_weak_family_a_lowfreq_004_b001_c01", "Long-period edge case"),
    ("clear_miss", "inj_0112_weak_family_c_ultraweak_lowfreq_012_b001_c01", "Ultraweak miss / failure case"),
)


@dataclass
class RenderedCase:
    key: str
    case_id: str
    reason: str
    image: Path
    score_map: np.ndarray
    periods: np.ndarray
    activity_z: np.ndarray
    baseline_activity_z: np.ndarray
    truth_start: int
    truth_stop: int
    true_period: float
    windows: list[dict[str, float | int]]
    metrics: dict[str, str]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def _resolve_retained_path(run_dir: Path, payload: dict[str, Any], key: str, retained: str) -> Path:
    configured = Path(str(payload[key]))
    return configured if configured.exists() else run_dir / retained


def _downsample_time(values: np.ndarray, maximum: int = 1200) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[-1] <= maximum:
        return array
    indices = np.linspace(0, array.shape[-1] - 1, maximum).astype(np.int64)
    return array[..., indices]


def _render_case(
    *,
    key: str,
    case_id: str,
    reason: str,
    reader: Any,
    spec: Any,
    metrics: dict[str, str],
    config: Any,
    periods: np.ndarray,
    algorithm: Any,
    algorithm_label: str,
    metric_distributions: dict[str, np.ndarray],
    output_dir: Path,
) -> RenderedCase:
    mode = str(algorithm.input_denoiser)
    prepared, baseline, injected = prepare_activity_component(
        reader=reader,
        spec=spec,
        config=config,
        periods=periods,
        input_denoisers=(mode,),
    )
    result = compute_cwt_activity(
        prepared.cwt_power[mode],
        prepared.valid_periods,
        algorithm,
        reference_power=prepared.reference_cwt_power.get(mode),
    )
    baseline_result = compute_cwt_activity(
        prepared.baseline_cwt_power[mode],
        prepared.valid_periods,
        algorithm,
        reference_power=prepared.baseline_reference_cwt_power.get(mode),
    )
    activity_z = robust_standardize(result.activity)
    baseline_activity_z = robust_standardize(baseline_result.activity)
    windows = _activity_windows(activity_z, config)
    truth_start = int(prepared.truth["record_start"])
    truth_stop = int(prepared.truth["record_stop"])
    true_period = float(spec.period_records)

    channel_index = int(prepared.background_channel_start)
    channel_start = max(0, channel_index - 12)
    channel_stop = min(int(reader.n_channels), channel_index + 13)
    raw_block = reader.read_block(
        slice(prepared.background_record_start, prepared.background_record_stop),
        slice(channel_start, channel_stop),
    )
    raw = np.asarray(raw_block.data, dtype=np.float32).copy()
    raw[:, channel_index - channel_start] += np.asarray(injected - baseline, dtype=np.float32)
    raw_freqs = np.asarray(raw_block.freqs_mhz, dtype=np.float64)

    score = np.asarray(result.score_map, dtype=np.float32)
    baseline_score = np.asarray(baseline_result.score_map, dtype=np.float32)
    cwt_power = np.asarray(prepared.cwt_power[mode], dtype=np.float32)
    positive_power = cwt_power[cwt_power > 0.0]
    eps = max(float(np.nanmedian(positive_power)) * 1e-6, 1e-12) if positive_power.size else 1e-12
    log_power = np.log10(cwt_power + eps)

    fig = plt.figure(figsize=(16, 16), constrained_layout=True)
    grid = fig.add_gridspec(6, 1, height_ratios=(1.05, 1.0, 1.0, 1.0, 0.9, 0.7))
    ax_raw = fig.add_subplot(grid[0])
    ax_cwt = fig.add_subplot(grid[1], sharex=ax_raw)
    ax_score = fig.add_subplot(grid[2], sharex=ax_raw)
    ax_base = fig.add_subplot(grid[3], sharex=ax_raw)
    ax_activity = fig.add_subplot(grid[4], sharex=ax_raw)
    ax_metrics = fig.add_subplot(grid[5])

    records = int(raw.shape[0])
    raw_lo, raw_hi = np.nanquantile(raw, [0.01, 0.99])
    raw_image = ax_raw.imshow(
        raw.T,
        origin="lower",
        aspect="auto",
        extent=(0, records, float(raw_freqs[0]), float(raw_freqs[-1])),
        cmap="cividis",
        vmin=float(raw_lo),
        vmax=float(raw_hi),
        interpolation="nearest",
    )
    ax_raw.axhline(float(reader.freqs_mhz[channel_index]), color="#ef476f", linewidth=1.1)
    ax_raw.set_ylabel("Frequency (MHz)")
    ax_raw.set_title("Raw CE4 time-frequency neighborhood with injected target channel")
    fig.colorbar(raw_image, ax=ax_raw, fraction=0.018, pad=0.01, label="Amplitude")

    def draw_map(ax: Any, values: np.ndarray, title: str, cmap: str, *, floor_zero: bool = False) -> None:
        finite = values[np.isfinite(values)]
        if floor_zero:
            lo, hi = 0.0, max(0.2, float(np.nanquantile(finite, 0.9995)) if finite.size else 1.0)
        else:
            lo = float(np.nanquantile(finite, 0.02)) if finite.size else 0.0
            hi = float(np.nanquantile(finite, 0.995)) if finite.size else 1.0
        image = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=(0, records, float(prepared.valid_periods[0]), float(prepared.valid_periods[-1])),
            cmap=cmap,
            vmin=lo,
            vmax=hi,
            interpolation="nearest",
        )
        ax.set_yscale("log")
        ax.axhline(true_period, color="#f6bd60", linewidth=1.0, linestyle="--")
        ax.set_ylabel("Period (records)")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.018, pad=0.01)

    draw_map(ax_cwt, log_power, "Target-channel raw CWT log-power", "magma")
    draw_map(ax_score, score, f"{algorithm_label} score map after injection", "inferno", floor_zero=True)
    draw_map(ax_base, baseline_score, f"{algorithm_label} score map on paired no-injection baseline", "inferno", floor_zero=True)

    for ax in (ax_raw, ax_cwt, ax_score, ax_base, ax_activity):
        ax.axvspan(truth_start, truth_stop, color="#06d6a0", alpha=0.12)
    ax_activity.plot(activity_z, color="#ef476f", linewidth=1.2, label="Injected activity z")
    ax_activity.plot(baseline_activity_z, color="#118ab2", linewidth=0.9, alpha=0.8, label="Baseline activity z")
    for index, window in enumerate(windows):
        ax_activity.axvspan(
            int(window["record_start"]),
            int(window["record_stop"]),
            color="#ffd166",
            alpha=0.18,
            label="PELT candidate" if index == 0 else None,
        )
    ax_activity.set_ylabel("Activity z")
    ax_activity.set_xlabel("Local record")
    ax_activity.set_title(f"{algorithm_label} 1D activity and candidate windows")
    ax_activity.legend(loc="upper right", ncol=3, fontsize=8)
    ax_activity.grid(alpha=0.18)

    metric_specs = (
        ("Global band contrast", "peak_global_band_contrast"),
        ("Period concentration", "peak_period_concentration"),
        ("Truth-window local contrast", "truth_window_local_band_contrast"),
    )
    percentiles: list[float] = []
    raw_values: list[float] = []
    for _label, field in metric_specs:
        value = float(metrics[field])
        distribution = metric_distributions[field]
        percentiles.append(float(np.mean(distribution <= value)))
        raw_values.append(value)
    bars = ax_metrics.barh(
        [label for label, _field in metric_specs],
        percentiles,
        color=("#ef476f", "#ffd166", "#118ab2"),
    )
    ax_metrics.set_xlim(0.0, 1.08)
    ax_metrics.set_xlabel("Empirical percentile among all 133 injected components")
    ax_metrics.set_title("Three scientific performance parameters")
    ax_metrics.grid(axis="x", alpha=0.18)
    for bar, percentile, raw_value in zip(bars, percentiles, raw_values, strict=True):
        ax_metrics.text(
            min(percentile + 0.015, 1.01),
            bar.get_y() + bar.get_height() / 2,
            f"raw={raw_value:.3g}",
            va="center",
            fontsize=9,
        )

    status = "HIT" if metrics["truth_window_hit"] == "1" else "MISS"
    fig.suptitle(
        f"{reason} | {status} | {case_id}\n"
        f"period={true_period:.3g}, amplitude/noise={float(metrics['amplitude_factor']):.3g}, "
        f"activity gain={float(metrics['paired_activity_gain_mean']):.3g}, "
        f"band gain={float(metrics['paired_band_gain_mean']):.3g}, "
        f"period error={float(metrics['truth_window_period_error_fraction']):.3g}",
        fontsize=13,
    )
    image_path = output_dir / f"{key}_{case_id}.png"
    fig.savefig(image_path, dpi=150)
    plt.close(fig)

    return RenderedCase(
        key=key,
        case_id=case_id,
        reason=reason,
        image=image_path,
        score_map=_downsample_time(score),
        periods=np.asarray(prepared.valid_periods, dtype=np.float64),
        activity_z=_downsample_time(activity_z),
        baseline_activity_z=_downsample_time(baseline_activity_z),
        truth_start=truth_start,
        truth_stop=truth_stop,
        true_period=true_period,
        windows=windows,
        metrics=metrics,
    )


def _render_overview(cases: list[RenderedCase], output_dir: Path, algorithm_label: str) -> Path:
    fig, axes = plt.subplots(4, 4, figsize=(20, 17), constrained_layout=True)
    for case_index, case in enumerate(cases):
        row = case_index // 2
        column = (case_index % 2) * 2
        ax_map = axes[row, column]
        ax_activity = axes[row, column + 1]
        records = int(float(case.metrics["records"]))
        ax_map.imshow(
            case.score_map,
            origin="lower",
            aspect="auto",
            extent=(0, records, float(case.periods[0]), float(case.periods[-1])),
            cmap="inferno",
            vmin=0.0,
            vmax=max(0.2, float(np.nanquantile(case.score_map, 0.9995))),
            interpolation="nearest",
        )
        ax_map.set_yscale("log")
        ax_map.axvspan(case.truth_start, case.truth_stop, color="#06d6a0", alpha=0.13)
        ax_map.axhline(case.true_period, color="#f6bd60", linewidth=0.9, linestyle="--")
        ax_map.set_title(f"{case.key}: {algorithm_label} score map", fontsize=9)
        ax_map.set_ylabel("Period")

        x = np.linspace(0, records, case.activity_z.size)
        ax_activity.plot(x, case.activity_z, color="#ef476f", linewidth=1.0)
        ax_activity.plot(x, case.baseline_activity_z, color="#118ab2", linewidth=0.8, alpha=0.75)
        ax_activity.axvspan(case.truth_start, case.truth_stop, color="#06d6a0", alpha=0.13)
        for window in case.windows:
            ax_activity.axvspan(
                int(window["record_start"]),
                int(window["record_stop"]),
                color="#ffd166",
                alpha=0.18,
            )
        status = "HIT" if case.metrics["truth_window_hit"] == "1" else "MISS"
        ax_activity.set_title(
            f"{status} | gain={float(case.metrics['paired_activity_gain_mean']):.2f} | "
            f"P={case.true_period:.1f}",
            fontsize=9,
        )
        ax_activity.set_ylabel("Activity z")
        ax_activity.grid(alpha=0.16)
    for ax in axes[-1, :]:
        ax.set_xlabel("Local record")
    fig.suptitle(f"{algorithm_label} representative CE4 review: injected score maps and activity", fontsize=15)
    path = output_dir / "00_representative_overview.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _render_negative_case(
    *,
    reader: Any,
    row: dict[str, str],
    config: Any,
    periods: np.ndarray,
    algorithm: Any,
    algorithm_label: str,
    output_dir: Path,
) -> Path:
    channel_index = int(row["channel_index"])
    channel_start = max(0, channel_index - 12)
    channel_stop = min(int(reader.n_channels), channel_index + 13)
    block = reader.read_block(slice(0, int(reader.n_records)), slice(channel_start, channel_stop))
    raw = np.asarray(block.data, dtype=np.float32)
    target = raw[:, channel_index - channel_start]
    power = cwt_power_cube(
        target[:, None],
        periods,
        wavelet=config.wavelet,
        normalize_channels=True,
        method=config.cwt_method,
        backend=config.cwt_backend,
        cuda_device=config.cuda_device,
    )[:, :, 0]
    valid_power, valid_periods, _mask = crop_valid_periods(
        power,
        periods,
        config.candidate_period_min_records,
        config.candidate_period_max_records,
    )
    result = compute_cwt_activity(valid_power, valid_periods, algorithm, reference_power=None)
    activity_z = robust_standardize(result.activity)
    windows = _activity_windows(activity_z, config)

    positive_power = valid_power[valid_power > 0.0]
    eps = max(float(np.nanmedian(positive_power)) * 1e-6, 1e-12) if positive_power.size else 1e-12
    log_power = np.log10(valid_power + eps)
    score = np.asarray(result.score_map, dtype=np.float32)
    raw_plot = _downsample_time(raw.T)
    cwt_plot = _downsample_time(log_power)
    score_plot = _downsample_time(score)
    activity_plot = _downsample_time(activity_z)
    records = int(reader.n_records)
    x_activity = np.linspace(0, records - 1, activity_plot.size)
    freqs = np.asarray(block.freqs_mhz, dtype=np.float64)

    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    grid = fig.add_gridspec(4, 1, height_ratios=(1.0, 1.15, 1.15, 0.9))
    ax_raw = fig.add_subplot(grid[0])
    ax_cwt = fig.add_subplot(grid[1], sharex=ax_raw)
    ax_score = fig.add_subplot(grid[2], sharex=ax_raw)
    ax_activity = fig.add_subplot(grid[3], sharex=ax_raw)
    raw_lo, raw_hi = np.nanquantile(raw_plot, [0.01, 0.99])
    image = ax_raw.imshow(
        raw_plot,
        origin="lower",
        aspect="auto",
        extent=(0, records, float(freqs[0]), float(freqs[-1])),
        cmap="cividis",
        vmin=float(raw_lo),
        vmax=float(raw_hi),
        interpolation="nearest",
    )
    ax_raw.axhline(float(reader.freqs_mhz[channel_index]), color="#ef476f", linewidth=1.0)
    ax_raw.set_ylabel("Frequency (MHz)")
    ax_raw.set_title("Raw CE4 time-frequency neighborhood; no injected signal")
    fig.colorbar(image, ax=ax_raw, fraction=0.018, pad=0.01, label="Amplitude")

    def draw_map(ax: Any, values: np.ndarray, title: str, cmap: str, floor_zero: bool) -> None:
        finite = values[np.isfinite(values)]
        lo = 0.0 if floor_zero else float(np.nanquantile(finite, 0.02))
        hi = float(np.nanquantile(finite, 0.9995 if floor_zero else 0.995))
        image = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=(0, records, float(valid_periods[0]), float(valid_periods[-1])),
            cmap=cmap,
            vmin=lo,
            vmax=max(lo + 1e-6, hi),
            interpolation="nearest",
        )
        ax.set_yscale("log")
        ax.set_ylabel("Period (records)")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.018, pad=0.01)

    draw_map(ax_cwt, cwt_plot, "Target-channel raw CWT log-power", "magma", False)
    draw_map(ax_score, score_plot, f"{algorithm_label} score map", "inferno", True)
    ax_activity.plot(x_activity, activity_plot, color="#ef476f", linewidth=0.9)
    for index, window in enumerate(windows):
        ax_activity.axvspan(
            int(window["record_start"]),
            int(window["record_stop"]),
            color="#ffd166",
            alpha=0.22,
            label="False PELT candidate" if index == 0 else None,
        )
    ax_activity.set_xlabel("Record")
    ax_activity.set_ylabel("Activity z")
    ax_activity.set_title(f"{algorithm_label} activity: {len(windows)} false candidate windows")
    ax_activity.grid(alpha=0.18)
    if windows:
        ax_activity.legend(loc="upper right")
    fig.suptitle(
        f"Real-data negative control | channel={channel_index} | "
        f"frequency={float(row['frequency_mhz']):.6f} MHz | no injection",
        fontsize=13,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"negative_ch{channel_index:03d}_{float(row['frequency_mhz']):.6f}mhz.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render_review(
    run_dir: Path,
    output_dir: Path,
    algorithm_name: str,
    algorithm_label: str,
    negative_top: int,
) -> None:
    payload = json.loads((run_dir / "cwt_activity_summary.json").read_text())
    input_path = Path(str(payload["input_path"]))
    injection_config = _resolve_retained_path(run_dir, payload, "injection_config", "injection_config.json")
    cwt_config = _resolve_retained_path(run_dir, payload, "cwt_config", "cwt_config.json")
    run = CWTActivityRun(
        output_dir=run_dir,
        input_path=input_path,
        injection_config=injection_config,
        cwt_config=cwt_config,
        algorithms=(algorithm_name,),
        cwt_backend=str(payload["config"]["cwt_backend"]),
        cuda_device=int(payload["config"]["cuda_device"]),
        candidate_period_max_records=float(payload["config"]["candidate_period_max_records"]),
        progress_every=0,
        negative_control=False,
    )
    config = activity_config_from_cwt(run)
    reader = open_spectrum_reader(input_path)
    specs = {
        spec.injection_id: spec
        for spec in make_injections_from_config(
            load_injection_config(injection_config),
            records=reader.n_records,
            channels=reader.n_channels,
            freqs_mhz=reader.freqs_mhz,
        )
    }
    metric_rows = {
        row["case_id"]: row
        for row in _read_rows(run_dir / "component_cases.csv")
        if row["algorithm"] == algorithm_name
    }
    metric_distributions = {
        field: np.asarray([float(row[field]) for row in metric_rows.values()], dtype=np.float64)
        for field in (
            "peak_global_band_contrast",
            "peak_period_concentration",
            "truth_window_local_band_contrast",
        )
    }
    periods = period_grid_records(
        config.period_min_records,
        config.period_max_records,
        config.period_count,
        config.period_spacing,
    )
    algorithm = resolve_cwt_activity_algorithms((algorithm_name,))[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        _render_case(
            key=key,
            case_id=case_id,
            reason=reason,
            reader=reader,
            spec=specs[case_id],
            metrics=metric_rows[case_id],
            config=config,
            periods=periods,
            algorithm=algorithm,
            algorithm_label=algorithm_label,
            metric_distributions=metric_distributions,
            output_dir=output_dir,
        )
        for key, case_id, reason in REPRESENTATIVES
    ]
    overview = _render_overview(cases, output_dir, algorithm_label)
    negative_images: list[str] = []
    negative_path = run_dir / "negative_control_cases.csv"
    if negative_top > 0 and negative_path.exists():
        negative_rows = [
            row for row in _read_rows(negative_path)
            if row["algorithm"] == algorithm_name and int(float(row["window_count"])) > 0
        ]
        negative_rows.sort(key=lambda row: int(float(row["window_count"])), reverse=True)
        negative_dir = output_dir / "real_negative_failures"
        for row in negative_rows[:negative_top]:
            image = _render_negative_case(
                reader=reader,
                row=row,
                config=config,
                periods=periods,
                algorithm=algorithm,
                algorithm_label=algorithm_label,
                output_dir=negative_dir,
            )
            negative_images.append(str(image.relative_to(output_dir)))

    fields = [
        "key",
        "case_id",
        "reason",
        "truth_window_hit",
        "peak_in_truth",
        "paired_activity_detected",
        "paired_band_detected",
        "amplitude_factor",
        "period_records",
        "paired_activity_gain_mean",
        "paired_band_gain_mean",
        "truth_window_period_error_fraction",
        "window_count",
        "image",
    ]
    with (output_dir / "review_index.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            row = {field: case.metrics.get(field, "") for field in fields}
            row.update(
                {
                    "key": case.key,
                    "case_id": case.case_id,
                    "reason": case.reason,
                    "image": case.image.name,
                }
            )
            writer.writerow(row)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "algorithm": algorithm_name,
                "algorithm_label": algorithm_label,
                "source_run": str(run_dir),
                "overview": overview.name,
                "real_negative_failures": negative_images,
                "cases": [
                    {
                        "key": case.key,
                        "case_id": case.case_id,
                        "reason": case.reason,
                        "image": case.image.name,
                    }
                    for case in cases
                ],
            },
            indent=2,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render representative Top1 CE4 review figures.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--algorithm", default=DEFAULT_ALGORITHM)
    parser.add_argument("--label", default="Top1")
    parser.add_argument("--negative-top", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    render_review(args.run_dir, args.output, args.algorithm, args.label, args.negative_top)
