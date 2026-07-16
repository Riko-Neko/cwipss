"""Wavelet-basis injection diagnostics for pytest-driven scientific tests."""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pywt

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cwipss.analysis.injection_config import load_injection_config, make_injections_from_config
from cwipss.analysis.simulation import InjectionSpec, inject_periodic_signal, injection_truth
from cwipss.config import CWTSearchConfig, cwt_config_to_nested_dict, load_cwt_config
from cwipss.data.readers import CE4_RECORD_LEN, open_spectrum_reader
from cwipss.reporting.plotting import (
    CWT_POWER_CMAP,
    CWT_POWER_COLORBAR,
    cwt_power_display_values,
    edges,
    finite_percentile_limits,
    save_figure,
)
from cwipss.signal.activity import (
    crop_valid_periods,
    robust_standardize,
)
from cwipss.signal.cpro import CPROParameters, cpro_activity, difference_noise_std, impulse_cwt_noise_gain
from cwipss.signal.cwt import cwt_power_cube, period_grid_records

PARAMETERIZED_WAVELET_NAMES = {
    "cmor": "cmor1.5-1.0",
    "fbsp": "fbsp2-1.0-0.5",
    "shan": "shan0.5-1.0",
}
INJECTION_COLOR = "#39ff14"


@dataclass(frozen=True)
class WaveletBasisRunConfig:
    input_path: Path | None
    input_dir: Path
    injection_config: Path
    cwt_config: Path
    output_dir: Path
    wavelets: tuple[str, ...]
    t_start: int | None = None
    t_stop: int | None = None
    period_min_records: float | None = None
    period_max_records: float | None = None
    period_count: int | None = None
    period_spacing: str | None = None
    candidate_period_min_records: float | None = None
    candidate_period_max_records: float | None = None
    cwt_method: str | None = None
    cwt_backend: str = "cpu"
    cuda_device: int | None = None
    max_injections: int = 0
    max_wavelets: int = 0
    dpi: int = 140
    progress_every: int = 10


@dataclass(frozen=True)
class WaveletBasisRunResult:
    output_dir: Path
    input_path: Path
    index_path: Path
    summary_path: Path
    truth_path: Path
    wavelets: list[str]
    injection_count: int
    case_count: int


def all_continuous_wavelets() -> list[str]:
    return [PARAMETERIZED_WAVELET_NAMES.get(name, name) for name in pywt.wavelist(kind="continuous")]


def resolve_wavelets(values: Iterable[str], *, max_wavelets: int = 0) -> list[str]:
    requested = [str(value).strip() for value in values if str(value).strip()]
    if not requested or any(value.lower() == "all" for value in requested):
        wavelets = all_continuous_wavelets()
    else:
        wavelets = [PARAMETERIZED_WAVELET_NAMES.get(value, value) for value in requested]
    unique: list[str] = []
    for wavelet in wavelets:
        if wavelet in unique:
            continue
        central = float(pywt.central_frequency(wavelet))
        if not np.isfinite(central) or central <= 0.0:
            raise ValueError(f"Cannot derive a positive central frequency for wavelet: {wavelet}")
        unique.append(wavelet)
    return unique[: int(max_wavelets)] if int(max_wavelets) > 0 else unique


def largest_complete_2c(input_dir: str | Path) -> Path:
    root = Path(input_dir)
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".2c")
    complete = [path for path in files if path.stat().st_size > 0 and path.stat().st_size % CE4_RECORD_LEN == 0]
    if not complete:
        raise FileNotFoundError(f"No complete CE4 .2C files found under: {root}")
    return max(complete, key=lambda path: path.stat().st_size)


def _load_run_cwt_config(config: WaveletBasisRunConfig) -> CWTSearchConfig:
    overrides = {
        "period_min_records": config.period_min_records,
        "period_max_records": config.period_max_records,
        "period_count": config.period_count,
        "period_spacing": config.period_spacing,
        "candidate_period_min_records": config.candidate_period_min_records,
        "candidate_period_max_records": config.candidate_period_max_records,
        "cwt_method": config.cwt_method,
        "cwt_backend": config.cwt_backend,
        "cuda_device": config.cuda_device,
    }
    return load_cwt_config(config.cwt_config,
                           overrides={key: value for key, value in overrides.items() if value is not None})


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "value"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _clean_previous_outputs(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for pattern in (
        "*__local_cwt_diagnostic_panel.png",
        "config.resolved.json",
        "index.md",
        "injection_truth.csv",
        "wavelet_summary.csv",
    ):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _truth_for_spec(spec: InjectionSpec, reader, record_offset: int) -> dict[str, Any]:
    truth = injection_truth(spec, reader.n_channels)
    start = int(truth["channel_start"])
    stop = int(truth["channel_stop"])
    stop_idx = min(max(stop - 1, start), reader.freqs_mhz.size - 1)
    center_idx = min(max(int(round(float(truth["channel_center"]))), 0), reader.freqs_mhz.size - 1)
    truth["record_start"] = int(record_offset + int(truth["record_start"]))
    truth["record_stop"] = int(record_offset + int(truth["record_stop"]))
    truth["freq_start_mhz"] = float(reader.freqs_mhz[start])
    truth["freq_stop_mhz"] = float(reader.freqs_mhz[stop_idx])
    truth["freq_center_mhz"] = float(reader.freqs_mhz[center_idx])
    return truth


def _inject_single_channel(data: np.ndarray, spec: InjectionSpec) -> tuple[np.ndarray, np.ndarray]:
    local_spec = replace(spec, channel_center=0.0, bandwidth_channels=1.0, drift_channels=0.0)
    baseline = np.asarray(data, dtype=np.float32)
    injected, _truth = inject_periodic_signal(baseline[:, None], local_spec)
    return baseline, np.asarray(injected[:, 0], dtype=np.float32)


def _compress_with_pipeline_functions(
    power_channel: np.ndarray,
    periods: np.ndarray,
    config: CWTSearchConfig,
    *,
    baseline: np.ndarray,
    wavelet: str,
) -> dict[str, Any]:
    valid_power, valid_periods, _mask = crop_valid_periods(
        power_channel,
        periods,
        config.candidate_period_min_records,
        config.candidate_period_max_records,
    )
    noise_gain = impulse_cwt_noise_gain(valid_periods, wavelet=wavelet, method=config.cwt_method)
    result = cpro_activity(
        valid_power,
        noise_std=difference_noise_std(baseline),
        noise_gain=noise_gain,
        params=CPROParameters(
            threshold_snr=config.cpro_threshold_snr,
            texture_quantile=config.cpro_texture_quantile,
            period_center_bins=config.cpro_period_center_bins,
            period_context_bins=config.cpro_period_context_bins,
            min_period_contrast=config.cpro_min_period_contrast,
            support_records=config.cpro_support_records,
            min_occupancy=config.cpro_min_occupancy,
            period_support_bins=config.cpro_period_support_bins,
            window_support_records=config.cpro_window_support_records,
            min_window_occupancy=config.cpro_min_window_occupancy,
        ),
    )
    activity_raw = np.asarray(result.activity, dtype=np.float32)
    activity_z = robust_standardize(activity_raw)
    return {
        "valid_periods": valid_periods,
        "structured": np.asarray(result.score_map, dtype=np.float32),
        "activity_raw": activity_raw,
        "activity_smooth": activity_raw,
        "activity_z": activity_z,
        "calibrated_threshold": float(result.threshold),
    }


def _raw_signal_plot(path: Path, baseline: np.ndarray, injected: np.ndarray, truth: dict[str, Any], offset: int,
                     title: str, dpi: int) -> Path:
    x = np.arange(offset, offset + baseline.size, dtype=np.int64)
    delta = injected - baseline

    def draw(ax) -> None:
        ax.plot(x, baseline, color="#7a7a7a", linewidth=0.7, alpha=0.7, label="background channel")
        ax.plot(x, injected, color="#0b5fff", linewidth=0.8, alpha=0.9, label="injected channel")
        ax.axvspan(
            float(truth["record_start"]),
            float(truth["record_stop"]),
            color=INJECTION_COLOR,
            alpha=0.12,
            linewidth=0,
        )
        ax.set(title=title, xlabel="Record", ylabel="Amplitude")
        ax2 = ax.twinx()
        ax2.plot(x, delta, color="#d1495b", linewidth=0.8, alpha=0.8, label="injected signal only")
        ax2.set_ylabel("Injected signal amplitude")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best", fontsize="small")

    return save_figure(path, dpi, draw)


def _activity_plot(path: Path, compressed: dict[str, Any], truth: dict[str, Any], offset: int, title: str,
                   dpi: int) -> Path:
    activity_raw = np.asarray(compressed["activity_raw"], dtype=np.float32)
    activity_smooth = np.asarray(compressed["activity_smooth"], dtype=np.float32)
    activity_z = np.asarray(compressed["activity_z"], dtype=np.float32)
    x = np.arange(offset, offset + activity_raw.size, dtype=np.int64)

    def draw(ax) -> None:
        ax.plot(x, activity_raw, color="#3454d1", linewidth=0.9, label="CPRO calibrated activity")
        ax.axvspan(
            float(truth["record_start"]),
            float(truth["record_stop"]),
            color=INJECTION_COLOR,
            alpha=0.12,
            linewidth=0,
        )
        ax.set(title=title, xlabel="Record", ylabel="Current-pipeline 1D activity")
        ax2 = ax.twinx()
        ax2.plot(x, activity_z, color="#111111", linewidth=0.8, alpha=0.8, label="robust standardized activity")
        ax2.set_ylabel("Robust standardized activity")
        ax.text(0.01, 0.98, f"truth period={float(truth['period_records']):.3g} records", transform=ax.transAxes,
                ha="left", va="top", fontsize="small")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best", fontsize="small")

    return save_figure(path, dpi, draw)


def _local_record_window(truth: dict[str, Any], selected_start: int, selected_stop: int) -> tuple[int, int]:
    """Return a compact absolute record window around one injected signal.

    The plot keeps 50% of the injected duration as context on each side.
    """
    signal_start = int(truth["record_start"])
    signal_stop = int(truth["record_stop"])
    signal_len = max(1, signal_stop - signal_start)
    margin = max(1, int(math.ceil(0.5 * signal_len)))
    start = max(int(selected_start), signal_start - margin)
    stop = min(int(selected_stop), signal_stop + margin)
    if stop <= start:
        raise ValueError(f"Invalid local CWT window for {truth}: start={start}, stop={stop}")
    return start, stop


def _raw_time_frequency_panel_data(reader, record_slice, channel_idx: int, baseline: np.ndarray, injected: np.ndarray,
                                   local_start: int, local_stop: int, *, half_width: int = 24) -> tuple[
    np.ndarray, np.ndarray, int, int]:
    """Return local time-frequency image data with the injected target channel overlaid.

    Rows are nearby frequency channels, columns are local records. This avoids a
    full-band image while still showing whether the synthetic signal lands in the
    intended time-frequency cell.
    """
    ch0 = max(0, int(channel_idx) - int(half_width))
    ch1 = min(int(reader.n_channels), int(channel_idx) + int(half_width) + 1)
    local_record_slice = slice(int(record_slice.start) + int(local_start), int(record_slice.start) + int(local_stop))
    block = reader.read_block(local_record_slice, slice(ch0, ch1))
    tf = np.asarray(block.data, dtype=np.float32).T.copy()
    target_row = int(channel_idx) - ch0
    if 0 <= target_row < tf.shape[0]:
        tf[target_row, :] += np.asarray(injected - baseline, dtype=np.float32)
    return tf, np.asarray(reader.freqs_mhz[ch0:ch1], dtype=np.float64), ch0, ch1


def _combined_diagnostic_plot(
        path: Path,
        raw_tf: np.ndarray,
        raw_tf_freqs: np.ndarray,
        baseline: np.ndarray,
        injected: np.ndarray,
        power_channel: np.ndarray,
        periods: np.ndarray,
        compressed: dict[str, Any],
        truth: dict[str, Any],
        channel_idx: int,
        channel_start: int,
        offset: int,
        title: str,
        dpi: int,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(offset, offset + injected.size, dtype=np.int64)
    valid_periods = np.asarray(compressed["valid_periods"], dtype=np.float64)
    activity_raw = np.asarray(compressed["activity_raw"], dtype=np.float32)
    activity_smooth = np.asarray(compressed["activity_smooth"], dtype=np.float32)
    activity_z = np.asarray(compressed["activity_z"], dtype=np.float32)
    structured = np.asarray(compressed["structured"], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(title, fontsize=12)

    ax = axes[0, 0]
    raw_tf = np.asarray(raw_tf, dtype=np.float32)
    raw_tf_freqs = np.asarray(raw_tf_freqs, dtype=np.float64)
    if raw_tf.size:
        lo, hi = np.nanpercentile(raw_tf, [2.0, 98.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(raw_tf)), float(np.nanmax(raw_tf))
    else:
        lo, hi = 0.0, 1.0
    y0 = float(np.nanmin(raw_tf_freqs))
    y1 = float(np.nanmax(raw_tf_freqs))
    if y1 <= y0:
        y1 = y0 + 1.0
    im = ax.imshow(
        raw_tf,
        aspect="auto",
        origin="lower",
        extent=(float(x[0]), float(x[-1] + 1), y0, y1),
        interpolation="nearest",
        vmin=lo,
        vmax=hi,
    )
    freq_a = float(truth.get("freq_start_mhz", truth.get("freq_center_mhz", raw_tf_freqs[0])))
    freq_b = float(truth.get("freq_stop_mhz", truth.get("freq_center_mhz", raw_tf_freqs[-1])))
    freq0 = min(freq_a, freq_b)
    freq1 = max(freq_a, freq_b)
    if freq1 <= freq0:
        freq_step = abs(float(np.nanmedian(np.diff(raw_tf_freqs)))) if raw_tf_freqs.size > 1 else 1.0
        freq0 -= 0.5 * freq_step
        freq1 += 0.5 * freq_step
    import matplotlib.patches as patches
    box = patches.Rectangle(
        (float(truth["record_start"]), freq0),
        float(truth["record_stop"]) - float(truth["record_start"]),
        freq1 - freq0,
        fill=False,
        linewidth=1.6,
        edgecolor=INJECTION_COLOR,
    )
    ax.add_patch(box)
    ax.axhline(float(
        truth.get("freq_center_mhz", raw_tf_freqs[min(max(channel_idx - channel_start, 0), raw_tf_freqs.size - 1)])),
               color=INJECTION_COLOR, linestyle="--", linewidth=0.8)
    ax.set_title("raw time-frequency + injection box")
    ax.set_xlabel("Record")
    ax.set_ylabel("Frequency / MHz")
    fig.colorbar(im, ax=ax, label="raw power")

    ax = axes[0, 1]
    display_power = cwt_power_display_values(power_channel)
    vmin, vmax = finite_percentile_limits(display_power)
    im = ax.pcolormesh(
        np.arange(offset, offset + display_power.shape[1] + 1, dtype=np.float64),
        edges(periods, True),
        display_power,
        shading="auto",
        cmap=CWT_POWER_CMAP,
        vmin=vmin,
        vmax=vmax,
    )
    ax.axhline(float(truth["period_records"]), color=INJECTION_COLOR, linestyle="--", linewidth=0.9,
               label="truth period")
    ax.axvspan(float(truth["record_start"]), float(truth["record_stop"]), color=INJECTION_COLOR, alpha=0.10,
               linewidth=0)
    ax.set_yscale("log")
    ax.set_title("target injected signal raw CWT power 2D")
    ax.set_xlabel("Record")
    ax.set_ylabel("Period / records")
    ax.legend(loc="best", fontsize="small")
    fig.colorbar(im, ax=ax, label=CWT_POWER_COLORBAR)

    ax = axes[1, 0]
    vmin, vmax = finite_percentile_limits(structured)
    im = ax.pcolormesh(
        np.arange(offset, offset + structured.shape[1] + 1, dtype=np.float64),
        edges(valid_periods, True),
        structured,
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax.axhline(float(truth["period_records"]), color=INJECTION_COLOR, linestyle="--", linewidth=0.9,
               label="truth period")
    ax.axvspan(float(truth["record_start"]), float(truth["record_stop"]), color=INJECTION_COLOR, alpha=0.10,
               linewidth=0)
    ax.set_yscale("log")
    ax.set_title("2D map before 1D compression")
    ax.set_xlabel("Record")
    ax.set_ylabel("Period / records")
    ax.legend(loc="best", fontsize="small")
    fig.colorbar(im, ax=ax, label="CPRO calibrated score")

    ax = axes[1, 1]
    ax.plot(x, activity_raw, linewidth=0.9, label="CPRO calibrated activity")
    ax.axvspan(float(truth["record_start"]), float(truth["record_stop"]), color=INJECTION_COLOR, alpha=0.12,
               linewidth=0)
    ax2 = ax.twinx()
    ax2.plot(x, activity_z, linewidth=0.8, alpha=0.75, label="robust z")
    ax.set_title("local 1D compression")
    ax.set_xlabel("Record")
    ax.set_ylabel("CPRO activity")
    ax2.set_ylabel("Robust z")
    ax.text(
        0.01,
        0.98,
        f"window={int(x[0])}-{int(x[-1] + 1)}; truth period={float(truth['period_records']):.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize="small",
    )
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best", fontsize="small")

    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _write_index(path: Path, title: str, items: Iterable[tuple[str, Path, str]], root: Path) -> None:
    lines = [f"# {title}", "", "Wavelet-basis injection diagnostics.", ""]
    for item_title, image_path, note in items:
        lines += [f"## {item_title}", "", note, "", f"![{item_title}]({image_path.relative_to(root)})", ""]
    path.write_text("\n".join(lines))


def _assert_supported_injection_specs(specs: list[InjectionSpec]) -> None:
    unsupported = [
        spec
        for spec in specs
        if spec.signal_model != "single_channel_periodic"
        or not math.isclose(float(spec.bandwidth_channels), 1.0)
        or not math.isclose(float(spec.drift_channels), 0.0)
    ]
    if not unsupported:
        return
    examples = ", ".join(
        f"{spec.injection_id}(model={spec.signal_model}, bw={spec.bandwidth_channels}, drift={spec.drift_channels})"
        for spec in unsupported[:5]
    )
    raise ValueError(
        "Wavelet-basis injected-channel diagnostics currently support only one-channel, non-drifting "
        f"single_channel_periodic specs; unsupported examples: {examples}"
    )


def _summary_row(wavelet: str, spec: InjectionSpec, truth: dict[str, Any], channel_idx: int, compressed: dict[str, Any],
                 cwt_seconds: float, compress_seconds: float) -> dict[str, Any]:
    structured = np.asarray(compressed["structured"], dtype=np.float32)
    activity_raw = np.asarray(compressed["activity_raw"], dtype=np.float32)
    activity_z = np.asarray(compressed["activity_z"], dtype=np.float32)
    profile = np.nanmean(structured, axis=1) if structured.size else np.zeros(0, dtype=np.float32)
    valid_periods = np.asarray(compressed["valid_periods"], dtype=np.float64)
    peak_idx = int(np.nanargmax(profile)) if profile.size else -1
    return {
        "wavelet": wavelet,
        "injection_id": spec.injection_id,
        "signal_model": spec.signal_model,
        "channel_index": channel_idx,
        "freq_center_mhz": truth["freq_center_mhz"],
        "truth_period_records": truth["period_records"],
        "truth_amplitude": truth["amplitude"],
        "record_start": truth["record_start"],
        "record_stop": truth["record_stop"],
        "calibrated_threshold": compressed["calibrated_threshold"],
        "profile_peak_period_records": float(valid_periods[peak_idx]) if peak_idx >= 0 else math.nan,
        "activity_raw_mean": float(np.nanmean(activity_raw)) if activity_raw.size else 0.0,
        "activity_raw_max": float(np.nanmax(activity_raw)) if activity_raw.size else 0.0,
        "activity_z_max": float(np.nanmax(activity_z)) if activity_z.size else 0.0,
        "structured_mean": float(np.nanmean(structured)) if structured.size else 0.0,
        "structured_max": float(np.nanmax(structured)) if structured.size else 0.0,
        "cwt_seconds": cwt_seconds,
        "compress_seconds": compress_seconds,
    }


def run_wavelet_basis_injection(config: WaveletBasisRunConfig) -> WaveletBasisRunResult:
    input_path = config.input_path or largest_complete_2c(config.input_dir)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _clean_previous_outputs(output_dir)
    cwt_config = _load_run_cwt_config(config)
    reader = open_spectrum_reader(input_path)
    record_slice = reader.record_slice(config.t_start, config.t_stop)
    records = int(record_slice.stop - record_slice.start)
    periods = period_grid_records(cwt_config.period_min_records, cwt_config.period_max_records, cwt_config.period_count,
                                  cwt_config.period_spacing)
    payload = load_injection_config(config.injection_config)
    specs = make_injections_from_config(payload, records=records, channels=reader.n_channels,
                                        freqs_mhz=reader.freqs_mhz, default_seed=cwt_config.validation_random_seed)
    if config.max_injections > 0:
        specs = specs[: int(config.max_injections)]
    _assert_supported_injection_specs(specs)
    wavelets = resolve_wavelets(config.wavelets, max_wavelets=config.max_wavelets)
    truths = [_truth_for_spec(spec, reader, int(record_slice.start)) for spec in specs]
    metadata = {
        "schema_version": 1,
        "input": str(input_path),
        "selected_record_start": int(record_slice.start),
        "selected_record_stop": int(record_slice.stop),
        "records": records,
        "channels": reader.n_channels,
        "injection_config": str(config.injection_config),
        "injection_set_count": len(payload.get("sets", [])),
        "injection_spec_count": len(specs),
        "wavelets": wavelets,
        "config": cwt_config_to_nested_dict(cwt_config),
        "production_functions": [
            "cwipss.signal.cwt.cwt_power_cube",
            "cwipss.signal.activity.crop_valid_periods",
            "cwipss.signal.cpro.difference_noise_std",
            "cwipss.signal.cpro.impulse_cwt_noise_gain",
            "cwipss.signal.cpro.cpro_activity",
            "cwipss.signal.activity.robust_standardize",
        ],
    }
    (output_dir / "config.resolved.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True))
    truth_path = output_dir / "injection_truth.csv"
    _write_csv(truth_path, sorted({key for row in truths for key in row}), truths)
    main_index = [
        "# Wavelet Basis Injection Diagnostics",
        "",
        "Each linked directory contains one four-panel local diagnostic image. CWT is computed only on a compact window around the injected signal.",
        "",
    ]
    summary_rows: list[dict[str, Any]] = []
    total = len(wavelets) * len(specs)
    rendered = 0
    started = perf_counter()
    for wavelet in wavelets:
        wavelet_slug = _slug(wavelet)
        wavelet_items: list[tuple[str, Path, str]] = []
        main_index.append(f"- {wavelet}")
        for spec, truth in zip(specs, truths, strict=True):
            channel_idx = min(max(int(round(float(spec.channel_center))), 0), reader.n_channels - 1)
            block = reader.read_block(record_slice, slice(channel_idx, channel_idx + 1))
            baseline_full, injected_full = _inject_single_channel(block.data[:, 0], spec)
            local_start_abs, local_stop_abs = _local_record_window(
                truth,
                int(record_slice.start),
                int(record_slice.stop),
            )
            local_start = local_start_abs - int(record_slice.start)
            local_stop = local_stop_abs - int(record_slice.start)
            baseline = baseline_full[local_start:local_stop]
            injected = injected_full[local_start:local_stop]
            local_records = int(local_stop - local_start)
            case_id = f"{_slug(spec.injection_id)}__ch{channel_idx:04d}__{wavelet_slug}"
            cwt_start = perf_counter()
            power = cwt_power_cube(injected[:, None], periods, wavelet=wavelet, normalize_channels=False,
                                   method=cwt_config.cwt_method,
                                   backend=cwt_config.cwt_backend, cuda_device=cwt_config.cuda_device)
            cwt_seconds = perf_counter() - cwt_start
            power_channel = np.asarray(power[:, :, 0], dtype=np.float32)
            if power_channel.shape != (periods.size, local_records):
                raise AssertionError(f"CWT shape mismatch for {wavelet} {spec.injection_id}: {power_channel.shape}")
            if not np.all(np.isfinite(power_channel)):
                raise AssertionError(f"CWT produced non-finite values for {wavelet} {spec.injection_id}")
            compress_start = perf_counter()
            compressed = _compress_with_pipeline_functions(
                power_channel,
                periods,
                cwt_config,
                baseline=baseline,
                wavelet=wavelet,
            )
            compress_seconds = perf_counter() - compress_start
            if compressed["structured"].shape[1] != local_records or compressed["activity_raw"].shape != (
                    local_records,):
                raise AssertionError(f"Compression shape mismatch for {wavelet} {spec.injection_id}")
            if not np.all(np.isfinite(compressed["activity_raw"])) or not np.all(np.isfinite(compressed["activity_z"])):
                raise AssertionError(
                    f"Activity compression produced non-finite values for {wavelet} {spec.injection_id}")
            raw_tf, raw_tf_freqs, raw_ch0, _raw_ch1 = _raw_time_frequency_panel_data(
                reader,
                record_slice,
                channel_idx,
                baseline,
                injected,
                local_start,
                local_stop,
            )
            panel_path = output_dir / f"{case_id}__local_cwt_diagnostic_panel.png"
            _combined_diagnostic_plot(
                panel_path,
                raw_tf,
                raw_tf_freqs,
                baseline,
                injected,
                power_channel,
                periods,
                compressed,
                truth,
                channel_idx,
                raw_ch0,
                local_start_abs,
                f"{wavelet}: {spec.injection_id} ch={channel_idx} local CWT diagnostic",
                config.dpi,
            )
            if not panel_path.exists():
                raise AssertionError(f"Missing diagnostic image: {panel_path}")
            wavelet_items.append(
                (
                    f"{spec.injection_id} ch={channel_idx} local diagnostic panel",
                    panel_path,
                    "Four-panel local diagnostic: raw time-frequency image with injection box, target-channel raw CWT power 2D, 2D map before 1D compression, and 1D compression. CWT is computed only on the cropped signal window.",
                )
            )
            row = _summary_row(wavelet, spec, truth, channel_idx, compressed, cwt_seconds, compress_seconds)
            row.update({"local_record_start": local_start_abs, "local_record_stop": local_stop_abs,
                        "local_records": local_records})
            summary_rows.append(row)
            rendered += 1
            if config.progress_every > 0 and (rendered % int(config.progress_every) == 0 or rendered == total):
                print(f"[wavelet basis] rendered {rendered}/{total} cases elapsed={perf_counter() - started:.1f}s",
                      flush=True)
            _write_csv(output_dir / "wavelet_summary.csv", list(summary_rows[0].keys()), summary_rows)
        # Images are intentionally written directly into output_dir for fast scrolling; no per-wavelet subfolders.
        for item_title, image_path, note in wavelet_items:
            main_index += ["", f"## {item_title}", "", note, "", f"![{item_title}]({image_path.name})", ""]
    index_path = output_dir / "index.md"
    index_path.write_text("\n".join(main_index + [""]))
    return WaveletBasisRunResult(
        output_dir=output_dir,
        input_path=input_path,
        index_path=index_path,
        summary_path=output_dir / "wavelet_summary.csv",
        truth_path=truth_path,
        wavelets=wavelets,
        injection_count=len(specs),
        case_count=total,
    )
