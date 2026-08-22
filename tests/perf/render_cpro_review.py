#!/usr/bin/env python3
"""Render standalone CPRO cases for scientific visual review."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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

from compression_config_rank import largest_complete_2c  # noqa: E402
from cwt_activity_rank import CWTActivityRun, activity_config_from_cwt, prepare_activity_component  # noqa: E402
from cwipss.analysis.injection_config import load_injection_config, make_injections_from_config  # noqa: E402
from cwipss.data.readers import open_spectrum_reader  # noqa: E402
from cwipss.signal.activity import crop_valid_periods  # noqa: E402
from cwipss.signal.cwt import cwt_power_cube, period_grid_records  # noqa: E402
from persistent_occupancy import (  # noqa: E402
    _period_ridge_contrast,
    difference_noise_std,
    impulse_cwt_noise_gain,
    persistent_occupancy_catalog,
    persistent_occupancy_windows,
)


RUN_DIR = PROJECT_DIR / "runs/compression_final/cpro_full_validation"
OUTPUT_DIR = RUN_DIR / "visualizations"
ALGORITHM = "cpro_e32_q938_r150_o65_b3_w385_v40_d096"

INJECTION_CASES = (
    ("normal-hit", "inj_0002_weak_family_a_lowfreq_002_b001_c01", "Normal hit"),
    ("short-ultraweak-hit", "inj_0108_weak_family_c_ultraweak_lowfreq_008_b001_c01", "Shortest ultraweak hit"),
    ("long-period-hit", "inj_0024_weak_family_a_lowfreq_024_b001_c01", "Longest-period hit"),
    ("multifreq-copy-a", "inj_0037_weak_family_b_multifreq_lowfreq_002_b001_c01", "Multifrequency group copy A"),
    ("multifreq-copy-b", "inj_0038_weak_family_b_multifreq_lowfreq_002_b001_c02", "Multifrequency group copy B"),
    ("long-period-miss", "inj_0004_weak_family_a_lowfreq_004_b001_c01", "Period-900 miss"),
    ("short-ultraweak-miss", "inj_0126_weak_family_c_ultraweak_lowfreq_026_b001_c01", "Short ultraweak miss"),
)

NEGATIVE_CASES = (
    ("worst-real-negative", 3, "Worst real no-injection channel"),
    ("clean-real-negative", 14, "Zero-window real no-injection channel"),
)


def _downsample(values: np.ndarray, maximum: int = 2200) -> tuple[np.ndarray, np.ndarray]:
    records = int(values.shape[-1])
    if records <= maximum:
        indices = np.arange(records, dtype=np.int64)
    else:
        indices = np.linspace(0, records - 1, maximum).round().astype(np.int64)
    return np.asarray(values)[..., indices], indices


def _finite_limits(values: np.ndarray, low: float, high: float) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.quantile(finite, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _draw_map(
    fig: Any,
    ax: Any,
    values: np.ndarray,
    *,
    records: int,
    periods: np.ndarray,
    title: str,
    cmap: str,
    limits: tuple[float, float] | None = None,
) -> None:
    shown, indices = _downsample(values)
    lo, hi = limits or _finite_limits(shown, 0.01, 0.995)
    image = ax.imshow(
        shown,
        origin="lower",
        aspect="auto",
        extent=(float(indices[0]), float(indices[-1]), float(periods[0]), float(periods[-1])),
        cmap=cmap,
        vmin=lo,
        vmax=hi,
        interpolation="nearest",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Period (records)")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.016, pad=0.008)
    ax.set_xlim(0, records - 1)


def _render(
    *,
    key: str,
    reason: str,
    raw: np.ndarray,
    raw_freqs: np.ndarray,
    target_frequency: float,
    power: np.ndarray,
    periods: np.ndarray,
    calibrated: np.ndarray,
    contrast: np.ndarray,
    result: Any,
    output_dir: Path,
    truth: tuple[int, int, float] | None,
) -> Path:
    records = int(power.shape[1])
    raw_shown, raw_indices = _downsample(raw.T)
    raw_lo, raw_hi = _finite_limits(raw_shown, 0.01, 0.99)
    positive = power[power > 0.0]
    eps = max(float(np.median(positive)) * 1e-8, np.finfo(np.float32).tiny) if positive.size else 1e-30
    log_power = np.log10(np.maximum(power, eps))
    log_calibrated = np.log10(np.maximum(calibrated, np.finfo(np.float32).tiny))
    accepted = np.where(result.absolute_score_map > 0.0, log_power, np.nan)

    fig = plt.figure(figsize=(16, 17), constrained_layout=True)
    grid = fig.add_gridspec(6, 1, height_ratios=(1.0, 1.0, 1.0, 1.0, 1.0, 0.78))
    ax_raw = fig.add_subplot(grid[0])
    ax_cwt = fig.add_subplot(grid[1], sharex=ax_raw)
    ax_calibrated = fig.add_subplot(grid[2], sharex=ax_raw)
    ax_contrast = fig.add_subplot(grid[3], sharex=ax_raw)
    ax_accepted = fig.add_subplot(grid[4], sharex=ax_raw)
    ax_activity = fig.add_subplot(grid[5], sharex=ax_raw)

    raw_image = ax_raw.imshow(
        raw_shown,
        origin="lower",
        aspect="auto",
        extent=(
            float(raw_indices[0]),
            float(raw_indices[-1]),
            float(raw_freqs[0]),
            float(raw_freqs[-1]),
        ),
        cmap="cividis",
        vmin=raw_lo,
        vmax=raw_hi,
        interpolation="nearest",
    )
    ax_raw.axhline(target_frequency, color="#ef476f", linewidth=1.0)
    ax_raw.set_ylabel("Frequency (MHz)")
    ax_raw.set_title("Raw CE4 time-frequency neighborhood; target channel marked")
    ax_raw.set_xlim(0, records - 1)
    fig.colorbar(raw_image, ax=ax_raw, fraction=0.016, pad=0.008, label="Amplitude")

    _draw_map(
        fig,
        ax_cwt,
        log_power,
        records=records,
        periods=periods,
        title="Absolute target-channel CWT power (log10)",
        cmap="magma",
    )
    _draw_map(
        fig,
        ax_calibrated,
        log_calibrated,
        records=records,
        periods=periods,
        title="Calibrated power Z (log10); absolute floor=32, texture quantile=0.9375",
        cmap="viridis",
    )
    _draw_map(
        fig,
        ax_contrast,
        np.clip(contrast, 0.0, 4.0),
        records=records,
        periods=periods,
        title="Local period-ridge contrast R; acceptance requires R >= 1.5",
        cmap="cividis",
        limits=(0.0, 4.0),
    )
    _draw_map(
        fig,
        ax_accepted,
        accepted,
        records=records,
        periods=periods,
        title="CPRO accepted same-period persistent ridge in absolute CWT units (log10)",
        cmap="inferno",
    )

    activity = np.asarray(result.absolute_activity, dtype=np.float64)
    activity_shown, activity_indices = _downsample(activity)
    ax_activity.plot(activity_indices, activity_shown, color="#ef476f", linewidth=1.0, label="CPRO activity")
    for index, window in enumerate(result.windows):
        ax_activity.axvspan(
            int(window["record_start"]),
            int(window["record_stop"]),
            color="#ffd166",
            alpha=0.22,
            label="CPRO window" if index == 0 else None,
        )
    ax_activity.set_ylabel("Absolute CWT power")
    ax_activity.set_xlabel("Local record")
    ax_activity.set_title(f"1D CPRO activity; windows={len(result.windows)}")
    ax_activity.grid(alpha=0.18)
    ax_activity.set_xlim(0, records - 1)

    status = "REAL NEGATIVE"
    if truth is not None:
        start, stop, true_period = truth
        status = "HIT" if any(
            int(window["record_start"]) < stop and int(window["record_stop"]) > start
            for window in result.windows
        ) else "MISS"
        for ax in (ax_raw, ax_cwt, ax_calibrated, ax_contrast, ax_accepted, ax_activity):
            ax.axvspan(start, stop, color="#06d6a0", alpha=0.12)
        for ax in (ax_cwt, ax_calibrated, ax_contrast, ax_accepted):
            ax.axhline(true_period, color="#f6bd60", linewidth=1.0, linestyle="--")
        ax_activity.legend(loc="upper right")
        fig.suptitle(
            f"{reason} | {status} | period={true_period:.3f} | truth={start}:{stop}",
            fontsize=13,
        )
    else:
        if result.windows:
            ax_activity.legend(loc="upper right")
        fig.suptitle(f"{reason} | {status} | frequency={target_frequency:.6f} MHz", fontsize=13)

    path = output_dir / f"{key}.png"
    fig.savefig(path, dpi=145)
    plt.close(fig)
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--input", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    input_path = args.input or largest_complete_2c(PROJECT_DIR / "data/CE4")
    reader = open_spectrum_reader(input_path)
    run = CWTActivityRun(
        output_dir=args.output,
        input_path=input_path,
        injection_config=RUN_DIR / "injection_config.json",
        cwt_config=RUN_DIR / "cwt_config.json",
        cwt_backend="cpu",
        candidate_period_max_records=1000.0,
    )
    config = activity_config_from_cwt(run)
    periods = period_grid_records(
        config.period_min_records,
        config.period_max_records,
        config.period_count,
        config.period_spacing,
    )
    specs = make_injections_from_config(
        load_injection_config(run.injection_config),
        records=reader.n_records,
        channels=reader.n_channels,
        freqs_mhz=reader.freqs_mhz,
    )
    specs_by_id = {spec.injection_id: spec for spec in specs}
    params = next(item for item in persistent_occupancy_catalog() if item.name == ALGORITHM)
    manifest: list[dict[str, Any]] = []
    noise_gain: np.ndarray | None = None

    for key, case_id, reason in INJECTION_CASES:
        spec = specs_by_id[case_id]
        prepared, baseline, injected = prepare_activity_component(
            reader=reader,
            spec=spec,
            config=config,
            periods=periods,
            input_denoisers=("absolute",),
        )
        if noise_gain is None:
            noise_gain = impulse_cwt_noise_gain(prepared.valid_periods, wavelet=config.wavelet)
        power = prepared.cwt_power["absolute"]
        noise_std = difference_noise_std(baseline)
        calibrated = power / (noise_std**2 * noise_gain[:, None])
        contrast = _period_ridge_contrast(
            calibrated,
            center_bins=params.period_center_bins,
            context_bins=params.period_context_bins,
        )
        result = persistent_occupancy_windows(
            power,
            noise_std=noise_std,
            noise_gain=noise_gain,
            params=params,
        )
        channel = int(round(float(spec.channel_center)))
        channel_start = max(0, channel - 12)
        channel_stop = min(reader.n_channels, channel + 13)
        raw_block = reader.read_block(
            slice(prepared.background_record_start, prepared.background_record_stop),
            slice(channel_start, channel_stop),
        )
        raw = np.asarray(raw_block.data, dtype=np.float32).copy()
        raw[:, channel - channel_start] += np.asarray(injected - baseline, dtype=np.float32)
        truth = (
            int(prepared.truth["record_start"]),
            int(prepared.truth["record_stop"]),
            float(spec.period_records),
        )
        path = _render(
            key=key,
            reason=f"{reason} | {case_id}",
            raw=raw,
            raw_freqs=np.asarray(raw_block.freqs_mhz),
            target_frequency=float(reader.freqs_mhz[channel]),
            power=power,
            periods=prepared.valid_periods,
            calibrated=calibrated,
            contrast=contrast,
            result=result,
            output_dir=args.output,
            truth=truth,
        )
        manifest.append(
            {
                "key": key,
                "kind": "injection",
                "case_id": case_id,
                "reason": reason,
                "image": path.name,
                "window_count": len(result.windows),
                "truth_hit": int(any(
                    int(window["record_start"]) < truth[1]
                    and int(window["record_stop"]) > truth[0]
                    for window in result.windows
                )),
            }
        )
        print(f"[cpro-visual] {path.name}", flush=True)

    for key, channel, reason in NEGATIVE_CASES:
        series = np.asarray(
            reader.read_block(slice(0, reader.n_records), slice(channel, channel + 1)).data[:, 0],
            dtype=np.float32,
        )
        cube = cwt_power_cube(
            series[:, None],
            periods,
            wavelet=config.wavelet,
            normalize_channels=False,
            method=config.cwt_method,
            backend="cpu",
        )
        valid_power, valid_periods, _mask = crop_valid_periods(
            cube,
            periods,
            config.candidate_period_min_records,
            config.candidate_period_max_records,
        )
        if noise_gain is None:
            noise_gain = impulse_cwt_noise_gain(valid_periods, wavelet=config.wavelet)
        power = valid_power[:, :, 0]
        noise_std = difference_noise_std(series)
        calibrated = power / (noise_std**2 * noise_gain[:, None])
        contrast = _period_ridge_contrast(
            calibrated,
            center_bins=params.period_center_bins,
            context_bins=params.period_context_bins,
        )
        result = persistent_occupancy_windows(
            power,
            noise_std=noise_std,
            noise_gain=noise_gain,
            params=params,
        )
        channel_start = max(0, channel - 12)
        channel_stop = min(reader.n_channels, channel + 13)
        raw_block = reader.read_block(slice(0, reader.n_records), slice(channel_start, channel_stop))
        path = _render(
            key=key,
            reason=reason,
            raw=np.asarray(raw_block.data, dtype=np.float32),
            raw_freqs=np.asarray(raw_block.freqs_mhz),
            target_frequency=float(reader.freqs_mhz[channel]),
            power=power,
            periods=valid_periods,
            calibrated=calibrated,
            contrast=contrast,
            result=result,
            output_dir=args.output,
            truth=None,
        )
        manifest.append(
            {
                "key": key,
                "kind": "negative",
                "case_id": f"channel_{channel:03d}",
                "reason": reason,
                "image": path.name,
                "window_count": len(result.windows),
                "truth_hit": "",
            }
        )
        print(f"[cpro-visual] {path.name}", flush=True)

    with (args.output / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
