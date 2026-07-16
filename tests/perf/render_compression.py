"""Render standalone Top-10 compression figures for supported benchmark runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

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

from compression_benchmark import (  # noqa: E402
    DEFAULT_TOP10_ALGORITHMS,
    CompressionBenchmarkConfig,
    _PreparedCase,
    _best_band_timeseries,
    _compress_structured,
    _evaluate_algorithm,
    _resolve_algorithms,
    compression_algorithm_map,
)
from compression_config_rank import (  # noqa: E402
    ConfiguredCompressionRun,
    compression_config_from_cwt,
    prepare_configured_component,
)
from cwipss.analysis.injection_config import (  # noqa: E402
    load_injection_config,
    make_injections_from_config,
)
from cwipss.analysis.injection import synthetic_background  # noqa: E402
from cwipss.analysis.simulation import InjectionSpec, inject_periodic_signal  # noqa: E402
from cwipss.data.readers import open_spectrum_reader  # noqa: E402
from cwipss.signal.activity import (  # noqa: E402
    coherent_structure_map,
    crop_valid_periods,
    low_fraction_noise_floor,
    relative_excess,
    robust_standardize,
)
from cwipss.signal.cwt import cwt_power_cube, period_grid_records  # noqa: E402
from cwipss.signal.windows import (  # noqa: E402
    active_windows_from_segments,
    merge_close_windows,
    pelt_mean_shift,
)


OUTPUT_DIR = PROJECT_DIR / "runs" / "compression_final"
TOP10_DIR = OUTPUT_DIR / "top10"
FINALISTS_DIR = PROJECT_DIR / "runs" / "pytest_compression_perf_finalists_compare_400cases_2seeds_v1"
WINDOWLOCAL_DIR = PROJECT_DIR / "runs" / "pytest_compression_perf_windowlocal_tuned_compare_400cases_2seeds_v1"
ALL44_DIR = PROJECT_DIR / "runs" / "pytest_compression_perf_background_compare_all44_120cases_v2"


TOP10 = (
    ("max_ratio_s1", "Overall default"),
    ("max_pool_s1", "Speed baseline"),
    ("band_contrast_w1_s1", "Objective 1 specialist"),
    ("band_zscore_w1_s1", "Objective 2 specialist"),
    ("band_local_contrast_w1_c1_s1", "Objective 3 raw-value specialist"),
    ("band_local_ratio_w1_s1", "High time-contrast finalist"),
    ("band_hybrid_w1_s1", "Hybrid finalist"),
    ("band_max_w5_s1", "Contiguous-band baseline"),
    ("band_ratio_w3_s1", "Band-energy ratio baseline"),
    ("topk_ratio_k3_s1", "Top-k concentration baseline"),
)

if tuple(name for name, _role in TOP10) != DEFAULT_TOP10_ALGORITHMS:
    raise RuntimeError("Synthetic visualization roles must match DEFAULT_TOP10_ALGORITHMS")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def _copy_csv_without_columns(source: Path, target: Path, excluded: set[str]) -> None:
    rows = _read_csv(source)
    if not rows:
        raise ValueError(f"Compression evidence is empty: {source}")
    fieldnames = [field for field in rows[0] if field not in excluded]
    with target.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in fieldnames}
            for row in rows
        )


def _copy_final_evidence() -> None:
    mappings = {
        FINALISTS_DIR / "metric_consensus.csv": OUTPUT_DIR / "finalist_metric_consensus.csv",
        WINDOWLOCAL_DIR / "metric_consensus.csv": OUTPUT_DIR / "windowlocal_metric_consensus.csv",
        WINDOWLOCAL_DIR / "suite_windowlocal_leaders.csv": OUTPUT_DIR / "windowlocal_suite_leaders.csv",
        ALL44_DIR / "background_consensus.csv": OUTPUT_DIR / "all44_background_consensus.csv",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for source, target in mappings.items():
        if source.exists():
            shutil.copy2(source, target)
        elif not target.exists():
            raise FileNotFoundError(f"Missing compression evidence: {source}")

    leaders_source = FINALISTS_DIR / "suite_metric_leaders.csv"
    leaders_target = OUTPUT_DIR / "finalist_suite_metric_leaders.csv"
    if leaders_source.exists():
        _copy_csv_without_columns(leaders_source, leaders_target, {"suite_path"})
    elif not leaders_target.exists():
        raise FileNotFoundError(f"Missing compression evidence: {leaders_source}")


def _prepare_representative_case() -> tuple[_PreparedCase, np.ndarray, np.ndarray, CompressionBenchmarkConfig]:
    records = 1024
    channels = 128
    noise_std = 1.0
    background = synthetic_background(
        records=records,
        channels=channels,
        noise_std=noise_std,
        seed=76,
        f_start_mhz=0.1,
        f_stop_mhz=40.0,
    )
    spec = InjectionSpec(
        injection_id="compression_visualization",
        signal_model="band_limited_periodic",
        period_records=64.0,
        amplitude=0.65,
        record_start=264,
        duration_records=697,
        channel_center=64.0,
        bandwidth_channels=5.0,
        duty_cycle=0.12,
        phase=0.18,
    )
    injected, truth = inject_periodic_signal(background.data, spec)
    config = CompressionBenchmarkConfig(
        output_dir=OUTPUT_DIR,
        records_min=records,
        records_max=records,
        channels_min=channels,
        channels_max=channels,
        background_modes=("synthetic",),
        period_min_records=2.0,
        period_max_records=256.0,
        period_count=96,
        candidate_period_min_records=10.0,
        candidate_period_max_records=160.0,
        structure_time_support_records=48,
        pelt_penalty=16.0,
        pelt_min_size_records=128,
        window_min_duration_records=128,
        window_min_activity_mean=0.0,
        window_merge_gap_records=48,
        progress_every=0,
    )
    periods = period_grid_records(
        config.period_min_records,
        config.period_max_records,
        config.period_count,
        config.period_spacing,
    )
    channel_index = int(round(spec.channel_center))
    preprocess_start = perf_counter()
    power = cwt_power_cube(
        injected[:, channel_index : channel_index + 1],
        wavelet=config.wavelet,
        periods=periods,
        method=config.cwt_method,
        backend=config.cwt_backend,
        cuda_device=config.cuda_device,
        normalize_channels=True,
    )[:, :, 0]
    valid_power, valid_periods, _mask = crop_valid_periods(
        power,
        periods,
        config.candidate_period_min_records,
        config.candidate_period_max_records,
    )
    noise_floor = low_fraction_noise_floor(valid_power, fraction=config.noise_floor_fraction)
    excess = relative_excess(valid_power, noise_floor, eps_fraction=config.excess_eps_fraction)
    structured = coherent_structure_map(
        excess,
        baseline_quantile=config.structure_baseline_quantile,
        scale_quantile=config.structure_scale_quantile,
        z_threshold=config.structure_z_threshold,
        time_support_records=config.structure_time_support_records,
        period_support_bins=config.structure_period_support_bins,
        min_support_fraction=config.structure_min_support_fraction,
    )
    preprocess_seconds = perf_counter() - preprocess_start

    best_contrast = np.full(records, -np.inf, dtype=np.float32)
    best_ratio = np.zeros(records, dtype=np.float32)
    best_periods = np.full(records, np.nan, dtype=np.float32)
    for width in config.band_widths:
        contrast, ratio, period_values = _best_band_timeseries(structured, valid_periods, width)
        update = contrast > best_contrast
        best_contrast = np.where(update, contrast, best_contrast)
        best_ratio = np.where(update, ratio, best_ratio)
        best_periods = np.where(update, period_values, best_periods)

    truth = dict(truth)
    truth["channel_index"] = channel_index
    prepared = _PreparedCase(
        case_id="representative_case",
        records=records,
        channels=channels,
        noise_std=noise_std,
        background_mode="synthetic",
        background_source="synthetic",
        background_record_start=0,
        background_record_stop=records,
        background_channel_start=0,
        background_channel_stop=channels,
        preprocess_seconds=preprocess_seconds,
        injection=spec,
        truth=truth,
        valid_periods=valid_periods,
        structured=structured,
        time_band_contrast=best_contrast.astype(np.float32, copy=False),
        time_band_ratio=best_ratio.astype(np.float32, copy=False),
        time_band_periods=best_periods.astype(np.float32, copy=False),
    )
    return prepared, injected, background.freqs_mhz, config


def _candidate_windows(activity_z: np.ndarray, config: CompressionBenchmarkConfig) -> list[dict[str, float | int]]:
    segments = pelt_mean_shift(
        np.asarray(activity_z, dtype=np.float64),
        penalty=config.pelt_penalty,
        min_size=config.pelt_min_size_records,
        jump=config.pelt_jump_records,
    )
    return merge_close_windows(
        active_windows_from_segments(
            segments,
            activity_z,
            min_duration=config.window_min_duration_records,
            min_mean=config.window_min_activity_mean,
        ),
        max_gap=config.window_merge_gap_records,
    )


def _normalizations(rows: list[dict[str, object]]) -> dict[str, tuple[float, float]]:
    fields = (
        "peak_global_band_contrast",
        "peak_period_concentration",
        "truth_window_local_band_contrast",
    )
    result: dict[str, tuple[float, float]] = {}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        result[field] = (float(np.nanmin(values)), float(np.nanmax(values)))
    return result


def _normalized(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    if not np.isfinite(value) or math.isclose(lo, hi):
        return 0.5
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _render_candidate(
    *,
    rank: int,
    role: str,
    prepared: _PreparedCase,
    raw_data: np.ndarray,
    freqs_mhz: np.ndarray,
    config: CompressionBenchmarkConfig,
    algorithm: object,
    row: dict[str, object],
    metric_bounds: dict[str, tuple[float, float]],
    top10_dir: Path = TOP10_DIR,
    title_context: str = "",
) -> Path:
    activity_z = robust_standardize(_compress_structured(prepared.structured, algorithm))
    windows = _candidate_windows(activity_z, config)
    truth_start = int(prepared.truth["record_start"])
    truth_stop = int(prepared.truth["record_stop"])
    channel_start = int(prepared.truth["channel_start"])
    channel_stop = int(prepared.truth["channel_stop"])
    truth_period = float(prepared.truth["period_records"])

    raw_lo, raw_hi = np.nanquantile(raw_data, [0.01, 0.99])
    metrics = (
        ("Global band contrast", "peak_global_band_contrast", float(row["peak_global_band_contrast"])),
        ("Period concentration", "peak_period_concentration", float(row["peak_period_concentration"])),
        ("Window local contrast", "truth_window_local_band_contrast", float(row["truth_window_local_band_contrast"])),
    )
    scores = [_normalized(value, metric_bounds[field]) for _label, field, value in metrics]

    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    grid = fig.add_gridspec(4, 1, height_ratios=(1.15, 1.35, 0.85, 0.72))
    ax_raw = fig.add_subplot(grid[0])
    ax_cwt = fig.add_subplot(grid[1], sharex=ax_raw)
    ax_activity = fig.add_subplot(grid[2], sharex=ax_raw)
    ax_metrics = fig.add_subplot(grid[3])

    raw_image = ax_raw.imshow(
        raw_data.T,
        aspect="auto",
        origin="lower",
        extent=[0, prepared.records, float(freqs_mhz[0]), float(freqs_mhz[-1])],
        cmap="cividis",
        vmin=raw_lo,
        vmax=raw_hi,
        interpolation="nearest",
    )
    ax_raw.axvspan(truth_start, truth_stop, color="#f6bd60", alpha=0.16)
    ax_raw.axhspan(
        float(freqs_mhz[channel_start]),
        float(freqs_mhz[min(channel_stop - 1, freqs_mhz.size - 1)]),
        color="#ef476f",
        alpha=0.22,
    )
    ax_raw.set_ylabel("Frequency (MHz)")
    ax_raw.set_title("Raw time-frequency data")
    fig.colorbar(raw_image, ax=ax_raw, label="Amplitude", fraction=0.025)

    cwt_image = ax_cwt.pcolormesh(
        np.arange(prepared.records, dtype=np.float64),
        prepared.valid_periods,
        prepared.structured,
        cmap="magma",
        shading="auto",
    )
    ax_cwt.axvspan(truth_start, truth_stop, color="#50c878", alpha=0.13)
    ax_cwt.axhline(truth_period, color="#7ce0a3", linestyle="--", linewidth=1.5, label="Injected period")
    ax_cwt.set_ylim(float(prepared.valid_periods[-1]), float(prepared.valid_periods[0]))
    ax_cwt.set_ylabel("Period (records)")
    ax_cwt.set_title("Structured CWT period-time map")
    ax_cwt.legend(loc="upper right")
    fig.colorbar(cwt_image, ax=ax_cwt, label="Structured excess", fraction=0.025)

    ax_activity.plot(activity_z, color="#245f73", linewidth=1.1)
    ax_activity.axvspan(truth_start, truth_stop, color="#50c878", alpha=0.13, label="Injected interval")
    for index, window in enumerate(windows):
        ax_activity.axvspan(
            int(window["record_start"]),
            int(window["record_stop"]),
            color="#e76f51",
            alpha=0.12,
            label="PELT candidate window" if index == 0 else None,
        )
    peak_record = int(row["peak_record"])
    ax_activity.axvline(peak_record, color="#d1603d", linestyle=":", linewidth=1.2, label="Activity peak")
    ax_activity.axhline(0.0, color="#333333", linewidth=0.6, alpha=0.6)
    ax_activity.set_ylabel("Robust z")
    ax_activity.set_xlabel("Record")
    ax_activity.set_title("Compressed 1D activity")
    ax_activity.grid(alpha=0.2)
    ax_activity.legend(loc="upper right", ncol=3, fontsize=8)

    labels = [label for label, _field, _value in metrics]
    bars = ax_metrics.barh(labels, scores, color=("#d1603d", "#d19b20", "#4f7c58"))
    ax_metrics.set_xlim(0.0, 1.08)
    ax_metrics.set_xlabel("Normalized score within Top 10")
    ax_metrics.set_title("Three scientific performance parameters")
    ax_metrics.grid(axis="x", alpha=0.2)
    for bar, (_label, _field, value), score in zip(bars, metrics, scores, strict=True):
        raw_text = f"{value:.3g}"
        ax_metrics.text(min(score + 0.02, 1.01), bar.get_y() + bar.get_height() / 2, raw_text, va="center", fontsize=9)

    context = f"{title_context}\n" if title_context else ""
    fig.suptitle(
        f"{context}Top {rank:02d} | {algorithm.name}\n{role} | {algorithm.description}",
        fontsize=16,
        weight="bold",
    )
    output = top10_dir / f"{rank:02d}_{algorithm.name}.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _write_outputs(rows: list[dict[str, object]], image_paths: list[Path], metadata: dict[str, object]) -> None:
    fieldnames = [
        "rank",
        "algorithm",
        "role",
        "algorithm_family",
        "algorithm_description",
        "peak_global_band_contrast",
        "peak_period_concentration",
        "truth_window_local_band_contrast",
        "time_contrast_z",
        "truth_window_hit",
        "algorithm_seconds",
        "image",
    ]
    with (OUTPUT_DIR / "top10_metrics.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)

    summary = {
        "top10": rows,
        "representative_case": metadata,
        "images": [str(path.relative_to(OUTPUT_DIR)) for path in image_paths],
        "metric_definitions": {
            "peak_global_band_contrast": "At the 1D activity peak, strongest local period band minus the full-period background.",
            "peak_period_concentration": "At the 1D activity peak, fraction of period-axis energy in the selected contiguous band.",
            "truth_window_local_band_contrast": "Inside the PELT window overlapping the injection, candidate period band minus nearby period background.",
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Final CWT 2D-to-1D Compression Top 10",
        "",
        "Each candidate is rendered separately using the same weak injected signal and the same raw/CWT input.",
        "",
        "Each PNG contains:",
        "",
        "- Raw time-frequency data",
        "- Structured CWT period-time map",
        "- Candidate 1D compressed activity and PELT windows",
        "- Three normalized scientific metrics with raw values",
        "",
        "## Candidates",
        "",
    ]
    for row, image_path in zip(rows, image_paths, strict=True):
        lines.append(
            f"{row['rank']}. `{row['algorithm']}`: {row['role']} "
            f"([image]({image_path.relative_to(OUTPUT_DIR)}))"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `top10_metrics.csv`: raw metrics for all ten candidates.",
            "- `summary.json`: machine-readable metadata and metric definitions.",
            "- `all44_background_consensus.csv`: retained initial 44-algorithm screening.",
            "- `finalist_metric_consensus.csv`: retained finalist consensus.",
            "- `windowlocal_metric_consensus.csv`: retained Objective 3 tuned consensus.",
            "- `compression_algorithm_map.json`: all supported compression configurations; Top 10 are marked selected.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python tests/perf/render_compression.py synthetic",
            "```",
        ]
    )
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n")


def render_synthetic_final() -> None:
    _copy_final_evidence()
    if TOP10_DIR.exists():
        shutil.rmtree(TOP10_DIR)
    TOP10_DIR.mkdir(parents=True)
    for obsolete in (
        OUTPUT_DIR / "compression_performance_dashboard.png",
        OUTPUT_DIR / "compression_2d_to_1d_example.png",
        OUTPUT_DIR / "representative_case.json",
    ):
        obsolete.unlink(missing_ok=True)

    prepared, raw_data, freqs_mhz, config = _prepare_representative_case()
    algorithms = {algorithm.name: algorithm for algorithm in _resolve_algorithms(tuple(name for name, _role in TOP10))}
    evaluated: list[dict[str, object]] = []
    for rank, (name, role) in enumerate(TOP10, start=1):
        algorithm = algorithms[name]
        row = _evaluate_algorithm(prepared, algorithm, config)
        row.update({"rank": rank, "role": role})
        evaluated.append(row)

    metric_bounds = _normalizations(evaluated)
    images: list[Path] = []
    output_rows: list[dict[str, object]] = []
    for row in evaluated:
        algorithm = algorithms[str(row["algorithm"])]
        image = _render_candidate(
            rank=int(row["rank"]),
            role=str(row["role"]),
            prepared=prepared,
            raw_data=raw_data,
            freqs_mhz=freqs_mhz,
            config=config,
            algorithm=algorithm,
            row=row,
            metric_bounds=metric_bounds,
        )
        row["image"] = str(image.relative_to(OUTPUT_DIR))
        output_rows.append(row)
        images.append(image)

    metadata = {
        "background": "synthetic",
        "background_seed": 76,
        "records": prepared.records,
        "channels": prepared.channels,
        "signal_model": prepared.injection.signal_model,
        "amplitude_factor": float(prepared.injection.amplitude / prepared.noise_std),
        "period_records": float(prepared.injection.period_records),
        "record_start": int(prepared.truth["record_start"]),
        "record_stop": int(prepared.truth["record_stop"]),
        "channel_start": int(prepared.truth["channel_start"]),
        "channel_stop": int(prepared.truth["channel_stop"]),
        "injection_spec": asdict(prepared.injection),
        "compression_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "reproduce_command": (
            "python tests/perf/render_compression.py synthetic"
        ),
    }
    algorithm_map = compression_algorithm_map()
    (OUTPUT_DIR / "compression_algorithm_map.json").write_text(
        json.dumps(
            {
                name: {
                    **configuration,
                    "selected": name in DEFAULT_TOP10_ALGORITHMS,
                }
                for name, configuration in algorithm_map.items()
            },
            indent=2,
        )
    )
    _write_outputs(output_rows, images, metadata)


CONFIGURED_ROLES = {
    "max_ratio_s1": "Max-ratio finalist",
    "band_local_ratio_w1_s1": "Local-ratio finalist",
    "band_hybrid_w1_s1": "Hybrid finalist",
    "topk_ratio_k3_s1": "Top-k concentration finalist",
    "band_ratio_w3_s1": "Band-ratio finalist",
    "band_contrast_w1_s1": "Global-band finalist",
    "max_pool_s1": "Speed baseline",
    "band_local_contrast_w1_c1_s1": "Local-context specialist",
    "band_max_w5_s1": "Contiguous-band finalist",
    "band_zscore_w1_s1": "Period-concentration specialist",
}


def _configured_role(
    name: str,
    rank: int,
    benchmark: dict[str, object],
) -> str:
    if rank == 1:
        return "Overall configured-rank winner"
    leaders = dict(benchmark.get("metric_leaders", {}))
    labels = (
        ("peak_in_truth_rate", "Peak-recovery leader"),
        ("truth_window_hit_rate", "Window-recovery leader"),
        ("mean_time_contrast_z", "Time-contrast leader"),
        ("mean_peak_global_band_contrast", "Global-band leader"),
        ("mean_peak_period_concentration", "Period-concentration leader"),
    )
    for field, label in labels:
        if leaders.get(field) == name:
            return label
    return CONFIGURED_ROLES.get(name, "Configured-rank candidate")


def _configured_component_rows(
    run_dir: Path,
    component_id: str,
) -> dict[str, dict[str, str]]:
    with (run_dir / "component_cases.csv").open(newline="") as fp:
        rows = [
            row
            for row in csv.DictReader(fp)
            if row["case_id"] == component_id
        ]
    if not rows:
        raise ValueError(f"No benchmark rows found for {component_id}")
    return {row["algorithm"]: row for row in rows}


def _configured_raw_time_frequency(
    reader: object,
    prepared: _PreparedCase,
    baseline: np.ndarray,
    injected: np.ndarray,
) -> tuple[_PreparedCase, np.ndarray, np.ndarray]:
    channel_index = int(prepared.background_channel_start)
    channel_start = max(0, channel_index - 24)
    channel_stop = min(int(reader.n_channels), channel_index + 25)
    block = reader.read_block(
        slice(prepared.background_record_start, prepared.background_record_stop),
        slice(channel_start, channel_stop),
    )
    raw = np.asarray(block.data, dtype=np.float32).copy()
    target = channel_index - channel_start
    raw[:, target] += np.asarray(injected - baseline, dtype=np.float32)
    truth = dict(prepared.truth)
    truth.update(
        {
            "channel_index": target,
            "channel_start": target,
            "channel_stop": target + 1,
        }
    )
    return (
        replace(prepared, channels=raw.shape[1], truth=truth),
        raw,
        np.asarray(block.freqs_mhz, dtype=np.float64),
    )


def render_configured_run(
    run_dir: Path,
    component_id: str,
    *,
    limit: int = 10,
) -> None:
    benchmark = json.loads((run_dir / "compression_summary.json").read_text())
    input_path = Path(str(benchmark["input_path"]))
    injection_config = Path(str(benchmark["injection_config"]))
    cwt_config = Path(str(benchmark["cwt_config"]))
    if not injection_config.exists():
        injection_config = run_dir / "injection_config.json"
    if not cwt_config.exists():
        cwt_config = run_dir / "cwt_config.json"
    ranked_names = tuple(
        str(row["algorithm"])
        for row in benchmark["summary_rows"][: max(1, int(limit))]
    )
    run = ConfiguredCompressionRun(
        output_dir=run_dir,
        input_path=input_path,
        injection_config=injection_config,
        cwt_config=cwt_config,
        algorithms=ranked_names,
        cwt_backend=str(benchmark["config"]["cwt_backend"]),
        cuda_device=int(benchmark["config"]["cuda_device"]),
        candidate_period_max_records=float(
            benchmark["config"]["candidate_period_max_records"]
        ),
        progress_every=0,
    )
    config = compression_config_from_cwt(run)
    reader = open_spectrum_reader(input_path)
    specs = make_injections_from_config(
        load_injection_config(injection_config),
        records=reader.n_records,
        channels=reader.n_channels,
        freqs_mhz=reader.freqs_mhz,
    )
    spec = next(
        candidate
        for candidate in specs
        if candidate.injection_id == component_id
    )
    periods = period_grid_records(
        config.period_min_records,
        config.period_max_records,
        config.period_count,
        config.period_spacing,
    )
    prepared, baseline, injected = prepare_configured_component(
        reader=reader,
        spec=spec,
        config=config,
        periods=periods,
    )
    render_prepared, raw_data, raw_freqs = _configured_raw_time_frequency(
        reader,
        prepared,
        baseline,
        injected,
    )

    output_dir = run_dir / "top10_visualizations"
    top10_dir = output_dir / "top10"
    if top10_dir.exists():
        shutil.rmtree(top10_dir)
    top10_dir.mkdir(parents=True)

    benchmark_rows = _configured_component_rows(run_dir, component_id)
    algorithms = {
        algorithm.name: algorithm
        for algorithm in _resolve_algorithms(ranked_names)
    }
    evaluated: list[dict[str, object]] = []
    for rank, name in enumerate(ranked_names, start=1):
        row = _evaluate_algorithm(prepared, algorithms[name], config)
        original = benchmark_rows[name]
        if int(row["peak_record"]) != int(original["peak_record"]):
            raise ValueError(f"Reproduced peak mismatch for {name}")
        row.update(
            {
                "rank": rank,
                "role": _configured_role(name, rank, benchmark),
            }
        )
        evaluated.append(row)

    bounds = _normalizations(evaluated)
    images: list[Path] = []
    for row in evaluated:
        algorithm = algorithms[str(row["algorithm"])]
        image = _render_candidate(
            rank=int(row["rank"]),
            role=str(row["role"]),
            prepared=render_prepared,
            raw_data=raw_data,
            freqs_mhz=raw_freqs,
            config=config,
            algorithm=algorithm,
            row=row,
            metric_bounds=bounds,
            top10_dir=top10_dir,
            title_context="Configured ultraweak CE4 injection",
        )
        row["image"] = str(image.relative_to(output_dir))
        images.append(image)

    fields = [
        "rank",
        "algorithm",
        "role",
        "peak_global_band_contrast",
        "peak_period_concentration",
        "truth_window_local_band_contrast",
        "time_contrast_z",
        "peak_in_truth",
        "truth_window_hit",
        "peak_band_period_error_fraction",
        "algorithm_seconds",
        "image",
    ]
    with (output_dir / "top10_metrics.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fields}
            for row in evaluated
        )

    metadata = {
        "representative_component_id": spec.injection_id,
        "source_config": str(injection_config),
        "background_source": input_path.name,
        "frequency_mhz": float(
            reader.freqs_mhz[int(round(float(spec.channel_center)))]
        ),
        "period_records": float(spec.period_records),
        "amplitude": float(spec.amplitude),
        "duration_records": int(spec.duration_records or 0),
        "duty_cycle": float(spec.duty_cycle),
        "local_record_start": prepared.background_record_start,
        "local_record_stop": prepared.background_record_stop,
        "all_truth_window_hit": all(
            int(row["truth_window_hit"]) == 1 for row in evaluated
        ),
        "peak_in_truth_count": sum(
            int(row["peak_in_truth"]) for row in evaluated
        ),
        "reproduce_command": (
            "python tests/perf/render_compression.py configured "
            f"--run-dir {run_dir} --component-id {component_id} "
            f"--limit {max(1, int(limit))}"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "top10": evaluated,
                "images": [
                    str(path.relative_to(output_dir))
                    for path in images
                ],
            },
            indent=2,
        )
    )
    lines = [
        "# Configured Low-Frequency CE4 Compression Top 10",
        "",
        f"Injection: `{spec.injection_id}`",
        f"Amplitude: `{spec.amplitude:.9g}`",
        f"Period: `{spec.period_records:.6g}` records",
        "",
        "Reproduce:",
        "",
        "```bash",
        (
            "python tests/perf/render_compression.py configured "
            f"--run-dir {run_dir} --component-id {component_id} "
            f"--limit {max(1, int(limit))}"
        ),
        "```",
        "",
    ]
    for row, image in zip(evaluated, images, strict=True):
        lines.append(
            f"{row['rank']}. `{row['algorithm']}`: "
            f"[image]({image.relative_to(output_dir)})"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render compression benchmark Top-10 figures."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser(
        "synthetic",
        help="Rebuild the retained synthetic compression_final package.",
    )
    configured = subparsers.add_parser(
        "configured",
        help="Render a configured real-CE4 compression rank run.",
    )
    configured.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "pytest_compression_rank_ce4_lowfreq_config_100",
    )
    configured.add_argument(
        "--component-id",
        default="inj_0107_weak_family_c_ultraweak_lowfreq_007_b001_c01",
    )
    configured.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of ranked algorithms to render. Defaults to Top 10.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "synthetic":
        render_synthetic_final()
        return
    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_DIR / args.run_dir
    render_configured_run(run_dir, args.component_id, limit=args.limit)


if __name__ == "__main__":
    main()
