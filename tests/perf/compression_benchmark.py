"""Randomized 2D-to-1D compression benchmark for injected CWT cases."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from cwipss.analysis.injection import BackgroundData, synthetic_background
from cwipss.analysis.simulation import InjectionSpec, inject_periodic_signal
from cwipss.data.readers import CE4Reader, CE4_RECORD_LEN
from cwipss.signal.activity import (
    coherent_structure_map,
    crop_valid_periods,
    low_fraction_noise_floor,
    relative_excess,
    robust_standardize,
    signed_trimmed_period_activity,
    smooth_activity,
)
from cwipss.signal.cwt import cwt_power_cube, period_grid_records
from cwipss.signal.windows import active_windows_from_segments, merge_close_windows, pelt_mean_shift, require_native_pelt
from stage_boundaries import pelt_parameters_from_config, segment_activity_with_pelt


def windowed_period_profile(excess: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Legacy sum-like profile retained only for peripheral rank reproduction."""
    values = np.asarray(excess, dtype=np.float32)
    start = max(0, min(int(start), values.shape[1]))
    stop = max(start + 1, min(int(stop), values.shape[1]))
    duration = max(1, stop - start)
    return (np.nansum(values[:, start:stop], axis=1) / np.sqrt(duration)).astype(np.float32)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "runs" / "compression_random_benchmark"

DEFAULT_TOP10_ALGORITHMS = (
    "max_ratio_s1",
    "max_pool_s1",
    "band_contrast_w1_s1",
    "band_zscore_w1_s1",
    "band_local_contrast_w1_c1_s1",
    "band_local_ratio_w1_s1",
    "band_hybrid_w1_s1",
    "band_max_w5_s1",
    "band_ratio_w3_s1",
    "topk_ratio_k3_s1",
)


CASE_FIELDNAMES = [
    "case_id",
    "algorithm",
    "algorithm_family",
    "algorithm_description",
    "records",
    "channels",
    "background_mode",
    "background_source",
    "background_record_start",
    "background_record_stop",
    "background_channel_start",
    "background_channel_stop",
    "noise_std",
    "signal_model",
    "amplitude",
    "amplitude_factor",
    "period_records",
    "duration_records",
    "duration_fraction",
    "record_start",
    "record_stop",
    "channel_index",
    "bandwidth_channels",
    "drift_channels",
    "duty_cycle",
    "peak_record",
    "peak_in_truth",
    "peak_activity_z",
    "truth_peak_z",
    "truth_mean_z",
    "outside_p95_z",
    "time_contrast_z",
    "peak_global_band_contrast",
    "peak_period_concentration",
    "peak_band_period_records",
    "peak_band_period_error_fraction",
    "window_count",
    "truth_window_hit",
    "truth_window_overlap_fraction",
    "best_window_local_band_contrast",
    "truth_window_local_band_contrast",
    "truth_window_period_records",
    "truth_window_period_error_fraction",
    "preprocess_seconds",
    "reduce_seconds",
    "window_seconds",
    "algorithm_seconds",
    "algorithm_over_preprocess_ratio",
]


SUMMARY_FIELDNAMES = [
    "algorithm",
    "algorithm_family",
    "algorithm_description",
    "case_count",
    "peak_in_truth_rate",
    "truth_window_hit_rate",
    "mean_time_contrast_z",
    "median_time_contrast_z",
    "mean_peak_global_band_contrast",
    "mean_peak_period_concentration",
    "mean_truth_window_local_band_contrast",
    "mean_best_window_local_band_contrast",
    "mean_peak_band_period_error_fraction",
    "mean_truth_window_period_error_fraction",
    "mean_reduce_seconds",
    "mean_window_seconds",
    "mean_algorithm_seconds",
    "p95_algorithm_seconds",
    "mean_algorithm_over_preprocess_ratio",
    "efficiency_pass",
    "rank_score",
]


REGIME_SUMMARY_FIELDNAMES = ["regime", "regime_description"] + SUMMARY_FIELDNAMES


STABILITY_FIELDNAMES = [
    "algorithm",
    "regime_count",
    "rank_wins",
    "efficient_rank_wins",
    "time_contrast_wins",
    "global_band_contrast_wins",
    "period_concentration_wins",
    "window_local_contrast_wins",
    "speed_wins",
    "efficiency_pass_rate",
    "mean_rank_score",
    "mean_peak_in_truth_rate",
    "mean_truth_window_hit_rate",
    "mean_time_contrast_z",
    "mean_peak_global_band_contrast",
    "mean_peak_period_concentration",
    "mean_truth_window_local_band_contrast",
    "mean_algorithm_seconds",
]


@dataclass(frozen=True)
class CompressionAlgorithm:
    name: str
    family: str
    description: str
    reducer: str
    smooth_records: int = 1
    trim_low: float = 0.0
    trim_high: float = 1.0
    quantile: float = 1.0
    top_k: int = 1
    lp_power: float = 2.0
    band_width: int = 1
    softmax_temperature: float = 1.0
    local_context_widths: int = 2


@dataclass(frozen=True)
class CompressionBenchmarkConfig:
    output_dir: Path = DEFAULT_OUTPUT_DIR
    case_count: int = 100
    seed: int = 12345
    records_min: int = 768
    records_max: int = 4096
    channels_min: int = 64
    channels_max: int = 2048
    noise_std_min: float = 1.0
    noise_std_max: float = 1.0
    background_modes: tuple[str, ...] = ("synthetic",)
    ce4_input_path: Path | None = None
    ce4_input_dir: Path = PROJECT_DIR / "data" / "CE4"
    ce4_f_start_mhz: float | None = None
    ce4_f_stop_mhz: float | None = None
    f_start_mhz: float = 0.1
    f_stop_mhz: float = 40.0
    signal_models: tuple[str, ...] = ("single_channel_periodic",)
    bandwidth_channels_min: float = 1.0
    bandwidth_channels_max: float = 9.0
    drift_channels_min: float = 0.0
    drift_channels_max: float = 6.0
    injection_period_min_records: float = 12.0
    injection_period_max_records: float = 256.0
    injection_duration_fraction_min: float = 0.35
    injection_duration_fraction_max: float = 1.0
    amplitude_factor_min: float = 0.25
    amplitude_factor_max: float = 1.50
    duty_cycle_min: float = 0.05
    duty_cycle_max: float = 0.20
    wavelet: str = "cmor1.5-1.0"
    cwt_method: str = "fft"
    cwt_backend: str = "cpu"
    cuda_device: int = 0
    period_min_records: float = 2.0
    period_max_records: float = 512.0
    period_count: int = 96
    period_spacing: str = "log"
    candidate_period_min_records: float = 10.0
    candidate_period_max_records: float = 200.0
    noise_floor_fraction: float = 0.20
    excess_eps_fraction: float = 1e-6
    structure_baseline_quantile: float = 0.10
    structure_scale_quantile: float = 0.20
    structure_z_threshold: float = 1.0
    structure_time_support_records: int = 64
    structure_period_support_bins: int = 3
    structure_min_support_fraction: float = 0.10
    pelt_penalty: float = 16.0
    pelt_min_size_records: int = 384
    pelt_jump_records: int = 1
    pelt_threads: int = 1
    window_min_duration_records: int = 384
    window_min_activity_mean: float = 0.05
    window_merge_gap_records: int = 256
    band_widths: tuple[int, ...] = (1, 3, 5, 7)
    algorithms: tuple[str, ...] = ("all",)
    progress_every: int = 10
    max_algorithm_over_preprocess_ratio: float = 1.0


@dataclass(frozen=True)
class CompressionBenchmarkResult:
    output_dir: Path
    cases_csv: Path
    summary_csv: Path
    summary_json: Path
    case_count: int
    algorithm_count: int
    best_algorithm: str


@dataclass(frozen=True)
class CompressionRegime:
    name: str
    description: str = ""
    case_count: int | None = None
    records_min: int | None = None
    records_max: int | None = None
    channels_min: int | None = None
    channels_max: int | None = None
    injection_period_max_records: float | None = None
    period_max_records: float | None = None
    period_count: int | None = None
    candidate_period_max_records: float | None = None
    structure_time_support_records: int | None = None
    pelt_min_size_records: int | None = None
    window_min_duration_records: int | None = None
    window_merge_gap_records: int | None = None


@dataclass(frozen=True)
class CompressionBenchmarkSuiteResult:
    output_dir: Path
    regime_summary_csv: Path
    stability_csv: Path
    suite_json: Path
    regime_count: int
    overall_best_algorithm: str
    overall_best_efficient_algorithm: str


@dataclass(frozen=True)
class _PreparedCase:
    case_id: str
    records: int
    channels: int
    noise_std: float
    background_mode: str
    background_source: str
    background_record_start: int
    background_record_stop: int
    background_channel_start: int
    background_channel_stop: int
    preprocess_seconds: float
    injection: InjectionSpec
    truth: dict[str, Any]
    valid_periods: np.ndarray
    structured: np.ndarray
    time_band_contrast: np.ndarray
    time_band_ratio: np.ndarray
    time_band_periods: np.ndarray


@dataclass(frozen=True)
class _BackgroundSample:
    mode: str
    source: str
    background: BackgroundData
    noise_std: float
    record_start: int
    record_stop: int
    channel_start: int
    channel_stop: int


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    result = float(np.nanmean(values))
    return result if np.isfinite(result) else 0.0


def _safe_max(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    result = float(np.nanmax(values))
    return result if np.isfinite(result) else 0.0


def _sample_log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    lo = max(float(lo), 1e-12)
    hi = max(float(hi), lo)
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _case_records(config: CompressionBenchmarkConfig, rng: np.random.Generator) -> int:
    lo = max(32, int(config.records_min))
    hi = max(lo, int(config.records_max))
    return int(rng.integers(lo, hi + 1))


def _case_channels(config: CompressionBenchmarkConfig, rng: np.random.Generator) -> int:
    lo = max(1, int(config.channels_min))
    hi = max(lo, int(config.channels_max))
    return int(rng.integers(lo, hi + 1))


def _case_noise_std(config: CompressionBenchmarkConfig, rng: np.random.Generator) -> float:
    lo = max(float(config.noise_std_min), 1e-6)
    hi = max(lo, float(config.noise_std_max))
    if math.isclose(lo, hi):
        return lo
    return float(rng.uniform(lo, hi))


def _complete_ce4_inputs(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".2c")
    return [path for path in files if path.stat().st_size > 0 and path.stat().st_size % CE4_RECORD_LEN == 0]


def _largest_complete_2c(input_dir: str | Path) -> Path:
    complete = _complete_ce4_inputs(input_dir)
    if not complete:
        raise FileNotFoundError(f"No complete CE4 .2C files found under: {input_dir}")
    return max(complete, key=lambda path: path.stat().st_size)


def _resolve_background_modes(modes: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    requested = [str(mode).strip().lower() for mode in modes if str(mode).strip()]
    if not requested:
        return ("synthetic",)
    resolved: list[str] = []
    for mode in requested:
        expanded = ("synthetic", "ce4") if mode == "mixed" else (mode,)
        for candidate in expanded:
            if candidate not in {"synthetic", "ce4"}:
                raise ValueError(f"Unknown background mode: {candidate}")
            if candidate not in resolved:
                resolved.append(candidate)
    return tuple(resolved)


def _resolve_ce4_catalog(config: CompressionBenchmarkConfig) -> tuple[Path, ...]:
    if config.ce4_input_path is not None:
        path = Path(config.ce4_input_path)
        if not path.is_absolute():
            path = PROJECT_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"CE4 input not found: {path}")
        if path.suffix.lower() != ".2c":
            raise ValueError(f"Expected a CE4 .2C input-format file, got: {path}")
        return (path,)
    input_dir = config.ce4_input_dir if Path(config.ce4_input_dir).is_absolute() else PROJECT_DIR / config.ce4_input_dir
    return tuple(_complete_ce4_inputs(input_dir))


def _robust_noise_std(data: np.ndarray) -> float:
    values = np.asarray(data, dtype=np.float64)
    if values.size == 0:
        return 1.0
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    robust = 1.4826 * mad
    if np.isfinite(robust) and robust > 1e-6:
        return robust
    fallback = float(np.nanstd(values))
    if np.isfinite(fallback) and fallback > 1e-6:
        return fallback
    return 1.0


def _resolve_algorithms(names: tuple[str, ...]) -> list[CompressionAlgorithm]:
    catalog = [
        CompressionAlgorithm(
            name="trimmed_mean_05_95_s16",
            family="trimmed_mean",
            description="Current pipeline trimmed mean with 5-95% period keep-range.",
            reducer="trimmed_mean",
            smooth_records=16,
            trim_low=0.05,
            trim_high=0.95,
        ),
        CompressionAlgorithm(
            name="trimmed_mean_20_100_s8",
            family="trimmed_mean",
            description="Upper-heavy trimmed mean that keeps the strongest 80% of period bins.",
            reducer="trimmed_mean",
            smooth_records=8,
            trim_low=0.20,
            trim_high=1.00,
        ),
        CompressionAlgorithm(
            name="quantile_p90_s8",
            family="quantile",
            description="90th-percentile period pooling.",
            reducer="quantile",
            smooth_records=8,
            quantile=0.90,
        ),
        CompressionAlgorithm(
            name="quantile_p95_s8",
            family="quantile",
            description="95th-percentile period pooling.",
            reducer="quantile",
            smooth_records=8,
            quantile=0.95,
        ),
        CompressionAlgorithm(
            name="quantile_p95_s1",
            family="quantile",
            description="Unsmooth 95th-percentile period pooling.",
            reducer="quantile",
            smooth_records=1,
            quantile=0.95,
        ),
        CompressionAlgorithm(
            name="quantile_p98_s4",
            family="quantile",
            description="98th-percentile period pooling with lighter smoothing.",
            reducer="quantile",
            smooth_records=4,
            quantile=0.98,
        ),
        CompressionAlgorithm(
            name="quantile_p98_s1",
            family="quantile",
            description="Unsmooth 98th-percentile period pooling.",
            reducer="quantile",
            smooth_records=1,
            quantile=0.98,
        ),
        CompressionAlgorithm(
            name="max_pool_s4",
            family="extreme",
            description="Max pooling across the period axis.",
            reducer="max",
            smooth_records=4,
        ),
        CompressionAlgorithm(
            name="max_pool_s1",
            family="extreme",
            description="Unsmooth max pooling across the period axis.",
            reducer="max",
            smooth_records=1,
        ),
        CompressionAlgorithm(
            name="topk_mean_k3_s8",
            family="topk",
            description="Average of the strongest 3 period bins per time step.",
            reducer="topk_mean",
            smooth_records=8,
            top_k=3,
        ),
        CompressionAlgorithm(
            name="topk_mean_k3_s1",
            family="topk",
            description="Unsmooth average of the strongest 3 period bins per time step.",
            reducer="topk_mean",
            smooth_records=1,
            top_k=3,
        ),
        CompressionAlgorithm(
            name="topk_mean_k5_s8",
            family="topk",
            description="Average of the strongest 5 period bins per time step.",
            reducer="topk_mean",
            smooth_records=8,
            top_k=5,
        ),
        CompressionAlgorithm(
            name="topk_mean_k5_s1",
            family="topk",
            description="Unsmooth average of the strongest 5 period bins per time step.",
            reducer="topk_mean",
            smooth_records=1,
            top_k=5,
        ),
        CompressionAlgorithm(
            name="topk_mean_k7_s8",
            family="topk",
            description="Average of the strongest 7 period bins per time step.",
            reducer="topk_mean",
            smooth_records=8,
            top_k=7,
        ),
        CompressionAlgorithm(
            name="topk_mean_k7_s1",
            family="topk",
            description="Unsmooth average of the strongest 7 period bins per time step.",
            reducer="topk_mean",
            smooth_records=1,
            top_k=7,
        ),
        CompressionAlgorithm(
            name="l2_norm_s8",
            family="lp_norm",
            description="L2 norm over the period axis.",
            reducer="lp_norm",
            smooth_records=8,
            lp_power=2.0,
        ),
        CompressionAlgorithm(
            name="l4_norm_s8",
            family="lp_norm",
            description="L4 norm over the period axis to emphasize narrow peaks.",
            reducer="lp_norm",
            smooth_records=8,
            lp_power=4.0,
        ),
        CompressionAlgorithm(
            name="l4_norm_s1",
            family="lp_norm",
            description="Unsmooth L4 norm over the period axis to emphasize narrow peaks.",
            reducer="lp_norm",
            smooth_records=1,
            lp_power=4.0,
        ),
        CompressionAlgorithm(
            name="softmax_pool_t075_s8",
            family="softmax",
            description="Softmax-weighted pooling with temperature 0.75.",
            reducer="softmax",
            smooth_records=8,
            softmax_temperature=0.75,
        ),
        CompressionAlgorithm(
            name="softmax_pool_t050_s4",
            family="softmax",
            description="Sharper softmax-weighted pooling with temperature 0.50.",
            reducer="softmax",
            smooth_records=4,
            softmax_temperature=0.50,
        ),
        CompressionAlgorithm(
            name="softmax_pool_t050_s1",
            family="softmax",
            description="Unsmooth sharper softmax-weighted pooling with temperature 0.50.",
            reducer="softmax",
            smooth_records=1,
            softmax_temperature=0.50,
        ),
        CompressionAlgorithm(
            name="band_max_w3_s4",
            family="band_pool",
            description="Strongest contiguous 3-bin band mean.",
            reducer="band_max",
            smooth_records=4,
            band_width=3,
        ),
        CompressionAlgorithm(
            name="band_max_w3_s1",
            family="band_pool",
            description="Unsmooth strongest contiguous 3-bin band mean.",
            reducer="band_max",
            smooth_records=1,
            band_width=3,
        ),
        CompressionAlgorithm(
            name="band_max_w5_s4",
            family="band_pool",
            description="Strongest contiguous 5-bin band mean.",
            reducer="band_max",
            smooth_records=4,
            band_width=5,
        ),
        CompressionAlgorithm(
            name="band_max_w5_s1",
            family="band_pool",
            description="Unsmooth strongest contiguous 5-bin band mean.",
            reducer="band_max",
            smooth_records=1,
            band_width=5,
        ),
        CompressionAlgorithm(
            name="band_max_w7_s4",
            family="band_pool",
            description="Strongest contiguous 7-bin band mean.",
            reducer="band_max",
            smooth_records=4,
            band_width=7,
        ),
        CompressionAlgorithm(
            name="band_max_w7_s1",
            family="band_pool",
            description="Unsmooth strongest contiguous 7-bin band mean.",
            reducer="band_max",
            smooth_records=1,
            band_width=7,
        ),
        CompressionAlgorithm(
            name="band_contrast_w3_s4",
            family="band_contrast",
            description="3-bin band mean contrasted against the global period background.",
            reducer="band_contrast",
            smooth_records=4,
            band_width=3,
        ),
        CompressionAlgorithm(
            name="band_contrast_w1_s1",
            family="band_contrast",
            description="Unsmooth single-bin contrast against the global period background.",
            reducer="band_contrast",
            smooth_records=1,
            band_width=1,
        ),
        CompressionAlgorithm(
            name="band_contrast_w3_s1",
            family="band_contrast",
            description="Unsmooth 3-bin band mean contrasted against the global period background.",
            reducer="band_contrast",
            smooth_records=1,
            band_width=3,
        ),
        CompressionAlgorithm(
            name="band_contrast_w7_s4",
            family="band_contrast",
            description="7-bin band mean contrasted against the global period background.",
            reducer="band_contrast",
            smooth_records=4,
            band_width=7,
        ),
        CompressionAlgorithm(
            name="band_contrast_w7_s1",
            family="band_contrast",
            description="Unsmooth 7-bin band mean contrasted against the global period background.",
            reducer="band_contrast",
            smooth_records=1,
            band_width=7,
        ),
        CompressionAlgorithm(
            name="band_contrast_w5_s4",
            family="band_contrast",
            description="5-bin band mean contrasted against the global period background.",
            reducer="band_contrast",
            smooth_records=4,
            band_width=5,
        ),
        CompressionAlgorithm(
            name="band_contrast_w5_s1",
            family="band_contrast",
            description="Unsmooth 5-bin band mean contrasted against the global period background.",
            reducer="band_contrast",
            smooth_records=1,
            band_width=5,
        ),
        CompressionAlgorithm(
            name="band_ratio_w3_s4",
            family="band_ratio",
            description="3-bin band mean weighted by its share of total period energy.",
            reducer="band_ratio",
            smooth_records=4,
            band_width=3,
        ),
        CompressionAlgorithm(
            name="band_ratio_w3_s1",
            family="band_ratio",
            description="Unsmooth 3-bin band mean weighted by its share of total period energy.",
            reducer="band_ratio",
            smooth_records=1,
            band_width=3,
        ),
        CompressionAlgorithm(
            name="band_ratio_w5_s4",
            family="band_ratio",
            description="5-bin band mean weighted by its share of total period energy.",
            reducer="band_ratio",
            smooth_records=4,
            band_width=5,
        ),
        CompressionAlgorithm(
            name="band_zscore_w3_s4",
            family="band_score",
            description="3-bin band contrast normalized by full-period spread.",
            reducer="band_zscore",
            smooth_records=4,
            band_width=3,
        ),
        CompressionAlgorithm(
            name="band_zscore_w3_s1",
            family="band_score",
            description="Unsmooth 3-bin band contrast normalized by full-period spread.",
            reducer="band_zscore",
            smooth_records=1,
            band_width=3,
        ),
        CompressionAlgorithm(
            name="band_zscore_w1_s1",
            family="band_score",
            description="Unsmooth single-bin contrast normalized by full-period spread.",
            reducer="band_zscore",
            smooth_records=1,
            band_width=1,
        ),
        CompressionAlgorithm(
            name="band_zscore_w5_s4",
            family="band_score",
            description="5-bin band contrast normalized by full-period spread.",
            reducer="band_zscore",
            smooth_records=4,
            band_width=5,
        ),
        CompressionAlgorithm(
            name="band_zscore_w5_s1",
            family="band_score",
            description="Unsmooth 5-bin band contrast normalized by full-period spread.",
            reducer="band_zscore",
            smooth_records=1,
            band_width=5,
        ),
        CompressionAlgorithm(
            name="band_zscore_w7_s1",
            family="band_score",
            description="Unsmooth 7-bin band contrast normalized by full-period spread.",
            reducer="band_zscore",
            smooth_records=1,
            band_width=7,
        ),
        CompressionAlgorithm(
            name="band_local_contrast_w3_s4",
            family="band_local",
            description="3-bin band mean contrasted against nearby period context.",
            reducer="band_local_contrast",
            smooth_records=4,
            band_width=3,
            local_context_widths=2,
        ),
        CompressionAlgorithm(
            name="band_local_contrast_w3_s1",
            family="band_local",
            description="Unsmooth 3-bin band mean contrasted against nearby period context.",
            reducer="band_local_contrast",
            smooth_records=1,
            band_width=3,
            local_context_widths=2,
        ),
        CompressionAlgorithm(
            name="band_local_contrast_w1_s1",
            family="band_local",
            description="Unsmooth single-bin contrast against nearby period context.",
            reducer="band_local_contrast",
            smooth_records=1,
            band_width=1,
            local_context_widths=2,
        ),
        CompressionAlgorithm(
            name="band_local_contrast_w1_c1_s1",
            family="band_local",
            description="Unsmooth single-bin contrast against the tightest nearby period context.",
            reducer="band_local_contrast",
            smooth_records=1,
            band_width=1,
            local_context_widths=1,
        ),
        CompressionAlgorithm(
            name="band_local_contrast_w1_c4_s1",
            family="band_local",
            description="Unsmooth single-bin contrast against a broader nearby period context.",
            reducer="band_local_contrast",
            smooth_records=1,
            band_width=1,
            local_context_widths=4,
        ),
        CompressionAlgorithm(
            name="band_local_contrast_w5_s4",
            family="band_local",
            description="5-bin band mean contrasted against nearby period context.",
            reducer="band_local_contrast",
            smooth_records=4,
            band_width=5,
            local_context_widths=2,
        ),
        CompressionAlgorithm(
            name="band_local_contrast_w5_s1",
            family="band_local",
            description="Unsmooth 5-bin band mean contrasted against nearby period context.",
            reducer="band_local_contrast",
            smooth_records=1,
            band_width=5,
            local_context_widths=2,
        ),
        CompressionAlgorithm(
            name="band_hybrid_w3_s4",
            family="band_hybrid",
            description="3-bin global contrast multiplied by band concentration.",
            reducer="band_hybrid",
            smooth_records=4,
            band_width=3,
        ),
        CompressionAlgorithm(
            name="band_hybrid_w3_s1",
            family="band_hybrid",
            description="Unsmooth 3-bin global contrast multiplied by band concentration.",
            reducer="band_hybrid",
            smooth_records=1,
            band_width=3,
        ),
        CompressionAlgorithm(
            name="band_hybrid_w1_s1",
            family="band_hybrid",
            description="Unsmooth single-bin global contrast multiplied by energy concentration.",
            reducer="band_hybrid",
            smooth_records=1,
            band_width=1,
        ),
        CompressionAlgorithm(
            name="band_hybrid_w5_s4",
            family="band_hybrid",
            description="5-bin global contrast multiplied by band concentration.",
            reducer="band_hybrid",
            smooth_records=4,
            band_width=5,
        ),
        CompressionAlgorithm(
            name="band_hybrid_w5_s1",
            family="band_hybrid",
            description="Unsmooth 5-bin global contrast multiplied by band concentration.",
            reducer="band_hybrid",
            smooth_records=1,
            band_width=5,
        ),
        CompressionAlgorithm(
            name="band_local_ratio_w1_s1",
            family="band_local_ratio",
            description="Unsmooth single-bin local contrast multiplied by energy concentration.",
            reducer="band_local_ratio",
            smooth_records=1,
            band_width=1,
            local_context_widths=2,
        ),
        CompressionAlgorithm(
            name="band_local_ratio_w1_c1_s1",
            family="band_local_ratio",
            description="Unsmooth single-bin tight-context local contrast multiplied by energy concentration.",
            reducer="band_local_ratio",
            smooth_records=1,
            band_width=1,
            local_context_widths=1,
        ),
        CompressionAlgorithm(
            name="band_local_ratio_w1_c4_s1",
            family="band_local_ratio",
            description="Unsmooth single-bin broad-context local contrast multiplied by energy concentration.",
            reducer="band_local_ratio",
            smooth_records=1,
            band_width=1,
            local_context_widths=4,
        ),
        CompressionAlgorithm(
            name="band_local_ratio_w3_s1",
            family="band_local_ratio",
            description="Unsmooth 3-bin local contrast multiplied by energy concentration.",
            reducer="band_local_ratio",
            smooth_records=1,
            band_width=3,
            local_context_widths=2,
        ),
        CompressionAlgorithm(
            name="band_local_zscore_w1_s1",
            family="band_local_zscore",
            description="Unsmooth single-bin local contrast normalized by nearby period spread.",
            reducer="band_local_zscore",
            smooth_records=1,
            band_width=1,
            local_context_widths=2,
        ),
        CompressionAlgorithm(
            name="band_local_zscore_w1_c4_s1",
            family="band_local_zscore",
            description="Unsmooth single-bin broad-context local contrast normalized by nearby period spread.",
            reducer="band_local_zscore",
            smooth_records=1,
            band_width=1,
            local_context_widths=4,
        ),
        CompressionAlgorithm(
            name="band_local_zratio_w1_s1",
            family="band_local_zratio",
            description="Unsmooth single-bin local z-score weighted by energy concentration.",
            reducer="band_local_zratio",
            smooth_records=1,
            band_width=1,
            local_context_widths=2,
        ),
        CompressionAlgorithm(
            name="band_local_zratio_w1_c4_s1",
            family="band_local_zratio",
            description="Unsmooth single-bin broad-context local z-score weighted by energy concentration.",
            reducer="band_local_zratio",
            smooth_records=1,
            band_width=1,
            local_context_widths=4,
        ),
        CompressionAlgorithm(
            name="top3_minus_median_s8",
            family="contrast",
            description="Top-3 period mean minus the period-axis median background.",
            reducer="topk_minus_median",
            smooth_records=8,
            top_k=3,
        ),
        CompressionAlgorithm(
            name="topk_ratio_k3_s8",
            family="topk_ratio",
            description="Top-3 period mean weighted by top-3 energy concentration.",
            reducer="topk_ratio",
            smooth_records=8,
            top_k=3,
        ),
        CompressionAlgorithm(
            name="topk_ratio_k3_s1",
            family="topk_ratio",
            description="Unsmooth top-3 period mean weighted by top-3 energy concentration.",
            reducer="topk_ratio",
            smooth_records=1,
            top_k=3,
        ),
        CompressionAlgorithm(
            name="topk_ratio_k5_s8",
            family="topk_ratio",
            description="Top-5 period mean weighted by top-5 energy concentration.",
            reducer="topk_ratio",
            smooth_records=8,
            top_k=5,
        ),
        CompressionAlgorithm(
            name="topk_ratio_k5_s1",
            family="topk_ratio",
            description="Unsmooth top-5 period mean weighted by top-5 energy concentration.",
            reducer="topk_ratio",
            smooth_records=1,
            top_k=5,
        ),
        CompressionAlgorithm(
            name="max_ratio_s4",
            family="extreme_ratio",
            description="Max period response weighted by its share of total period energy.",
            reducer="max_ratio",
            smooth_records=4,
        ),
        CompressionAlgorithm(
            name="max_ratio_s1",
            family="extreme_ratio",
            description="Unsmooth max period response weighted by its share of total period energy.",
            reducer="max_ratio",
            smooth_records=1,
        ),
        CompressionAlgorithm(
            name="max_ratio_s8",
            family="extreme_ratio",
            description="Heavily smoothed max period response weighted by its share of total period energy.",
            reducer="max_ratio",
            smooth_records=8,
        ),
        CompressionAlgorithm(
            name="trimmed_mean_35_100_s4",
            family="trimmed_mean",
            description="Aggressive upper-tail trimmed mean retaining the strongest 65% of bins.",
            reducer="trimmed_mean",
            smooth_records=4,
            trim_low=0.35,
            trim_high=1.00,
        ),
    ]
    requested = tuple(str(name).strip() for name in names if str(name).strip())
    if not requested or any(name.lower() == "all" for name in requested):
        return catalog
    selected: list[CompressionAlgorithm] = []
    missing: list[str] = []
    by_name = {algorithm.name: algorithm for algorithm in catalog}
    for name in requested:
        if name not in by_name:
            missing.append(name)
            continue
        selected.append(by_name[name])
    if missing:
        raise ValueError(f"Unknown compression algorithms: {', '.join(sorted(missing))}")
    return selected


def compression_algorithm_map() -> dict[str, dict[str, Any]]:
    """Return every supported compression configuration keyed by name."""
    return {
        algorithm.name: asdict(algorithm)
        for algorithm in _resolve_algorithms(("all",))
    }


def default_regimes(*, include_xlarge: bool = False) -> tuple[CompressionRegime, ...]:
    regimes = [
        CompressionRegime(
            name="small",
            description="Small slices with short periods and low PELT minimums.",
            records_min=256,
            records_max=768,
            channels_min=32,
            channels_max=128,
            injection_period_max_records=96.0,
            period_max_records=128.0,
            period_count=32,
            candidate_period_max_records=96.0,
            structure_time_support_records=32,
            pelt_min_size_records=48,
            window_min_duration_records=48,
            window_merge_gap_records=16,
        ),
        CompressionRegime(
            name="medium",
            description="Medium windows with broader random frequency/channel spans.",
            records_min=768,
            records_max=2048,
            channels_min=128,
            channels_max=512,
            injection_period_max_records=192.0,
            period_max_records=256.0,
            period_count=64,
            candidate_period_max_records=192.0,
            structure_time_support_records=64,
            pelt_min_size_records=128,
            window_min_duration_records=128,
            window_merge_gap_records=48,
        ),
        CompressionRegime(
            name="large",
            description="Large windows and wider period search for stress-style cases.",
            records_min=2048,
            records_max=4096,
            channels_min=512,
            channels_max=2048,
            injection_period_max_records=256.0,
            period_max_records=512.0,
            period_count=96,
            candidate_period_max_records=200.0,
            structure_time_support_records=96,
            pelt_min_size_records=256,
            window_min_duration_records=256,
            window_merge_gap_records=96,
        ),
    ]
    if include_xlarge:
        regimes.append(
            CompressionRegime(
                name="xlarge",
                description="Extra-large windows for deeper stress cases and longer periodic structure.",
                records_min=4096,
                records_max=8192,
                channels_min=1024,
                channels_max=2048,
                injection_period_max_records=512.0,
                period_max_records=1024.0,
                period_count=128,
                candidate_period_max_records=384.0,
                structure_time_support_records=128,
                pelt_min_size_records=512,
                window_min_duration_records=512,
                window_merge_gap_records=160,
            )
        )
    return tuple(regimes)


def _sliding_band_sum(values: np.ndarray, width: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    period_count = int(matrix.shape[0])
    width = max(1, min(int(width), period_count))
    cumsum = np.cumsum(matrix, axis=0, dtype=np.float64)
    padded = np.vstack([np.zeros((1, matrix.shape[1]), dtype=np.float64), cumsum])
    sums = padded[width:, :] - padded[:-width, :]
    return sums.astype(np.float32, copy=False)


def _best_band_timeseries(
    values: np.ndarray,
    periods: np.ndarray,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("values must have shape (periods, records)")
    period_values = np.asarray(periods, dtype=np.float64)
    period_count = int(matrix.shape[0])
    if period_count == 0:
        empty = np.zeros(matrix.shape[1], dtype=np.float32)
        return empty, empty, empty
    width = max(1, min(int(width), period_count))
    sums = _sliding_band_sum(matrix, width)
    starts = np.nanargmax(sums, axis=0).astype(np.int64)
    best_sum = sums[starts, np.arange(sums.shape[1], dtype=np.int64)]
    best_mean = best_sum / float(width)
    total = np.nansum(matrix, axis=0)
    denom = np.maximum(total, 1e-12)
    ratio = np.clip(best_sum / denom, 0.0, 1.0)
    outside_count = max(1, period_count - width)
    outside_sum = total - best_sum
    outside_mean = outside_sum / float(outside_count)
    contrast = best_mean - outside_mean
    centers = starts + (width - 1) / 2.0
    period_grid = np.arange(period_count, dtype=np.float64)
    period_records = np.interp(centers, period_grid, period_values).astype(np.float32, copy=False)
    return contrast.astype(np.float32, copy=False), ratio.astype(np.float32, copy=False), period_records


def _best_band_timeseries_metrics(
    values: np.ndarray,
    periods: np.ndarray,
    width: int,
    *,
    local_context_widths: int = 2,
) -> dict[str, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("values must have shape (periods, records)")
    period_values = np.asarray(periods, dtype=np.float64)
    period_count, record_count = matrix.shape
    if period_count == 0:
        empty = np.zeros(record_count, dtype=np.float32)
        return {
            "band_mean": empty,
            "band_ratio": empty,
            "global_contrast": empty,
            "global_zscore": empty,
            "local_contrast": empty,
            "local_zscore": empty,
            "local_relative": empty,
            "period_records": empty,
        }
    width = max(1, min(int(width), period_count))
    sums = _sliding_band_sum(matrix, width)
    cols = np.arange(record_count, dtype=np.int64)
    starts = np.nanargmax(sums, axis=0).astype(np.int64)
    stops = starts + width
    band_sum = sums[starts, cols]
    band_mean = band_sum / float(width)

    total = np.nansum(matrix, axis=0)
    total_sq = np.nansum(np.square(matrix.astype(np.float64, copy=False)), axis=0)
    global_mean = total / float(period_count)
    global_var = np.maximum(total_sq / float(period_count) - np.square(global_mean), 0.0)
    global_std = np.sqrt(global_var)
    global_contrast = band_mean - global_mean
    global_zscore = global_contrast / np.maximum(global_std, 1e-6)
    band_ratio = np.clip(band_sum / np.maximum(total, 1e-12), 0.0, 1.0)

    context_widths = max(1, int(local_context_widths))
    prefix = np.vstack([np.zeros((1, record_count), dtype=np.float64), np.cumsum(matrix, axis=0, dtype=np.float64)])
    prefix_sq = np.vstack(
        [
            np.zeros((1, record_count), dtype=np.float64),
            np.cumsum(np.square(matrix.astype(np.float64, copy=False)), axis=0, dtype=np.float64),
        ]
    )
    local_lo = np.maximum(0, starts - context_widths * width)
    local_hi = np.minimum(period_count, stops + context_widths * width)
    local_segment_sum = prefix[local_hi, cols] - prefix[local_lo, cols]
    local_segment_sq_sum = prefix_sq[local_hi, cols] - prefix_sq[local_lo, cols]
    local_segment_count = local_hi - local_lo
    local_context_sum = local_segment_sum - band_sum
    band_sq_sum = prefix_sq[stops, cols] - prefix_sq[starts, cols]
    local_context_sq_sum = local_segment_sq_sum - band_sq_sum
    local_context_count = local_segment_count - width
    outside_sum = total - band_sum
    outside_sq_sum = total_sq - band_sq_sum
    outside_count = np.maximum(1, period_count - width)
    local_background = np.where(
        local_context_count > 0,
        local_context_sum / np.maximum(local_context_count, 1),
        outside_sum / outside_count,
    )
    local_context_sq_mean = np.where(
        local_context_count > 0,
        local_context_sq_sum / np.maximum(local_context_count, 1),
        outside_sq_sum / outside_count,
    )
    local_var = np.maximum(local_context_sq_mean - np.square(local_background), 0.0)
    local_std = np.sqrt(local_var)
    local_contrast = band_mean - local_background
    local_zscore = local_contrast / np.maximum(local_std, 1e-6)
    local_relative = local_contrast / np.maximum(np.abs(local_background), 1e-6)

    centers = starts + (width - 1) / 2.0
    period_grid = np.arange(period_count, dtype=np.float64)
    period_records = np.interp(centers, period_grid, period_values).astype(np.float32, copy=False)
    return {
        "band_mean": band_mean.astype(np.float32, copy=False),
        "band_ratio": band_ratio.astype(np.float32, copy=False),
        "global_contrast": global_contrast.astype(np.float32, copy=False),
        "global_zscore": global_zscore.astype(np.float32, copy=False),
        "local_contrast": local_contrast.astype(np.float32, copy=False),
        "local_zscore": local_zscore.astype(np.float32, copy=False),
        "local_relative": local_relative.astype(np.float32, copy=False),
        "period_records": period_records,
    }


def _best_band_profile(profile: np.ndarray, periods: np.ndarray, widths: tuple[int, ...]) -> dict[str, float]:
    values = np.asarray(profile, dtype=np.float32)
    period_values = np.asarray(periods, dtype=np.float64)
    period_count = int(values.size)
    if period_count == 0:
        return {
            "local_contrast": 0.0,
            "band_ratio": 0.0,
            "peak_period_records": math.nan,
        }
    best_local_contrast = -math.inf
    best_ratio = 0.0
    best_period = math.nan
    total = float(np.nansum(values))
    for width in widths:
        local_width = max(1, min(int(width), period_count))
        sums = _sliding_band_sum(values[:, None], local_width)[:, 0]
        start = int(np.nanargmax(sums))
        stop = start + local_width
        best_sum = float(sums[start])
        band_mean = best_sum / float(local_width)
        context_lo = max(0, start - 2 * local_width)
        context_hi = min(period_count, stop + 2 * local_width)
        local_context = np.concatenate([values[context_lo:start], values[stop:context_hi]])
        if local_context.size == 0:
            outside = np.concatenate([values[:start], values[stop:]])
            local_context = outside if outside.size else np.zeros(1, dtype=np.float32)
        local_background = _safe_mean(local_context)
        local_contrast = band_mean - local_background
        if local_contrast > best_local_contrast:
            center = start + (local_width - 1) / 2.0
            best_local_contrast = float(local_contrast)
            best_ratio = float(np.clip(best_sum / max(total, 1e-12), 0.0, 1.0))
            best_period = float(np.interp(center, np.arange(period_count, dtype=np.float64), period_values))
    return {
        "local_contrast": best_local_contrast if np.isfinite(best_local_contrast) else 0.0,
        "band_ratio": best_ratio,
        "peak_period_records": best_period,
    }


def _compress_structured(structured: np.ndarray, algorithm: CompressionAlgorithm) -> np.ndarray:
    values = np.asarray(structured, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("structured must have shape (periods, records)")
    if values.size == 0:
        return np.zeros(values.shape[1], dtype=np.float32)
    if algorithm.reducer in {
        "band_ratio",
        "band_zscore",
        "band_local_contrast",
        "band_hybrid",
        "band_local_ratio",
        "band_local_zscore",
        "band_local_zratio",
    }:
        period_count = int(values.shape[0])
        width = max(1, min(int(algorithm.band_width), period_count))
        period_axis = np.arange(period_count, dtype=np.float64)
        band = _best_band_timeseries_metrics(
            values,
            period_axis,
            width,
            local_context_widths=int(algorithm.local_context_widths),
        )
    if algorithm.reducer == "trimmed_mean":
        activity = signed_trimmed_period_activity(
            values,
            trim_low=algorithm.trim_low,
            trim_high=algorithm.trim_high,
        )
    elif algorithm.reducer == "quantile":
        activity = np.nanquantile(values, float(algorithm.quantile), axis=0).astype(np.float32, copy=False)
    elif algorithm.reducer == "max":
        activity = np.nanmax(values, axis=0).astype(np.float32, copy=False)
    elif algorithm.reducer == "topk_mean":
        k = max(1, min(int(algorithm.top_k), values.shape[0]))
        top = np.partition(values, kth=values.shape[0] - k, axis=0)[-k:, :]
        activity = np.nanmean(top, axis=0).astype(np.float32, copy=False)
    elif algorithm.reducer == "lp_norm":
        p = max(float(algorithm.lp_power), 1.0)
        values64 = values.astype(np.float64, copy=False)
        activity = np.power(np.nanmean(np.power(values64, p), axis=0), 1.0 / p).astype(np.float32, copy=False)
    elif algorithm.reducer == "softmax":
        temperature = max(float(algorithm.softmax_temperature), 1e-3)
        shifted = values / temperature
        shifted -= np.nanmax(shifted, axis=0, keepdims=True)
        weights = np.exp(shifted)
        activity = (np.nansum(values * weights, axis=0) / np.maximum(np.nansum(weights, axis=0), 1e-12)).astype(
            np.float32,
            copy=False,
        )
    elif algorithm.reducer == "band_max":
        sums = _sliding_band_sum(values, int(algorithm.band_width))
        width = max(1, min(int(algorithm.band_width), values.shape[0]))
        activity = (np.nanmax(sums, axis=0) / float(width)).astype(np.float32, copy=False)
    elif algorithm.reducer == "band_contrast":
        width = max(1, min(int(algorithm.band_width), values.shape[0]))
        sums = _sliding_band_sum(values, width)
        best_sum = np.nanmax(sums, axis=0)
        best_mean = best_sum / float(width)
        global_mean = np.nanmean(values, axis=0)
        activity = (best_mean - global_mean).astype(np.float32, copy=False)
    elif algorithm.reducer == "band_ratio":
        activity = (band["band_mean"] * band["band_ratio"]).astype(np.float32, copy=False)
    elif algorithm.reducer == "band_zscore":
        activity = band["global_zscore"].astype(np.float32, copy=False)
    elif algorithm.reducer == "band_local_contrast":
        activity = band["local_contrast"].astype(np.float32, copy=False)
    elif algorithm.reducer == "band_hybrid":
        activity = (band["global_contrast"] * band["band_ratio"]).astype(np.float32, copy=False)
    elif algorithm.reducer == "band_local_ratio":
        activity = (band["local_contrast"] * band["band_ratio"]).astype(np.float32, copy=False)
    elif algorithm.reducer == "band_local_zscore":
        activity = band["local_zscore"].astype(np.float32, copy=False)
    elif algorithm.reducer == "band_local_zratio":
        activity = (band["local_zscore"] * band["band_ratio"]).astype(np.float32, copy=False)
    elif algorithm.reducer == "topk_minus_median":
        k = max(1, min(int(algorithm.top_k), values.shape[0]))
        top = np.partition(values, kth=values.shape[0] - k, axis=0)[-k:, :]
        activity = (np.nanmean(top, axis=0) - np.nanmedian(values, axis=0)).astype(np.float32, copy=False)
    elif algorithm.reducer == "topk_ratio":
        k = max(1, min(int(algorithm.top_k), values.shape[0]))
        top = np.partition(values, kth=values.shape[0] - k, axis=0)[-k:, :]
        top_mean = np.nanmean(top, axis=0)
        top_sum = np.nansum(top, axis=0)
        total = np.maximum(np.nansum(values, axis=0), 1e-12)
        activity = (top_mean * np.clip(top_sum / total, 0.0, 1.0)).astype(np.float32, copy=False)
    elif algorithm.reducer == "max_ratio":
        max_values = np.nanmax(values, axis=0)
        total = np.maximum(np.nansum(values, axis=0), 1e-12)
        activity = (max_values * np.clip(max_values / total, 0.0, 1.0)).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown reducer: {algorithm.reducer}")
    smooth_width = max(1, int(algorithm.smooth_records))
    return smooth_activity(activity, smooth_records=smooth_width)


def _period_error_fraction(estimate: float, truth: float) -> float:
    if not np.isfinite(estimate) or truth <= 0:
        return 1.0
    return abs(float(estimate) - float(truth)) / max(float(truth), 1e-12)


def _span_overlap(start_a: int, stop_a: int, start_b: int, stop_b: int) -> float:
    lo_a, hi_a = sorted([int(start_a), int(stop_a)])
    lo_b, hi_b = sorted([int(start_b), int(stop_b)])
    overlap = max(0, min(hi_a, hi_b) - max(lo_a, lo_b))
    denom = max(1, min(hi_a - lo_a, hi_b - lo_b))
    return float(overlap) / float(denom)


def _sample_injection_spec(
    config: CompressionBenchmarkConfig,
    rng: np.random.Generator,
    *,
    case_id: str,
    records: int,
    channels: int,
    noise_std: float,
) -> InjectionSpec:
    duration_fraction = float(
        rng.uniform(
            min(config.injection_duration_fraction_min, config.injection_duration_fraction_max),
            max(config.injection_duration_fraction_min, config.injection_duration_fraction_max),
        )
    )
    duration = max(8, min(records, int(math.ceil(duration_fraction * records))))
    start_max = max(0, records - duration)
    record_start = int(rng.integers(0, start_max + 1)) if start_max else 0
    max_period = min(
        float(config.injection_period_max_records),
        max(float(config.injection_period_min_records), 0.35 * float(duration)),
    )
    period = _sample_log_uniform(rng, float(config.injection_period_min_records), max_period)
    amplitude_factor = _sample_log_uniform(rng, float(config.amplitude_factor_min), float(config.amplitude_factor_max))
    duty_cycle = float(
        rng.uniform(
            min(config.duty_cycle_min, config.duty_cycle_max),
            max(config.duty_cycle_min, config.duty_cycle_max),
        )
    )
    signal_model = str(rng.choice(np.asarray(config.signal_models, dtype=object)))
    bandwidth = 1.0
    if signal_model != "single_channel_periodic":
        bw_lo = max(1.0, float(config.bandwidth_channels_min))
        bw_hi = max(bw_lo, min(float(config.bandwidth_channels_max), max(1.0, channels / 4.0)))
        bandwidth = float(rng.uniform(bw_lo, bw_hi))
    drift = 0.0
    if signal_model == "drifting_ridge":
        drift_lo = max(0.0, float(config.drift_channels_min))
        drift_hi = max(drift_lo, min(float(config.drift_channels_max), max(0.0, channels / 3.0)))
        drift = float(rng.uniform(drift_lo, drift_hi))
        if drift > 0.0 and rng.random() < 0.5:
            drift *= -1.0
    return InjectionSpec(
        injection_id=f"{case_id}_inj",
        signal_model=signal_model,
        period_records=period,
        amplitude=float(noise_std * amplitude_factor),
        record_start=record_start,
        duration_records=duration,
        channel_center=float(rng.integers(0, max(1, channels))),
        bandwidth_channels=bandwidth,
        duty_cycle=duty_cycle,
        phase=float(rng.uniform(0.0, 1.0)),
        drift_channels=drift,
    )


def _sample_synthetic_background(
    config: CompressionBenchmarkConfig,
    rng: np.random.Generator,
    *,
    records: int,
    channels: int,
    noise_std: float,
) -> _BackgroundSample:
    background = synthetic_background(
        records=records,
        channels=channels,
        noise_std=noise_std,
        seed=int(rng.integers(0, 2**31 - 1)),
        f_start_mhz=config.f_start_mhz,
        f_stop_mhz=config.f_stop_mhz,
    )
    return _BackgroundSample(
        mode="synthetic",
        source=background.source_name,
        background=background,
        noise_std=float(noise_std),
        record_start=0,
        record_stop=records,
        channel_start=0,
        channel_stop=channels,
    )


def _sample_ce4_background(
    config: CompressionBenchmarkConfig,
    rng: np.random.Generator,
    *,
    requested_records: int,
    requested_channels: int,
    catalog: tuple[Path, ...],
    reader_cache: dict[Path, CE4Reader],
) -> _BackgroundSample:
    if not catalog:
        input_dir = config.ce4_input_dir if Path(config.ce4_input_dir).is_absolute() else PROJECT_DIR / config.ce4_input_dir
        raise FileNotFoundError(f"No complete CE4 .2C files found under: {input_dir}")
    eligible = [path for path in catalog if int(path.stat().st_size // CE4_RECORD_LEN) >= int(requested_records)]
    if not eligible:
        raise ValueError(f"No CE4 .2C input is long enough for records={requested_records}")
    path = eligible[int(rng.integers(0, len(eligible)))]
    reader = reader_cache.setdefault(path, CE4Reader(path))
    allowed = reader.freq_slice(config.ce4_f_start_mhz, config.ce4_f_stop_mhz)
    allowed_start = int(allowed.start)
    allowed_stop = int(allowed.stop)
    available_channels = max(1, allowed_stop - allowed_start)
    channels = min(max(1, int(requested_channels)), available_channels)
    records = min(max(1, int(requested_records)), int(reader.n_records))
    record_start_max = max(0, int(reader.n_records) - records)
    channel_start_max = max(0, available_channels - channels)
    record_start = int(rng.integers(0, record_start_max + 1)) if record_start_max else 0
    channel_offset = int(rng.integers(0, channel_start_max + 1)) if channel_start_max else 0
    channel_start = allowed_start + channel_offset
    record_slice = slice(record_start, record_start + records)
    channel_slice = slice(channel_start, channel_start + channels)
    block = reader.read_block(record_slice, channel_slice)
    background = BackgroundData(
        data=np.asarray(block.data, dtype=np.float32),
        freqs_mhz=np.asarray(block.freqs_mhz, dtype=np.float64),
        source_name=str(path),
        tsamp_seconds=float(reader.tsamp_seconds),
    )
    return _BackgroundSample(
        mode="ce4",
        source=str(path),
        background=background,
        noise_std=_robust_noise_std(block.data),
        record_start=int(record_slice.start),
        record_stop=int(record_slice.stop),
        channel_start=int(channel_slice.start),
        channel_stop=int(channel_slice.stop),
    )


def _sample_background(
    config: CompressionBenchmarkConfig,
    rng: np.random.Generator,
    *,
    requested_records: int,
    requested_channels: int,
    requested_noise_std: float,
    catalog: tuple[Path, ...],
    reader_cache: dict[Path, CE4Reader],
) -> _BackgroundSample:
    modes = _resolve_background_modes(config.background_modes)
    mode = modes[int(rng.integers(0, len(modes)))]
    if mode == "synthetic":
        return _sample_synthetic_background(
            config,
            rng,
            records=requested_records,
            channels=requested_channels,
            noise_std=requested_noise_std,
        )
    return _sample_ce4_background(
        config,
        rng,
        requested_records=requested_records,
        requested_channels=requested_channels,
        catalog=catalog,
        reader_cache=reader_cache,
    )


def _prepare_case(
    case_index: int,
    config: CompressionBenchmarkConfig,
    rng: np.random.Generator,
    *,
    ce4_catalog: tuple[Path, ...],
    reader_cache: dict[Path, CE4Reader],
) -> _PreparedCase:
    case_id = f"case_{case_index:04d}"
    requested_records = _case_records(config, rng)
    requested_channels = _case_channels(config, rng)
    requested_noise_std = _case_noise_std(config, rng)
    sample = _sample_background(
        config,
        rng,
        requested_records=requested_records,
        requested_channels=requested_channels,
        requested_noise_std=requested_noise_std,
        catalog=ce4_catalog,
        reader_cache=reader_cache,
    )
    background = sample.background
    records, channels = map(int, background.data.shape)
    noise_std = float(sample.noise_std)
    spec = _sample_injection_spec(config, rng, case_id=case_id, records=records, channels=channels, noise_std=noise_std)
    injected_data, truth = inject_periodic_signal(background.data, spec)
    channel_idx = int(min(max(int(round(float(spec.channel_center))), 0), channels - 1))
    channel_data = injected_data[:, channel_idx : channel_idx + 1]
    preprocess_start = perf_counter()
    resolved_period_max = min(float(config.period_max_records), max(float(config.period_min_records), 0.5 * records))
    resolved_candidate_max = min(
        float(config.candidate_period_max_records),
        max(float(config.candidate_period_min_records), resolved_period_max),
    )
    case_periods = period_grid_records(
        float(config.period_min_records),
        resolved_period_max,
        int(config.period_count),
        str(config.period_spacing),
    )
    power = cwt_power_cube(
        channel_data,
        wavelet=config.wavelet,
        periods=case_periods,
        method=config.cwt_method,
        backend=config.cwt_backend,
        cuda_device=config.cuda_device,
        normalize_channels=True,
    )[:, :, 0]
    valid_power, valid_periods, _mask = crop_valid_periods(
        power,
        case_periods,
        float(config.candidate_period_min_records),
        resolved_candidate_max,
    )
    noise_floor = low_fraction_noise_floor(valid_power, fraction=float(config.noise_floor_fraction))
    excess = relative_excess(valid_power, noise_floor, eps_fraction=float(config.excess_eps_fraction))
    time_support = min(int(config.structure_time_support_records), max(1, records // 8))
    structured = coherent_structure_map(
        excess,
        baseline_quantile=float(config.structure_baseline_quantile),
        scale_quantile=float(config.structure_scale_quantile),
        z_threshold=float(config.structure_z_threshold),
        time_support_records=time_support,
        period_support_bins=int(config.structure_period_support_bins),
        min_support_fraction=float(config.structure_min_support_fraction),
    )
    preprocess_seconds = perf_counter() - preprocess_start
    widths = tuple(width for width in config.band_widths if int(width) > 0)
    best_contrast = np.full(records, -np.inf, dtype=np.float32)
    best_ratio = np.zeros(records, dtype=np.float32)
    best_periods = np.full(records, np.nan, dtype=np.float32)
    for width in widths:
        contrast, ratio, period_records = _best_band_timeseries(structured, valid_periods, int(width))
        update = contrast > best_contrast
        best_contrast = np.where(update, contrast, best_contrast)
        best_ratio = np.where(update, ratio, best_ratio)
        best_periods = np.where(update, period_records, best_periods)
    truth = dict(truth)
    truth["record_stop"] = int(truth["record_start"] + truth["duration_records"])
    truth["channel_index"] = channel_idx
    return _PreparedCase(
        case_id=case_id,
        records=records,
        channels=channels,
        noise_std=noise_std,
        background_mode=sample.mode,
        background_source=sample.source,
        background_record_start=int(sample.record_start),
        background_record_stop=int(sample.record_stop),
        background_channel_start=int(sample.channel_start),
        background_channel_stop=int(sample.channel_stop),
        preprocess_seconds=float(preprocess_seconds),
        injection=spec,
        truth=truth,
        valid_periods=valid_periods,
        structured=structured,
        time_band_contrast=best_contrast.astype(np.float32, copy=False),
        time_band_ratio=best_ratio.astype(np.float32, copy=False),
        time_band_periods=best_periods.astype(np.float32, copy=False),
    )


def _activity_truth_metrics(activity_z: np.ndarray, truth: dict[str, Any]) -> dict[str, float | int]:
    values = np.asarray(activity_z, dtype=np.float32)
    truth_start = max(0, min(int(truth["record_start"]), values.size))
    truth_stop = max(truth_start + 1, min(int(truth["record_stop"]), values.size))
    truth_mask = np.zeros(values.size, dtype=bool)
    truth_mask[truth_start:truth_stop] = True
    outside = values[~truth_mask]
    peak_record = int(np.nanargmax(values)) if values.size else 0
    truth_values = values[truth_mask]
    return {
        "peak_record": peak_record,
        "peak_in_truth": int(truth_start <= peak_record < truth_stop),
        "peak_activity_z": _safe_max(values),
        "truth_peak_z": _safe_max(truth_values),
        "truth_mean_z": _safe_mean(truth_values),
        "outside_p95_z": float(np.nanquantile(outside, 0.95)) if outside.size else 0.0,
    }


def _window_rows(
    activity_z: np.ndarray,
    truth: dict[str, Any],
    structured: np.ndarray,
    periods: np.ndarray,
    config: CompressionBenchmarkConfig,
) -> dict[str, float | int]:
    windows = segment_activity_with_pelt(
        activity_z,
        pelt_parameters_from_config(config),
        activity_z=activity_z,
    ).windows
    widths = tuple(width for width in config.band_widths if int(width) > 0)
    best_window_contrast = 0.0
    truth_window_hit = 0
    truth_window_overlap = 0.0
    truth_window_contrast = 0.0
    truth_window_period = math.nan
    for window in windows:
        start = int(window["record_start"])
        stop = int(window["record_stop"])
        profile = windowed_period_profile(structured, start, stop)
        band = _best_band_profile(profile, periods, widths)
        best_window_contrast = max(best_window_contrast, float(band["local_contrast"]))
        overlap = _span_overlap(start, stop, int(truth["record_start"]), int(truth["record_stop"]))
        if overlap > truth_window_overlap:
            truth_window_hit = int(overlap > 0.0)
            truth_window_overlap = overlap
            truth_window_contrast = float(band["local_contrast"])
            truth_window_period = float(band["peak_period_records"])
    return {
        "window_count": len(windows),
        "truth_window_hit": truth_window_hit,
        "truth_window_overlap_fraction": truth_window_overlap,
        "best_window_local_band_contrast": best_window_contrast,
        "truth_window_local_band_contrast": truth_window_contrast,
        "truth_window_period_records": truth_window_period,
    }


def _evaluate_algorithm(
    prepared: _PreparedCase,
    algorithm: CompressionAlgorithm,
    config: CompressionBenchmarkConfig,
) -> dict[str, Any]:
    reduce_start = perf_counter()
    activity = _compress_structured(prepared.structured, algorithm)
    activity_z = robust_standardize(activity)
    reduce_seconds = perf_counter() - reduce_start
    window_start = perf_counter()
    truth_metrics = _activity_truth_metrics(activity_z, prepared.truth)
    window_metrics = _window_rows(activity_z, prepared.truth, prepared.structured, prepared.valid_periods, config)
    window_seconds = perf_counter() - window_start
    algorithm_seconds = reduce_seconds + window_seconds
    peak_record = int(truth_metrics["peak_record"])
    peak_period = float(prepared.time_band_periods[peak_record]) if prepared.time_band_periods.size else math.nan
    truth_period = float(prepared.truth["period_records"])
    row = {
        "case_id": prepared.case_id,
        "algorithm": algorithm.name,
        "algorithm_family": algorithm.family,
        "algorithm_description": algorithm.description,
        "records": prepared.records,
        "channels": prepared.channels,
        "background_mode": prepared.background_mode,
        "background_source": prepared.background_source,
        "background_record_start": prepared.background_record_start,
        "background_record_stop": prepared.background_record_stop,
        "background_channel_start": prepared.background_channel_start,
        "background_channel_stop": prepared.background_channel_stop,
        "noise_std": prepared.noise_std,
        "signal_model": prepared.injection.signal_model,
        "amplitude": float(prepared.injection.amplitude),
        "amplitude_factor": float(prepared.injection.amplitude / max(prepared.noise_std, 1e-12)),
        "period_records": truth_period,
        "duration_records": int(prepared.truth["duration_records"]),
        "duration_fraction": float(prepared.truth["duration_records"]) / float(max(1, prepared.records)),
        "record_start": int(prepared.truth["record_start"]),
        "record_stop": int(prepared.truth["record_stop"]),
        "channel_index": int(prepared.truth["channel_index"]),
        "bandwidth_channels": float(prepared.injection.bandwidth_channels),
        "drift_channels": float(prepared.injection.drift_channels),
        "duty_cycle": float(prepared.injection.duty_cycle),
        "peak_record": peak_record,
        "peak_in_truth": int(truth_metrics["peak_in_truth"]),
        "peak_activity_z": float(truth_metrics["peak_activity_z"]),
        "truth_peak_z": float(truth_metrics["truth_peak_z"]),
        "truth_mean_z": float(truth_metrics["truth_mean_z"]),
        "outside_p95_z": float(truth_metrics["outside_p95_z"]),
        "time_contrast_z": float(truth_metrics["truth_peak_z"]) - float(truth_metrics["outside_p95_z"]),
        "peak_global_band_contrast": float(prepared.time_band_contrast[peak_record]) if prepared.time_band_contrast.size else 0.0,
        "peak_period_concentration": float(prepared.time_band_ratio[peak_record]) if prepared.time_band_ratio.size else 0.0,
        "peak_band_period_records": peak_period,
        "peak_band_period_error_fraction": _period_error_fraction(peak_period, truth_period),
        "window_count": int(window_metrics["window_count"]),
        "truth_window_hit": int(window_metrics["truth_window_hit"]),
        "truth_window_overlap_fraction": float(window_metrics["truth_window_overlap_fraction"]),
        "best_window_local_band_contrast": float(window_metrics["best_window_local_band_contrast"]),
        "truth_window_local_band_contrast": float(window_metrics["truth_window_local_band_contrast"]),
        "truth_window_period_records": float(window_metrics["truth_window_period_records"]),
        "truth_window_period_error_fraction": _period_error_fraction(
            float(window_metrics["truth_window_period_records"]),
            truth_period,
        ),
        "preprocess_seconds": float(prepared.preprocess_seconds),
        "reduce_seconds": float(reduce_seconds),
        "window_seconds": float(window_seconds),
        "algorithm_seconds": float(algorithm_seconds),
        "algorithm_over_preprocess_ratio": float(algorithm_seconds / max(prepared.preprocess_seconds, 1e-12)),
    }
    return row


def _summary_rows(
    case_rows: list[dict[str, Any]],
    algorithms: list[CompressionAlgorithm],
    *,
    max_algorithm_over_preprocess_ratio: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for algorithm in algorithms:
        selected = [row for row in case_rows if row["algorithm"] == algorithm.name]
        if not selected:
            continue
        reduce = np.asarray([float(row["reduce_seconds"]) for row in selected], dtype=np.float64)
        window = np.asarray([float(row["window_seconds"]) for row in selected], dtype=np.float64)
        algorithm_seconds = np.asarray([float(row["algorithm_seconds"]) for row in selected], dtype=np.float64)
        algorithm_ratio = np.asarray(
            [float(row["algorithm_over_preprocess_ratio"]) for row in selected],
            dtype=np.float64,
        )
        period_error = np.asarray([float(row["peak_band_period_error_fraction"]) for row in selected], dtype=np.float64)
        truth_window_error = np.asarray(
            [float(row["truth_window_period_error_fraction"]) for row in selected if int(row["truth_window_hit"]) > 0],
            dtype=np.float64,
        )
        rows.append(
            {
                "algorithm": algorithm.name,
                "algorithm_family": algorithm.family,
                "algorithm_description": algorithm.description,
                "case_count": len(selected),
                "peak_in_truth_rate": _safe_mean(np.asarray([int(row["peak_in_truth"]) for row in selected], dtype=np.float64)),
                "truth_window_hit_rate": _safe_mean(
                    np.asarray([int(row["truth_window_hit"]) for row in selected], dtype=np.float64)
                ),
                "mean_time_contrast_z": _safe_mean(np.asarray([float(row["time_contrast_z"]) for row in selected], dtype=np.float64)),
                "median_time_contrast_z": float(
                    np.nanmedian(np.asarray([float(row["time_contrast_z"]) for row in selected], dtype=np.float64))
                ),
                "mean_peak_global_band_contrast": _safe_mean(
                    np.asarray([float(row["peak_global_band_contrast"]) for row in selected], dtype=np.float64)
                ),
                "mean_peak_period_concentration": _safe_mean(
                    np.asarray([float(row["peak_period_concentration"]) for row in selected], dtype=np.float64)
                ),
                "mean_truth_window_local_band_contrast": _safe_mean(
                    np.asarray([float(row["truth_window_local_band_contrast"]) for row in selected], dtype=np.float64)
                ),
                "mean_best_window_local_band_contrast": _safe_mean(
                    np.asarray([float(row["best_window_local_band_contrast"]) for row in selected], dtype=np.float64)
                ),
                "mean_peak_band_period_error_fraction": _safe_mean(period_error),
                "mean_truth_window_period_error_fraction": _safe_mean(truth_window_error),
                "mean_reduce_seconds": _safe_mean(reduce),
                "mean_window_seconds": _safe_mean(window),
                "mean_algorithm_seconds": _safe_mean(algorithm_seconds),
                "p95_algorithm_seconds": float(np.nanquantile(algorithm_seconds, 0.95)) if algorithm_seconds.size else 0.0,
                "mean_algorithm_over_preprocess_ratio": _safe_mean(algorithm_ratio),
                "efficiency_pass": "unknown",
                "rank_score": 0.0,
            }
        )
    if not rows:
        return rows

    def scale(key: str, inverse: bool = False) -> dict[str, float]:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
        if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(lo, hi):
            return {str(row["algorithm"]): 0.5 for row in rows}
        scaled = (values - lo) / (hi - lo)
        if inverse:
            scaled = 1.0 - scaled
        return {str(row["algorithm"]): float(value) for row, value in zip(rows, scaled, strict=True)}

    score_maps = {
        "peak_in_truth_rate": scale("peak_in_truth_rate"),
        "truth_window_hit_rate": scale("truth_window_hit_rate"),
        "mean_time_contrast_z": scale("mean_time_contrast_z"),
        "mean_peak_global_band_contrast": scale("mean_peak_global_band_contrast"),
        "mean_peak_period_concentration": scale("mean_peak_period_concentration"),
        "mean_truth_window_local_band_contrast": scale("mean_truth_window_local_band_contrast"),
        "mean_algorithm_seconds": scale("mean_algorithm_seconds", inverse=True),
        "mean_algorithm_over_preprocess_ratio": scale("mean_algorithm_over_preprocess_ratio", inverse=True),
    }
    for row in rows:
        name = str(row["algorithm"])
        row["rank_score"] = (
            0.20 * score_maps["peak_in_truth_rate"][name]
            + 0.15 * score_maps["truth_window_hit_rate"][name]
            + 0.20 * score_maps["mean_time_contrast_z"][name]
            + 0.20 * score_maps["mean_peak_global_band_contrast"][name]
            + 0.15 * score_maps["mean_peak_period_concentration"][name]
            + 0.05 * score_maps["mean_truth_window_local_band_contrast"][name]
            + 0.03 * score_maps["mean_algorithm_seconds"][name]
            + 0.02 * score_maps["mean_algorithm_over_preprocess_ratio"][name]
        )
        row["efficiency_pass"] = (
            "1"
            if float(row["mean_algorithm_over_preprocess_ratio"]) <= float(max_algorithm_over_preprocess_ratio)
            else "0"
        )
    rows.sort(key=lambda row: (float(row["rank_score"]), float(row["peak_in_truth_rate"])), reverse=True)
    return rows


def run_random_compression_benchmark(config: CompressionBenchmarkConfig) -> CompressionBenchmarkResult:
    require_native_pelt()
    output_dir = config.output_dir if config.output_dir.is_absolute() else PROJECT_DIR / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    algorithms = _resolve_algorithms(config.algorithms)
    background_modes = _resolve_background_modes(config.background_modes)
    ce4_catalog = _resolve_ce4_catalog(config) if "ce4" in background_modes else ()
    reader_cache: dict[Path, CE4Reader] = {}
    rng = np.random.default_rng(int(config.seed))
    case_rows: list[dict[str, Any]] = []
    case_backgrounds: list[dict[str, Any]] = []
    case_total = max(1, int(config.case_count))
    for case_index in range(1, case_total + 1):
        if int(config.progress_every) > 0 and (case_index == 1 or case_index % int(config.progress_every) == 0):
            print(f"[compression-benchmark] case {case_index}/{case_total}")
        prepared = _prepare_case(case_index, config, rng, ce4_catalog=ce4_catalog, reader_cache=reader_cache)
        case_backgrounds.append(
            {
                "case_id": prepared.case_id,
                "background_mode": prepared.background_mode,
                "background_source": prepared.background_source,
                "background_record_start": prepared.background_record_start,
                "background_record_stop": prepared.background_record_stop,
                "background_channel_start": prepared.background_channel_start,
                "background_channel_stop": prepared.background_channel_stop,
                "records": prepared.records,
                "channels": prepared.channels,
            }
        )
        for algorithm in algorithms:
            case_rows.append(_evaluate_algorithm(prepared, algorithm, config))
    summary_rows = _summary_rows(
        case_rows,
        algorithms,
        max_algorithm_over_preprocess_ratio=float(config.max_algorithm_over_preprocess_ratio),
    )
    cases_csv = output_dir / "compression_cases.csv"
    summary_csv = output_dir / "compression_summary.csv"
    summary_json = output_dir / "compression_summary.json"
    _write_csv(cases_csv, CASE_FIELDNAMES, case_rows)
    _write_csv(summary_csv, SUMMARY_FIELDNAMES, summary_rows)
    best_algorithm = str(summary_rows[0]["algorithm"]) if summary_rows else ""
    efficient_rows = [row for row in summary_rows if str(row.get("efficiency_pass", "0")) == "1"]
    best_efficient_algorithm = str(efficient_rows[0]["algorithm"]) if efficient_rows else ""
    background_mode_counts: dict[str, int] = {}
    background_source_counts: dict[str, int] = {}
    for row in case_backgrounds:
        background_mode_counts[str(row["background_mode"])] = background_mode_counts.get(str(row["background_mode"]), 0) + 1
        background_source_counts[str(row["background_source"])] = background_source_counts.get(str(row["background_source"]), 0) + 1
    summary_payload = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "algorithm_count": len(algorithms),
        "case_count": case_total,
        "background_mode_counts": background_mode_counts,
        "background_source_counts": background_source_counts,
        "best_algorithm": best_algorithm,
        "best_efficient_algorithm": best_efficient_algorithm,
        "metric_leaders": {
            "rank_score": best_algorithm,
            "efficient_rank_score": best_efficient_algorithm,
            "mean_time_contrast_z": max(summary_rows, key=lambda row: float(row["mean_time_contrast_z"]))["algorithm"] if summary_rows else "",
            "mean_peak_global_band_contrast": max(summary_rows, key=lambda row: float(row["mean_peak_global_band_contrast"]))["algorithm"] if summary_rows else "",
            "mean_peak_period_concentration": max(summary_rows, key=lambda row: float(row["mean_peak_period_concentration"]))["algorithm"] if summary_rows else "",
            "mean_truth_window_local_band_contrast": max(summary_rows, key=lambda row: float(row["mean_truth_window_local_band_contrast"]))["algorithm"] if summary_rows else "",
            "mean_algorithm_seconds": min(summary_rows, key=lambda row: float(row["mean_algorithm_seconds"]))["algorithm"] if summary_rows else "",
        },
        "case_backgrounds": case_backgrounds,
        "summary_rows": summary_rows,
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=True))
    return CompressionBenchmarkResult(
        output_dir=output_dir,
        cases_csv=cases_csv,
        summary_csv=summary_csv,
        summary_json=summary_json,
        case_count=case_total,
        algorithm_count=len(algorithms),
        best_algorithm=best_algorithm,
    )


def _config_for_regime(
    base_config: CompressionBenchmarkConfig,
    regime: CompressionRegime,
    *,
    output_dir: Path,
) -> CompressionBenchmarkConfig:
    overrides: dict[str, Any] = {
        "output_dir": output_dir,
    }
    for key in (
        "case_count",
        "records_min",
        "records_max",
        "channels_min",
        "channels_max",
        "injection_period_max_records",
        "period_max_records",
        "period_count",
        "candidate_period_max_records",
        "structure_time_support_records",
        "pelt_min_size_records",
        "window_min_duration_records",
        "window_merge_gap_records",
    ):
        value = getattr(regime, key)
        if value is not None:
            overrides[key] = value
    return replace(base_config, **overrides)


def run_stratified_compression_benchmark(
    *,
    base_config: CompressionBenchmarkConfig,
    regimes: tuple[CompressionRegime, ...] | list[CompressionRegime] | None = None,
    output_dir: Path | None = None,
) -> CompressionBenchmarkSuiteResult:
    selected_regimes = tuple(regimes or default_regimes())
    suite_output_dir = output_dir or (base_config.output_dir if base_config.output_dir.is_absolute() else PROJECT_DIR / base_config.output_dir)
    if not suite_output_dir.is_absolute():
        suite_output_dir = PROJECT_DIR / suite_output_dir
    suite_output_dir.mkdir(parents=True, exist_ok=True)

    regime_rows: list[dict[str, Any]] = []
    stability: dict[str, dict[str, float]] = {}
    regime_rank_leaders: dict[str, str] = {}
    regime_background_mode_counts: dict[str, dict[str, int]] = {}
    regime_background_source_counts: dict[str, dict[str, int]] = {}
    suite_background_mode_counts: dict[str, int] = {}
    suite_background_source_counts: dict[str, int] = {}
    best_single_regime_algorithm = ""
    best_single_regime_score = -math.inf

    for regime in selected_regimes:
        regime_output = suite_output_dir / regime.name
        result = run_random_compression_benchmark(_config_for_regime(base_config, regime, output_dir=regime_output))
        payload = json.loads(result.summary_json.read_text())
        summary_rows = list(payload.get("summary_rows", []))
        metric_leaders = dict(payload.get("metric_leaders", {}))
        mode_counts = {str(key): int(value) for key, value in dict(payload.get("background_mode_counts", {})).items()}
        source_counts = {str(key): int(value) for key, value in dict(payload.get("background_source_counts", {})).items()}
        regime_background_mode_counts[regime.name] = mode_counts
        regime_background_source_counts[regime.name] = source_counts
        for key, value in mode_counts.items():
            suite_background_mode_counts[key] = suite_background_mode_counts.get(key, 0) + int(value)
        for key, value in source_counts.items():
            suite_background_source_counts[key] = suite_background_source_counts.get(key, 0) + int(value)
        if summary_rows:
            regime_rank_leaders[regime.name] = str(summary_rows[0]["algorithm"])
            if float(summary_rows[0].get("rank_score", -math.inf)) > best_single_regime_score:
                best_single_regime_score = float(summary_rows[0]["rank_score"])
                best_single_regime_algorithm = str(summary_rows[0]["algorithm"])
        for row in summary_rows:
            algorithm = str(row["algorithm"])
            enriched = {"regime": regime.name, "regime_description": regime.description}
            enriched.update(row)
            regime_rows.append(enriched)
            bucket = stability.setdefault(
                algorithm,
                {
                    "algorithm": algorithm,
                    "regime_count": 0,
                    "rank_wins": 0,
                    "efficient_rank_wins": 0,
                    "time_contrast_wins": 0,
                    "global_band_contrast_wins": 0,
                    "period_concentration_wins": 0,
                    "window_local_contrast_wins": 0,
                    "speed_wins": 0,
                    "efficiency_pass_count": 0,
                    "sum_rank_score": 0.0,
                    "sum_peak_in_truth_rate": 0.0,
                    "sum_truth_window_hit_rate": 0.0,
                    "sum_time_contrast_z": 0.0,
                    "sum_peak_global_band_contrast": 0.0,
                    "sum_peak_period_concentration": 0.0,
                    "sum_truth_window_local_band_contrast": 0.0,
                    "sum_algorithm_seconds": 0.0,
                },
            )
            bucket["regime_count"] += 1
            bucket["sum_rank_score"] += float(row["rank_score"])
            bucket["sum_peak_in_truth_rate"] += float(row["peak_in_truth_rate"])
            bucket["sum_truth_window_hit_rate"] += float(row["truth_window_hit_rate"])
            bucket["sum_time_contrast_z"] += float(row["mean_time_contrast_z"])
            bucket["sum_peak_global_band_contrast"] += float(row["mean_peak_global_band_contrast"])
            bucket["sum_peak_period_concentration"] += float(row["mean_peak_period_concentration"])
            bucket["sum_truth_window_local_band_contrast"] += float(row["mean_truth_window_local_band_contrast"])
            bucket["sum_algorithm_seconds"] += float(row["mean_algorithm_seconds"])
            bucket["efficiency_pass_count"] += 1 if str(row.get("efficiency_pass", "0")) == "1" else 0
        if metric_leaders.get("rank_score"):
            stability[metric_leaders["rank_score"]]["rank_wins"] += 1
        if metric_leaders.get("efficient_rank_score"):
            stability[metric_leaders["efficient_rank_score"]]["efficient_rank_wins"] += 1
        if metric_leaders.get("mean_time_contrast_z"):
            stability[metric_leaders["mean_time_contrast_z"]]["time_contrast_wins"] += 1
        if metric_leaders.get("mean_peak_global_band_contrast"):
            stability[metric_leaders["mean_peak_global_band_contrast"]]["global_band_contrast_wins"] += 1
        if metric_leaders.get("mean_peak_period_concentration"):
            stability[metric_leaders["mean_peak_period_concentration"]]["period_concentration_wins"] += 1
        if metric_leaders.get("mean_truth_window_local_band_contrast"):
            stability[metric_leaders["mean_truth_window_local_band_contrast"]]["window_local_contrast_wins"] += 1
        if metric_leaders.get("mean_algorithm_seconds"):
            stability[metric_leaders["mean_algorithm_seconds"]]["speed_wins"] += 1

    stability_rows: list[dict[str, Any]] = []
    for bucket in stability.values():
        count = max(1, int(bucket["regime_count"]))
        stability_rows.append(
            {
                "algorithm": bucket["algorithm"],
                "regime_count": count,
                "rank_wins": int(bucket["rank_wins"]),
                "efficient_rank_wins": int(bucket["efficient_rank_wins"]),
                "time_contrast_wins": int(bucket["time_contrast_wins"]),
                "global_band_contrast_wins": int(bucket["global_band_contrast_wins"]),
                "period_concentration_wins": int(bucket["period_concentration_wins"]),
                "window_local_contrast_wins": int(bucket["window_local_contrast_wins"]),
                "speed_wins": int(bucket["speed_wins"]),
                "efficiency_pass_rate": float(bucket["efficiency_pass_count"]) / count,
                "mean_rank_score": float(bucket["sum_rank_score"]) / count,
                "mean_peak_in_truth_rate": float(bucket["sum_peak_in_truth_rate"]) / count,
                "mean_truth_window_hit_rate": float(bucket["sum_truth_window_hit_rate"]) / count,
                "mean_time_contrast_z": float(bucket["sum_time_contrast_z"]) / count,
                "mean_peak_global_band_contrast": float(bucket["sum_peak_global_band_contrast"]) / count,
                "mean_peak_period_concentration": float(bucket["sum_peak_period_concentration"]) / count,
                "mean_truth_window_local_band_contrast": float(bucket["sum_truth_window_local_band_contrast"]) / count,
                "mean_algorithm_seconds": float(bucket["sum_algorithm_seconds"]) / count,
            }
        )
    stability_rows.sort(
        key=lambda row: (
            float(row["mean_rank_score"]),
            int(row["rank_wins"]),
            int(row["time_contrast_wins"]),
        ),
        reverse=True,
    )
    overall_best_algorithm = str(stability_rows[0]["algorithm"]) if stability_rows else ""
    efficient_stability_rows = [row for row in stability_rows if float(row["efficiency_pass_rate"]) >= 1.0]
    overall_best_efficient_algorithm = str(efficient_stability_rows[0]["algorithm"]) if efficient_stability_rows else ""

    def leader_max(key: str, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""
        return str(max(rows, key=lambda row: float(row[key]))["algorithm"])

    def leader_min(key: str, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""
        return str(min(rows, key=lambda row: float(row[key]))["algorithm"])

    regime_summary_csv = suite_output_dir / "regime_summary.csv"
    stability_csv = suite_output_dir / "stability_summary.csv"
    suite_json = suite_output_dir / "suite_summary.json"
    _write_csv(regime_summary_csv, REGIME_SUMMARY_FIELDNAMES, regime_rows)
    _write_csv(stability_csv, STABILITY_FIELDNAMES, stability_rows)
    suite_payload = {
        "regime_count": len(selected_regimes),
        "overall_best_algorithm": overall_best_algorithm,
        "overall_best_efficient_algorithm": overall_best_efficient_algorithm,
        "best_single_regime_algorithm": best_single_regime_algorithm,
        "regime_rank_leaders": regime_rank_leaders,
        "background_mode_counts": suite_background_mode_counts,
        "background_source_counts": suite_background_source_counts,
        "regime_background_mode_counts": regime_background_mode_counts,
        "regime_background_source_counts": regime_background_source_counts,
        "metric_stability_leaders": {
            "mean_rank_score": overall_best_algorithm,
            "mean_efficient_rank_score": overall_best_efficient_algorithm,
            "mean_peak_in_truth_rate": leader_max("mean_peak_in_truth_rate", stability_rows),
            "mean_truth_window_hit_rate": leader_max("mean_truth_window_hit_rate", stability_rows),
            "mean_time_contrast_z": leader_max("mean_time_contrast_z", stability_rows),
            "mean_peak_global_band_contrast": leader_max("mean_peak_global_band_contrast", stability_rows),
            "mean_peak_period_concentration": leader_max("mean_peak_period_concentration", stability_rows),
            "mean_truth_window_local_band_contrast": leader_max("mean_truth_window_local_band_contrast", stability_rows),
            "mean_algorithm_seconds": leader_min("mean_algorithm_seconds", stability_rows),
        },
        "metric_win_leaders": {
            "rank_wins": leader_max("rank_wins", stability_rows),
            "efficient_rank_wins": leader_max("efficient_rank_wins", stability_rows),
            "time_contrast_wins": leader_max("time_contrast_wins", stability_rows),
            "global_band_contrast_wins": leader_max("global_band_contrast_wins", stability_rows),
            "period_concentration_wins": leader_max("period_concentration_wins", stability_rows),
            "window_local_contrast_wins": leader_max("window_local_contrast_wins", stability_rows),
            "speed_wins": leader_max("speed_wins", stability_rows),
        },
        "regimes": [asdict(regime) for regime in selected_regimes],
        "stability_rows": stability_rows,
    }
    suite_json.write_text(json.dumps(suite_payload, indent=2, ensure_ascii=True))
    return CompressionBenchmarkSuiteResult(
        output_dir=suite_output_dir,
        regime_summary_csv=regime_summary_csv,
        stability_csv=stability_csv,
        suite_json=suite_json,
        regime_count=len(selected_regimes),
        overall_best_algorithm=overall_best_algorithm,
        overall_best_efficient_algorithm=overall_best_efficient_algorithm,
    )
