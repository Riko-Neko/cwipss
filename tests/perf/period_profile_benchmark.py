"""Reproducible period-axis rank behind activity -> PELT stage boundaries."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
PERF_DIR = Path(__file__).resolve().parent
for search_path in (SRC_DIR, PERF_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from period_profile_algorithms import (  # noqa: E402
    PeriodProfileAlgorithm,
    evaluate_period_profile,
    period_profile_catalog,
)
from cwt_activity_algorithms import (  # noqa: E402
    CWTActivityAlgorithm,
    compute_cwt_activity,
    resolve_cwt_activity_algorithms,
)
from stage_boundaries import (  # noqa: E402
    pelt_parameters_from_config,
    segment_activity_with_pelt,
    stage3_windows,
    standardize_activity_for_pelt,
)
from cwipss.analysis.injection_config import load_injection_config, make_injections_from_config  # noqa: E402
from cwipss.analysis.simulation import InjectionSpec, inject_periodic_signal  # noqa: E402
from cwipss.config import load_cwt_config  # noqa: E402
from cwipss.data.readers import CE4_RECORD_LEN, open_spectrum_reader  # noqa: E402
from cwipss.signal.cpro import cpro_period_mask, difference_noise_std, impulse_cwt_noise_gain  # noqa: E402
from cwipss.signal.cwt import cwt_power_cube, period_grid_records  # noqa: E402


CASE_FIELDNAMES = [
    "case_id",
    "case_kind",
    "algorithm",
    "signal_model",
    "channel_index",
    "frequency_mhz",
    "record_start",
    "record_stop",
    "duration_records",
    "truth_period_records",
    "accepted",
    "peak_period_records",
    "period_error_fraction",
    "period_hit_05",
    "period_hit_10",
    "period_hit_20",
    "truth_inside_peak_band",
    "harmonic_confusion",
    "peak_strength",
    "integrated_strength",
    "width_bins",
    "period_start_records",
    "period_stop_records",
    "band_concentration",
    "band_persistence",
    "local_contrast",
    "harmonic_2_score",
    "harmonic_3_score",
    "harmonic_support_count",
    "base_score",
    "total_score",
    "algorithm_seconds",
]

SUMMARY_FIELDNAMES = [
    "rank",
    "algorithm",
    "scientific_gate_pass",
    "positive_case_count",
    "negative_case_count",
    "positive_accept_rate",
    "period_hit_05_rate",
    "period_hit_10_rate",
    "period_hit_20_rate",
    "accepted_period_hit_10_rate",
    "truth_inside_peak_band_rate",
    "mean_accepted_period_error_fraction",
    "harmonic_confusion_rate",
    "negative_false_accept_rate",
    "negative_reject_rate",
    "mean_positive_width_bins",
    "mean_positive_band_persistence",
    "mean_positive_harmonic_support",
    "mean_negative_harmonic_support",
    "mean_algorithm_seconds",
    "p95_algorithm_seconds",
    "rank_score",
]

MODEL_SUMMARY_FIELDNAMES = [
    "algorithm",
    "signal_model",
    "positive_case_count",
    "positive_accept_rate",
    "period_hit_05_rate",
    "period_hit_10_rate",
    "period_hit_20_rate",
    "accepted_period_hit_10_rate",
    "truth_inside_peak_band_rate",
    "mean_accepted_period_error_fraction",
    "harmonic_confusion_rate",
    "mean_harmonic_2_score",
    "mean_harmonic_3_score",
    "mean_harmonic_support_count",
]

STAGE1_FIELDNAMES = [
    "case_id",
    "signal_model",
    "channel_index",
    "frequency_mhz",
    "truth_period_records",
    "truth_record_start",
    "truth_record_stop",
    "stage1_status",
    "pelt_window_count",
    "stage3_window_count",
    "stage1_window_count",
    "matched_record_start",
    "matched_record_stop",
    "matched_overlap_fraction",
    "preprocess_seconds",
    "error",
]


@dataclass(frozen=True)
class PeriodProfileBenchmarkConfig:
    output_dir: Path = PROJECT_DIR / "runs/period_profile_rank_initial"
    input_path: Path | None = None
    input_dir: Path = PROJECT_DIR / "data/CE4"
    injection_config: Path = PROJECT_DIR / "configs/injection_lowfreq_random_100.json"
    cwt_config: Path = PROJECT_DIR / "configs/cwt_default.json"
    activity_algorithm: str = "sm_cpro_w769"
    candidate_period_max_records: float = 1000.0
    stage3_min_window_records: int = 96
    profile_threshold_snr: float = 32.0
    profile_texture_quantile: float = 0.9375
    algorithms: tuple[str, ...] = ()
    max_positive_cases: int = 0
    negative_channel_indices: tuple[int, ...] = (3, 4, 5, 6, 7, 8, 9, 10)
    max_negative_windows_per_channel: int = 0
    cwt_backend: str = "cpu"
    cuda_device: int = 0
    progress_every: int = 10


@dataclass(frozen=True)
class PeriodProfileBenchmarkResult:
    output_dir: Path
    cases_csv: Path
    summary_csv: Path
    summary_json: Path
    model_summary_csv: Path
    stage1_csv: Path
    algorithm_map_json: Path


@dataclass(frozen=True)
class _WindowCase:
    case_id: str
    case_kind: str
    signal_model: str
    channel_index: int
    frequency_mhz: float
    record_start: int
    record_stop: int
    truth_period_records: float
    periods: np.ndarray
    normalized_score_map: np.ndarray


def _largest_complete_2c(root: Path) -> Path:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".2c")
    complete = [
        path
        for path in files
        if path.stat().st_size > 0 and path.stat().st_size % CE4_RECORD_LEN == 0
    ]
    if not complete:
        raise FileNotFoundError(f"No complete CE4 .2C files found under: {root}")
    return max(complete, key=lambda path: path.stat().st_size)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _resolve_algorithms(names: tuple[str, ...]) -> tuple[PeriodProfileAlgorithm, ...]:
    catalog = {item.name: item for item in period_profile_catalog()}
    if not names:
        return tuple(catalog.values())
    missing = sorted(set(names) - set(catalog))
    if missing:
        raise ValueError(f"Unknown period-profile algorithms: {', '.join(missing)}")
    return tuple(catalog[name] for name in names)


def _resolve_activity_algorithm(name: str) -> CWTActivityAlgorithm:
    selected = resolve_cwt_activity_algorithms((name,))
    if len(selected) != 1:
        raise ValueError("period-profile rank requires exactly one frozen stage-1 algorithm")
    algorithm = selected[0]
    if algorithm.input_denoiser not in {"none", "absolute"}:
        raise ValueError("stage-1 algorithm must use only the target frequency channel")
    if algorithm.complexity != "O(P*T)":
        raise ValueError("stage-1 algorithm must have linear O(P*T) complexity")
    return algorithm


def _local_record_window(spec: InjectionSpec, total_records: int, minimum_margin: int) -> tuple[int, int]:
    signal_start = int(spec.record_start)
    signal_stop = signal_start + int(spec.duration_records or 0)
    signal_length = max(1, signal_stop - signal_start)
    margin = max(int(minimum_margin), int(math.ceil(0.5 * signal_length)))
    start = max(0, signal_start - margin)
    stop = min(int(total_records), signal_stop + margin)
    if stop <= start:
        raise ValueError(f"Invalid local window for {spec.injection_id}: {start}:{stop}")
    return start, stop


def _span_overlap(start_a: int, stop_a: int, start_b: int, stop_b: int) -> float:
    overlap = max(0, min(int(stop_a), int(stop_b)) - max(int(start_a), int(start_b)))
    truth_duration = max(1, int(stop_b) - int(start_b))
    return float(overlap / truth_duration)


def _profile_normalization_threshold(
    power: np.ndarray,
    *,
    noise_std: float,
    noise_gain: np.ndarray,
    threshold_snr: float,
    texture_quantile: float,
) -> float:
    if threshold_snr <= 0.0:
        raise ValueError("profile_threshold_snr must be positive")
    if not 0.0 <= texture_quantile < 1.0:
        raise ValueError("profile_texture_quantile must be in [0, 1)")
    denominator = float(noise_std) ** 2 * np.asarray(noise_gain, dtype=np.float64)[:, None]
    calibrated = np.asarray(power, dtype=np.float64) / np.maximum(
        denominator, np.finfo(np.float32).tiny
    )
    threshold = float(threshold_snr)
    if texture_quantile > 0.0:
        threshold = max(threshold, float(np.quantile(calibrated, texture_quantile)))
    return threshold


def _normalized_cwt_power(
    power: np.ndarray,
    *,
    noise_std: float,
    noise_gain: np.ndarray,
    calibrated_threshold: float,
) -> np.ndarray:
    denominator = float(noise_std) ** 2 * np.asarray(noise_gain, dtype=np.float64)[:, None]
    normalized = np.asarray(power, dtype=np.float64) / np.maximum(
        denominator * float(calibrated_threshold), np.finfo(np.float32).tiny
    )
    return np.asarray(normalized, dtype=np.float32)


def _stage1_power(
    series: np.ndarray,
    absolute_power: np.ndarray,
    *,
    algorithm: CWTActivityAlgorithm,
    periods: np.ndarray,
    wavelet: str,
    method: str,
    backend: str,
    cuda_device: int,
) -> np.ndarray:
    if algorithm.input_denoiser == "absolute":
        return np.asarray(absolute_power, dtype=np.float32)
    return np.asarray(
        cwt_power_cube(
            np.asarray(series, dtype=np.float32)[:, None],
            periods,
            wavelet=wavelet,
            normalize_channels=True,
            method=method,
            backend=backend,
            cuda_device=cuda_device,
        )[:, :, 0],
        dtype=np.float32,
    )


def _pelt_windows_for_power(
    series: np.ndarray,
    absolute_power: np.ndarray,
    *,
    algorithm: CWTActivityAlgorithm,
    periods: np.ndarray,
    noise_std: float,
    noise_gain: np.ndarray,
    pelt_parameters: Any,
    stage3_min_window_records: int,
    wavelet: str,
    method: str,
    backend: str,
    cuda_device: int,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    power = _stage1_power(
        series,
        absolute_power,
        algorithm=algorithm,
        periods=periods,
        wavelet=wavelet,
        method=method,
        backend=backend,
        cuda_device=cuda_device,
    )
    result = compute_cwt_activity(
        power,
        periods,
        algorithm,
        noise_std=noise_std,
        noise_gain=noise_gain,
    )
    activity = np.asarray(result.activity, dtype=np.float32)
    activity_z = standardize_activity_for_pelt(
        activity,
        absolute_calibrated=algorithm.method == "single_map_cpro_activity",
        native_absolute=(
            algorithm.input_denoiser == "absolute"
            and algorithm.method != "single_map_cpro_activity"
        ),
    )
    segmented = segment_activity_with_pelt(
        activity,
        pelt_parameters,
        activity_z=activity_z,
    )
    pelt_windows = [dict(window) for window in segmented.windows]
    accepted = list(
        stage3_windows(
            segmented.windows,
            minimum_duration_records=stage3_min_window_records,
        )
    )
    return pelt_windows, accepted


def _prepare_positive_case(
    *,
    reader: Any,
    spec: InjectionSpec,
    periods: np.ndarray,
    noise_gain: np.ndarray,
    wavelet: str,
    method: str,
    backend: str,
    cuda_device: int,
    activity_algorithm: CWTActivityAlgorithm,
    pelt_parameters: Any,
    stage3_min_window_records: int,
    profile_threshold_snr: float,
    profile_texture_quantile: float,
) -> tuple[_WindowCase | None, dict[str, Any]]:
    start_time = perf_counter()
    channel = min(max(int(round(float(spec.channel_center))), 0), int(reader.n_channels) - 1)
    local_start, local_stop = _local_record_window(
        spec,
        int(reader.n_records),
        minimum_margin=1,
    )
    baseline = np.asarray(
        reader.read_block(slice(local_start, local_stop), slice(channel, channel + 1)).data[:, 0],
        dtype=np.float32,
    )
    local_spec = replace(
        spec,
        record_start=int(spec.record_start) - local_start,
        channel_center=0.0,
        bandwidth_channels=1.0,
        drift_channels=0.0,
    )
    injected, truth = inject_periodic_signal(baseline[:, None], local_spec)
    absolute_power = cwt_power_cube(
        injected,
        periods,
        wavelet=wavelet,
        normalize_channels=False,
        method=method,
        backend=backend,
        cuda_device=cuda_device,
    )[:, :, 0]
    noise_std = difference_noise_std(baseline)
    pelt_windows, accepted_windows = _pelt_windows_for_power(
        np.asarray(injected[:, 0], dtype=np.float32),
        absolute_power,
        algorithm=activity_algorithm,
        periods=periods,
        noise_std=noise_std,
        noise_gain=noise_gain,
        pelt_parameters=pelt_parameters,
        stage3_min_window_records=stage3_min_window_records,
        wavelet=wavelet,
        method=method,
        backend=backend,
        cuda_device=cuda_device,
    )
    truth_start = int(truth["record_start"])
    truth_stop = int(truth["record_stop"])
    pelt_overlaps = [
        _span_overlap(
            int(window["record_start"]),
            int(window["record_stop"]),
            truth_start,
            truth_stop,
        )
        for window in pelt_windows
    ]
    overlaps = [
        _span_overlap(
            int(window["record_start"]),
            int(window["record_stop"]),
            truth_start,
            truth_stop,
        )
        for window in accepted_windows
    ]
    matched_index = int(np.argmax(overlaps)) if overlaps else -1
    matched_overlap = float(overlaps[matched_index]) if matched_index >= 0 else 0.0
    pelt_hit = bool(pelt_overlaps and float(np.max(pelt_overlaps)) > 0.0)
    status = "hit" if matched_overlap > 0.0 else ("duration_rejected" if pelt_hit else "miss")
    stage1_row = {
        "case_id": spec.injection_id,
        "signal_model": spec.signal_model,
        "channel_index": channel,
        "frequency_mhz": float(reader.freqs_mhz[channel]),
        "truth_period_records": float(spec.period_records),
        "truth_record_start": int(spec.record_start),
        "truth_record_stop": int(spec.record_start + int(spec.duration_records or 0)),
        "stage1_status": status,
        "pelt_window_count": len(pelt_windows),
        "stage3_window_count": len(accepted_windows),
        "stage1_window_count": len(accepted_windows),
        "matched_record_start": "",
        "matched_record_stop": "",
        "matched_overlap_fraction": matched_overlap,
        "preprocess_seconds": perf_counter() - start_time,
        "error": "",
    }
    if matched_index < 0 or matched_overlap <= 0.0:
        return None, stage1_row
    window = accepted_windows[matched_index]
    window_start = int(window["record_start"])
    window_stop = int(window["record_stop"])
    stage1_row["matched_record_start"] = local_start + window_start
    stage1_row["matched_record_stop"] = local_start + window_stop
    profile_threshold = _profile_normalization_threshold(
        absolute_power,
        noise_std=noise_std,
        noise_gain=noise_gain,
        threshold_snr=profile_threshold_snr,
        texture_quantile=profile_texture_quantile,
    )
    normalized = _normalized_cwt_power(
        absolute_power[:, window_start:window_stop],
        noise_std=noise_std,
        noise_gain=noise_gain,
        calibrated_threshold=profile_threshold,
    )
    return (
        _WindowCase(
            case_id=spec.injection_id,
            case_kind="positive",
            signal_model=spec.signal_model,
            channel_index=channel,
            frequency_mhz=float(reader.freqs_mhz[channel]),
            record_start=local_start + window_start,
            record_stop=local_start + window_stop,
            truth_period_records=float(spec.period_records),
            periods=periods,
            normalized_score_map=normalized,
        ),
        stage1_row,
    )


def _prepare_negative_cases(
    *,
    reader: Any,
    channel: int,
    periods: np.ndarray,
    noise_gain: np.ndarray,
    wavelet: str,
    method: str,
    backend: str,
    cuda_device: int,
    activity_algorithm: CWTActivityAlgorithm,
    pelt_parameters: Any,
    stage3_min_window_records: int,
    profile_threshold_snr: float,
    profile_texture_quantile: float,
    maximum_windows: int,
) -> list[_WindowCase]:
    series = np.asarray(
        reader.read_block(slice(0, reader.n_records), slice(channel, channel + 1)).data[:, 0],
        dtype=np.float32,
    )
    absolute_power = cwt_power_cube(
        series[:, None],
        periods,
        wavelet=wavelet,
        normalize_channels=False,
        method=method,
        backend=backend,
        cuda_device=cuda_device,
    )[:, :, 0]
    noise_std = difference_noise_std(series)
    _pelt_windows, windows = _pelt_windows_for_power(
        series,
        absolute_power,
        algorithm=activity_algorithm,
        periods=periods,
        noise_std=noise_std,
        noise_gain=noise_gain,
        pelt_parameters=pelt_parameters,
        stage3_min_window_records=stage3_min_window_records,
        wavelet=wavelet,
        method=method,
        backend=backend,
        cuda_device=cuda_device,
    )
    profile_threshold = _profile_normalization_threshold(
        absolute_power,
        noise_std=noise_std,
        noise_gain=noise_gain,
        threshold_snr=profile_threshold_snr,
        texture_quantile=profile_texture_quantile,
    )
    normalized = _normalized_cwt_power(
        absolute_power,
        noise_std=noise_std,
        noise_gain=noise_gain,
        calibrated_threshold=profile_threshold,
    )
    if maximum_windows > 0:
        windows = sorted(
            windows,
            key=lambda item: float(item.get("activity_max", 0.0)),
            reverse=True,
        )[:maximum_windows]
    return [
        _WindowCase(
            case_id=f"negative_ch{channel:04d}_w{index:04d}",
            case_kind="negative",
            signal_model="real_ce4_no_injection",
            channel_index=channel,
            frequency_mhz=float(reader.freqs_mhz[channel]),
            record_start=int(window["record_start"]),
            record_stop=int(window["record_stop"]),
            truth_period_records=math.nan,
            periods=periods,
            normalized_score_map=normalized[:, int(window["record_start"]):int(window["record_stop"])],
        )
        for index, window in enumerate(windows, start=1)
    ]


def _period_error(estimate: float, truth: float) -> float:
    return float(abs(float(estimate) - float(truth)) / max(abs(float(truth)), 1e-12))


def _harmonic_confusion(estimate: float, truth: float) -> int:
    direct_error = _period_error(estimate, truth)
    if direct_error <= 0.20:
        return 0
    aliases = (truth / 2.0, truth / 3.0, truth * 2.0, truth * 3.0)
    return int(any(_period_error(estimate, alias) <= 0.10 for alias in aliases))


def _evaluate_case(case: _WindowCase, algorithm: PeriodProfileAlgorithm) -> dict[str, Any]:
    start = perf_counter()
    result = evaluate_period_profile(case.normalized_score_map, case.periods, algorithm)
    seconds = perf_counter() - start
    positive = case.case_kind == "positive"
    period_error = _period_error(result.peak_period_records, case.truth_period_records) if positive else math.nan
    band_lo, band_hi = sorted((result.period_start_records, result.period_stop_records))
    accepted = int(result.accepted)
    return {
        "case_id": case.case_id,
        "case_kind": case.case_kind,
        "algorithm": algorithm.name,
        "signal_model": case.signal_model,
        "channel_index": case.channel_index,
        "frequency_mhz": case.frequency_mhz,
        "record_start": case.record_start,
        "record_stop": case.record_stop,
        "duration_records": case.record_stop - case.record_start,
        "truth_period_records": case.truth_period_records if positive else "",
        "accepted": accepted,
        "peak_period_records": result.peak_period_records,
        "period_error_fraction": period_error if positive else "",
        "period_hit_05": int(positive and result.accepted and period_error <= 0.05),
        "period_hit_10": int(positive and result.accepted and period_error <= 0.10),
        "period_hit_20": int(positive and result.accepted and period_error <= 0.20),
        "truth_inside_peak_band": int(
            positive and result.accepted and band_lo <= case.truth_period_records <= band_hi
        ),
        "harmonic_confusion": _harmonic_confusion(result.peak_period_records, case.truth_period_records)
        if positive and result.accepted
        else 0,
        "peak_strength": result.peak_strength,
        "integrated_strength": result.integrated_strength,
        "width_bins": result.width_bins,
        "period_start_records": result.period_start_records,
        "period_stop_records": result.period_stop_records,
        "band_concentration": result.band_concentration,
        "band_persistence": result.band_persistence,
        "local_contrast": result.local_contrast,
        "harmonic_2_score": result.harmonic_2_score,
        "harmonic_3_score": result.harmonic_3_score,
        "harmonic_support_count": result.harmonic_support_count,
        "base_score": result.base_score,
        "total_score": result.total_score,
        "algorithm_seconds": seconds,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(row[key]) for row in rows], dtype=np.float64))


def _summaries(case_rows: list[dict[str, Any]], algorithms: tuple[PeriodProfileAlgorithm, ...]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for algorithm in algorithms:
        rows = [row for row in case_rows if row["algorithm"] == algorithm.name]
        positives = [row for row in rows if row["case_kind"] == "positive"]
        negatives = [row for row in rows if row["case_kind"] == "negative"]
        accepted_positives = [row for row in positives if int(row["accepted"])]
        accepted_errors = [float(row["period_error_fraction"]) for row in accepted_positives]
        positive_accept = _mean(positives, "accepted")
        hit05 = _mean(positives, "period_hit_05")
        hit10 = _mean(positives, "period_hit_10")
        hit20 = _mean(positives, "period_hit_20")
        band_coverage = _mean(positives, "truth_inside_peak_band")
        confusion = _mean(positives, "harmonic_confusion")
        false_accept = _mean(negatives, "accepted")
        negative_reject = 1.0 - false_accept if negatives else 0.0
        accepted_hit10 = (
            float(np.mean([float(row["period_hit_10"]) for row in accepted_positives]))
            if accepted_positives
            else 0.0
        )
        rank_score = (
            0.35 * hit10
            + 0.15 * hit20
            + 0.15 * positive_accept
            + 0.20 * negative_reject
            + 0.10 * band_coverage
            + 0.05 * (1.0 - confusion)
        )
        scientific_gate_pass = int(
            algorithm.min_width_bins >= 2
            and algorithm.min_peak_strength > 0.0
            and positive_accept >= 0.35
            and accepted_hit10 >= 0.80
            and false_accept <= 0.50
        )
        times = np.asarray([float(row["algorithm_seconds"]) for row in rows], dtype=np.float64)
        summaries.append(
            {
                "algorithm": algorithm.name,
                "scientific_gate_pass": scientific_gate_pass,
                "positive_case_count": len(positives),
                "negative_case_count": len(negatives),
                "positive_accept_rate": positive_accept,
                "period_hit_05_rate": hit05,
                "period_hit_10_rate": hit10,
                "period_hit_20_rate": hit20,
                "accepted_period_hit_10_rate": accepted_hit10,
                "truth_inside_peak_band_rate": band_coverage,
                "mean_accepted_period_error_fraction": float(np.mean(accepted_errors))
                if accepted_errors
                else math.nan,
                "harmonic_confusion_rate": confusion,
                "negative_false_accept_rate": false_accept,
                "negative_reject_rate": negative_reject,
                "mean_positive_width_bins": _mean(positives, "width_bins"),
                "mean_positive_band_persistence": _mean(positives, "band_persistence"),
                "mean_positive_harmonic_support": _mean(positives, "harmonic_support_count"),
                "mean_negative_harmonic_support": _mean(negatives, "harmonic_support_count"),
                "mean_algorithm_seconds": float(np.mean(times)) if times.size else 0.0,
                "p95_algorithm_seconds": float(np.quantile(times, 0.95)) if times.size else 0.0,
                "rank_score": rank_score,
            }
        )
    summaries.sort(
        key=lambda row: (
            int(row["scientific_gate_pass"]),
            float(row["rank_score"]),
            float(row["period_hit_10_rate"]),
            -float(row["negative_false_accept_rate"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(summaries, start=1):
        row["rank"] = rank
    return summaries


def _model_summaries(
    case_rows: list[dict[str, Any]],
    algorithms: tuple[PeriodProfileAlgorithm, ...],
) -> list[dict[str, Any]]:
    models = sorted(
        {
            str(row["signal_model"])
            for row in case_rows
            if row["case_kind"] == "positive"
        }
    )
    summaries: list[dict[str, Any]] = []
    for algorithm in algorithms:
        for model in models:
            rows = [
                row
                for row in case_rows
                if row["algorithm"] == algorithm.name
                and row["case_kind"] == "positive"
                and row["signal_model"] == model
            ]
            if not rows:
                continue
            accepted = [row for row in rows if int(row["accepted"])]
            errors = [float(row["period_error_fraction"]) for row in accepted]
            summaries.append(
                {
                    "algorithm": algorithm.name,
                    "signal_model": model,
                    "positive_case_count": len(rows),
                    "positive_accept_rate": _mean(rows, "accepted"),
                    "period_hit_05_rate": _mean(rows, "period_hit_05"),
                    "period_hit_10_rate": _mean(rows, "period_hit_10"),
                    "period_hit_20_rate": _mean(rows, "period_hit_20"),
                    "accepted_period_hit_10_rate": _mean(accepted, "period_hit_10"),
                    "truth_inside_peak_band_rate": _mean(rows, "truth_inside_peak_band"),
                    "mean_accepted_period_error_fraction": float(np.mean(errors))
                    if errors
                    else math.nan,
                    "harmonic_confusion_rate": _mean(rows, "harmonic_confusion"),
                    "mean_harmonic_2_score": _mean(rows, "harmonic_2_score"),
                    "mean_harmonic_3_score": _mean(rows, "harmonic_3_score"),
                    "mean_harmonic_support_count": _mean(rows, "harmonic_support_count"),
                }
            )
    return summaries


def run_period_profile_benchmark(config: PeriodProfileBenchmarkConfig) -> PeriodProfileBenchmarkResult:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    algorithms = _resolve_algorithms(config.algorithms)
    activity_algorithm = _resolve_activity_algorithm(config.activity_algorithm)
    if int(config.stage3_min_window_records) < 1:
        raise ValueError("stage3_min_window_records must be positive")
    input_path = Path(config.input_path) if config.input_path else _largest_complete_2c(Path(config.input_dir))
    reader = open_spectrum_reader(input_path)
    search = load_cwt_config(
        config.cwt_config,
        overrides={"cwt_backend": config.cwt_backend, "cuda_device": config.cuda_device},
    )
    full_periods = period_grid_records(
        search.period_min_records,
        search.period_max_records,
        search.period_count,
        search.period_spacing,
    )
    period_mask = cpro_period_mask(
        full_periods,
        search.candidate_period_min_records,
        config.candidate_period_max_records,
    )
    periods = full_periods[period_mask]
    noise_gain = impulse_cwt_noise_gain(periods, wavelet=search.wavelet, method=search.cwt_method)
    pelt_parameters = pelt_parameters_from_config(search)
    specs = make_injections_from_config(
        load_injection_config(config.injection_config),
        records=reader.n_records,
        channels=reader.n_channels,
        freqs_mhz=reader.freqs_mhz,
    )
    eligible = [
        spec
        for spec in specs
        if float(periods[0]) <= float(spec.period_records) <= float(periods[-1])
    ]
    if config.max_positive_cases > 0:
        eligible = eligible[: config.max_positive_cases]

    cases: list[_WindowCase] = []
    stage1_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(eligible, start=1):
        try:
            case, stage1 = _prepare_positive_case(
                reader=reader,
                spec=spec,
                periods=periods,
                noise_gain=noise_gain,
                wavelet=search.wavelet,
                method=search.cwt_method,
                backend=config.cwt_backend,
                cuda_device=config.cuda_device,
                activity_algorithm=activity_algorithm,
                pelt_parameters=pelt_parameters,
                stage3_min_window_records=config.stage3_min_window_records,
                profile_threshold_snr=config.profile_threshold_snr,
                profile_texture_quantile=config.profile_texture_quantile,
            )
        except Exception as exc:
            stage1 = {
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
            case = None
        stage1_rows.append(stage1)
        if case is not None:
            cases.append(case)
        if config.progress_every > 0 and index % config.progress_every == 0:
            print(f"[period-profile] positive {index}/{len(eligible)}", flush=True)

    for index, channel in enumerate(config.negative_channel_indices, start=1):
        if not 0 <= int(channel) < int(reader.n_channels):
            raise ValueError(f"negative channel out of range: {channel}")
        negative_cases = _prepare_negative_cases(
            reader=reader,
            channel=int(channel),
            periods=periods,
            noise_gain=noise_gain,
            wavelet=search.wavelet,
            method=search.cwt_method,
            backend=config.cwt_backend,
            cuda_device=config.cuda_device,
            activity_algorithm=activity_algorithm,
            pelt_parameters=pelt_parameters,
            stage3_min_window_records=config.stage3_min_window_records,
            profile_threshold_snr=config.profile_threshold_snr,
            profile_texture_quantile=config.profile_texture_quantile,
            maximum_windows=config.max_negative_windows_per_channel,
        )
        cases.extend(negative_cases)
        if config.progress_every > 0 and (
            index % config.progress_every == 0 or index == len(config.negative_channel_indices)
        ):
            print(
                f"[period-profile] negative channel {channel} "
                f"({index}/{len(config.negative_channel_indices)}): "
                f"{len(negative_cases)} PELT windows entered stage 3",
                flush=True,
            )

    if not any(case.case_kind == "positive" for case in cases):
        raise RuntimeError("No stage-1-positive injection windows were available for period-profile ranking")
    if not any(case.case_kind == "negative" for case in cases):
        raise RuntimeError("No real PELT windows passed the stage-3 duration gate")

    # Rotate the first evaluator so cold-cache reads of each 2-D map are shared
    # evenly. A fixed order otherwise makes the first algorithm appear slower.
    case_rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        offset = case_index % len(algorithms)
        balanced_order = algorithms[offset:] + algorithms[:offset]
        case_rows.extend(_evaluate_case(case, algorithm) for algorithm in balanced_order)
    summaries = _summaries(case_rows, algorithms)
    cases_csv = output_dir / "period_profile_cases.csv"
    summary_csv = output_dir / "period_profile_summary.csv"
    model_summary_csv = output_dir / "period_profile_model_summary.csv"
    stage1_csv = output_dir / "stage1_cases.csv"
    summary_json = output_dir / "period_profile_summary.json"
    algorithm_map_json = output_dir / "period_profile_algorithm_map.json"
    _write_csv(cases_csv, CASE_FIELDNAMES, case_rows)
    _write_csv(summary_csv, SUMMARY_FIELDNAMES, summaries)
    _write_csv(model_summary_csv, MODEL_SUMMARY_FIELDNAMES, _model_summaries(case_rows, algorithms))
    _write_csv(stage1_csv, STAGE1_FIELDNAMES, stage1_rows)
    algorithm_map_json.write_text(
        json.dumps({item.name: item.to_dict() for item in algorithms}, indent=2, ensure_ascii=True)
    )
    payload = {
        "schema_version": 2,
        "benchmark": "pelt_window_period_profile_rank",
        "config": {
            **asdict(config),
            "output_dir": str(output_dir),
            "input_path": str(input_path),
            "input_dir": str(config.input_dir),
            "injection_config": str(config.injection_config),
            "cwt_config": str(config.cwt_config),
        },
        "reproducibility": {
            "input_sha256": _sha256(input_path),
            "injection_config_sha256": _sha256(Path(config.injection_config)),
            "cwt_config_sha256": _sha256(Path(config.cwt_config)),
        },
        "period_domain_records": [float(periods[0]), float(periods[-1])],
        "stage_boundaries": {
            "stage1_algorithm": activity_algorithm.name,
            "stage1_input": "single-channel CWT period x time power",
            "stage1_output": "one-dimensional time activity only",
            "stage2": "native C++ PELT",
            "pelt_parameters": asdict(pelt_parameters),
            "stage3_min_window_records": int(config.stage3_min_window_records),
            "stage3_input": "unmasked absolute CWT power sliced only by accepted PELT indices",
            "profile_threshold_snr": float(config.profile_threshold_snr),
            "profile_texture_quantile": float(config.profile_texture_quantile),
        },
        "eligible_injection_count": len(eligible),
        "stage1_hit_count": sum(row["stage1_status"] == "hit" for row in stage1_rows),
        "stage1_miss_count": sum(row["stage1_status"] == "miss" for row in stage1_rows),
        "stage1_duration_rejected_count": sum(
            row["stage1_status"] == "duration_rejected" for row in stage1_rows
        ),
        "stage1_error_count": sum(row["stage1_status"] == "error" for row in stage1_rows),
        "positive_window_count": sum(case.case_kind == "positive" for case in cases),
        "negative_window_count": sum(case.case_kind == "negative" for case in cases),
        "algorithm_count": len(algorithms),
        "algorithm_timing_order": "deterministic_per_case_rotation",
        "rank_formula": (
            "0.35*hit10 + 0.15*hit20 + 0.15*positive_accept + "
            "0.20*negative_reject + 0.10*truth_band_coverage + 0.05*(1-harmonic_confusion)"
        ),
        "scientific_gate": (
            "main min_width_bins>=2 and min_peak_strength>0; positive_accept>=0.35; "
            "accepted_hit10>=0.80; real_negative_false_accept<=0.50"
        ),
        "top10": summaries[:10],
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str))
    return PeriodProfileBenchmarkResult(
        output_dir=output_dir,
        cases_csv=cases_csv,
        summary_csv=summary_csv,
        summary_json=summary_json,
        model_summary_csv=model_summary_csv,
        stage1_csv=stage1_csv,
        algorithm_map_json=algorithm_map_json,
    )
