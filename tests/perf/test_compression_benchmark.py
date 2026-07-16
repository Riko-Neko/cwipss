from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from compression_benchmark import (
    DEFAULT_TOP10_ALGORITHMS,
    CompressionBenchmarkConfig,
    CompressionRegime,
    _largest_complete_2c,
    default_regimes,
    run_random_compression_benchmark,
    run_stratified_compression_benchmark,
    _resolve_algorithms,
)
from compression_config_rank import (
    ConfiguredCompressionRun,
    largest_complete_2c,
    run_configured_compression_rank,
)
from cwt_activity_algorithms import (
    DEFAULT_CWT_ACTIVITY_ALGORITHMS,
    cwt_activity_algorithm_map,
    resolve_cwt_activity_algorithms,
)
from cwt_activity_rank import (
    CWTActivityRun,
    run_cwt_activity_rank,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or not value.strip() else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or not value.strip() else float(value)


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_optional_path(name: str) -> Path | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return Path(value.strip())


def _ce4_input_or_none() -> Path | None:
    explicit = _env_optional_path("CWIPSS_TEST_CE4_INPUT")
    if explicit is not None:
        return explicit if explicit.is_absolute() else PROJECT_DIR / explicit
    try:
        return _largest_complete_2c(PROJECT_DIR / "data" / "CE4")
    except FileNotFoundError:
        return None


def _requested_background_modes(name: str, default: list[str]) -> tuple[str, ...]:
    requested = _env_list(name, default)
    resolved: list[str] = []
    for mode in requested:
        for candidate in (["synthetic", "ce4"] if mode.strip().lower() == "mixed" else [mode.strip().lower()]):
            if candidate and candidate not in resolved:
                resolved.append(candidate)
    return tuple(resolved or ["synthetic"])


def test_builtin_compression_catalog_contains_ten_plus_algorithms() -> None:
    algorithms = _resolve_algorithms(("all",))

    assert len(algorithms) >= 72
    assert "trimmed_mean_05_95_s16" in {algorithm.name for algorithm in algorithms}
    assert "quantile_p95_s1" in {algorithm.name for algorithm in algorithms}
    assert "quantile_p98_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_contrast_w7_s4" in {algorithm.name for algorithm in algorithms}
    assert "band_contrast_w1_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_contrast_w3_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_contrast_w5_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_contrast_w7_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_max_w7_s4" in {algorithm.name for algorithm in algorithms}
    assert "band_max_w7_s1" in {algorithm.name for algorithm in algorithms}
    assert "softmax_pool_t050_s4" in {algorithm.name for algorithm in algorithms}
    assert "softmax_pool_t050_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_local_contrast_w3_s4" in {algorithm.name for algorithm in algorithms}
    assert "band_local_contrast_w1_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_local_contrast_w1_c1_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_local_contrast_w1_c4_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_local_contrast_w5_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_hybrid_w3_s4" in {algorithm.name for algorithm in algorithms}
    assert "band_hybrid_w1_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_hybrid_w5_s1" in {algorithm.name for algorithm in algorithms}
    assert "topk_ratio_k3_s8" in {algorithm.name for algorithm in algorithms}
    assert "topk_ratio_k3_s1" in {algorithm.name for algorithm in algorithms}
    assert "topk_mean_k7_s1" in {algorithm.name for algorithm in algorithms}
    assert "l4_norm_s1" in {algorithm.name for algorithm in algorithms}
    assert "max_ratio_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_hybrid_w5_s4" in {algorithm.name for algorithm in algorithms}
    assert "band_local_ratio_w1_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_local_ratio_w1_c1_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_local_ratio_w1_c4_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_local_zscore_w1_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_local_zscore_w1_c4_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_local_zratio_w1_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_local_zratio_w1_c4_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_zscore_w1_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_zscore_w3_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_zscore_w5_s4" in {algorithm.name for algorithm in algorithms}
    assert "band_zscore_w5_s1" in {algorithm.name for algorithm in algorithms}
    assert "band_zscore_w7_s1" in {algorithm.name for algorithm in algorithms}


def test_cwt_activity_catalog_contains_ten_plus_independent_algorithms() -> None:
    algorithms = resolve_cwt_activity_algorithms(("all",))
    names = {algorithm.name for algorithm in algorithms}
    families = {algorithm.family for algorithm in algorithms}
    algorithm_map = cwt_activity_algorithm_map()

    assert len(algorithms) >= 10
    assert len(families) >= 10
    assert set(DEFAULT_CWT_ACTIVITY_ALGORITHMS).issubset(names)
    assert set(DEFAULT_CWT_ACTIVITY_ALGORITHMS).issubset(algorithm_map)
    calibrated = algorithm_map["post_freq_max8_center_8cycle_s70_f20"]
    assert calibrated["input_denoiser"] == "post_cwt_neighbor8"
    assert calibrated["complexity"] == "O(K*P*T)"
    assert calibrated["params"]["map_normalization"] == "center"
    assert {
        "horizontal_matched_filter",
        "viterbi_ridge_path",
        "spectral_kurtosis_window",
        "svd_rank1_projection",
        "low_rank_residual_max",
        "row_mad_viterbi_path",
        "row_mad_svd_rank1",
        "row_mad_low_rank_residual",
    }.isdisjoint(DEFAULT_CWT_ACTIVITY_ALGORITHMS)
    assert all(algorithm.source_refs for algorithm in algorithms)
    assert all("O(" in algorithm.complexity for algorithm in algorithms)


def test_random_compression_benchmark_writes_ranked_outputs(tmp_path: Path) -> None:
    result = run_random_compression_benchmark(
        CompressionBenchmarkConfig(
            output_dir=tmp_path / "compression_benchmark",
            case_count=2,
            seed=7,
            records_min=192,
            records_max=224,
            channels_min=16,
            channels_max=32,
            noise_std_min=1.0,
            noise_std_max=1.0,
            injection_period_min_records=10.0,
            injection_period_max_records=48.0,
            injection_duration_fraction_min=0.4,
            injection_duration_fraction_max=0.9,
            amplitude_factor_min=0.8,
            amplitude_factor_max=1.4,
            period_min_records=2.0,
            period_max_records=64.0,
            period_count=16,
            candidate_period_min_records=8.0,
            candidate_period_max_records=48.0,
            structure_time_support_records=16,
            pelt_penalty=5.0,
            pelt_min_size_records=24,
            pelt_jump_records=1,
            window_min_duration_records=24,
            window_min_activity_mean=0.0,
            window_merge_gap_records=8,
            progress_every=0,
        )
    )

    assert result.cases_csv.exists()
    assert result.summary_csv.exists()
    assert result.summary_json.exists()
    assert result.algorithm_count >= 10
    assert result.case_count == 2

    with result.cases_csv.open(newline="") as fp:
        case_rows = list(csv.DictReader(fp))
    assert len(case_rows) == result.algorithm_count * result.case_count
    assert all(row["algorithm"] for row in case_rows)
    assert all(row["algorithm_seconds"] for row in case_rows)
    assert all("signal_model" in row and row["signal_model"] for row in case_rows)

    with result.summary_csv.open(newline="") as fp:
        summary_rows = list(csv.DictReader(fp))
    assert len(summary_rows) == result.algorithm_count
    assert float(summary_rows[0]["rank_score"]) >= float(summary_rows[-1]["rank_score"])
    assert float(summary_rows[0]["mean_algorithm_seconds"]) >= 0.0

    payload = json.loads(result.summary_json.read_text())
    assert payload["best_algorithm"] == summary_rows[0]["algorithm"]
    assert payload["algorithm_count"] == result.algorithm_count
    assert payload["case_count"] == result.case_count
    assert payload["metric_leaders"]["rank_score"] == payload["best_algorithm"]
    assert "best_efficient_algorithm" in payload


def test_random_compression_benchmark_accepts_ce4_background(tmp_path: Path) -> None:
    ce4_input = _ce4_input_or_none()
    if ce4_input is None:
        pytest.skip("no complete CE4 .2C file available for CE4-backed benchmark smoke")

    result = run_random_compression_benchmark(
        CompressionBenchmarkConfig(
            output_dir=tmp_path / "compression_benchmark_ce4",
            case_count=1,
            seed=13,
            records_min=96,
            records_max=128,
            channels_min=8,
            channels_max=16,
            background_modes=("ce4",),
            ce4_input_path=ce4_input,
            injection_period_min_records=10.0,
            injection_period_max_records=48.0,
            injection_duration_fraction_min=0.4,
            injection_duration_fraction_max=0.9,
            amplitude_factor_min=0.4,
            amplitude_factor_max=1.0,
            period_min_records=2.0,
            period_max_records=64.0,
            period_count=16,
            candidate_period_min_records=8.0,
            candidate_period_max_records=48.0,
            structure_time_support_records=16,
            pelt_penalty=5.0,
            pelt_min_size_records=24,
            pelt_jump_records=1,
            window_min_duration_records=24,
            window_min_activity_mean=0.0,
            window_merge_gap_records=8,
            progress_every=0,
        )
    )

    with result.cases_csv.open(newline="") as fp:
        case_rows = list(csv.DictReader(fp))
    assert case_rows
    assert all(row["background_mode"] == "ce4" for row in case_rows)
    assert all(str(ce4_input) in row["background_source"] for row in case_rows)

    payload = json.loads(result.summary_json.read_text())
    assert payload["background_mode_counts"]["ce4"] == 1
    assert str(ce4_input) in payload["background_source_counts"]


def test_stratified_compression_benchmark_writes_regime_and_stability_outputs(tmp_path: Path) -> None:
    suite = run_stratified_compression_benchmark(
        base_config=CompressionBenchmarkConfig(
            output_dir=tmp_path / "compression_suite",
            case_count=1,
            seed=11,
            records_min=160,
            records_max=224,
            channels_min=16,
            channels_max=32,
            noise_std_min=1.0,
            noise_std_max=1.0,
            injection_period_min_records=10.0,
            injection_period_max_records=48.0,
            injection_duration_fraction_min=0.4,
            injection_duration_fraction_max=0.9,
            amplitude_factor_min=0.8,
            amplitude_factor_max=1.4,
            period_min_records=2.0,
            period_max_records=64.0,
            period_count=16,
            candidate_period_min_records=8.0,
            candidate_period_max_records=48.0,
            structure_time_support_records=16,
            pelt_penalty=5.0,
            pelt_min_size_records=24,
            pelt_jump_records=1,
            window_min_duration_records=24,
            window_min_activity_mean=0.0,
            window_merge_gap_records=8,
            progress_every=0,
        ),
        regimes=(
            CompressionRegime(
                name="small_smoke",
                description="Small smoke regime",
                case_count=1,
                records_min=160,
                records_max=192,
                channels_min=16,
                channels_max=24,
                injection_period_max_records=32.0,
                period_max_records=48.0,
                period_count=12,
                candidate_period_max_records=32.0,
                structure_time_support_records=12,
                pelt_min_size_records=20,
                window_min_duration_records=20,
                window_merge_gap_records=6,
            ),
            CompressionRegime(
                name="medium_smoke",
                description="Medium smoke regime",
                case_count=1,
                records_min=192,
                records_max=224,
                channels_min=24,
                channels_max=32,
                injection_period_max_records=48.0,
                period_max_records=64.0,
                period_count=16,
                candidate_period_max_records=48.0,
                structure_time_support_records=16,
                pelt_min_size_records=24,
                window_min_duration_records=24,
                window_merge_gap_records=8,
            ),
        ),
    )

    assert suite.regime_summary_csv.exists()
    assert suite.stability_csv.exists()
    assert suite.suite_json.exists()
    assert suite.regime_count == 2
    assert suite.overall_best_algorithm
    assert suite.overall_best_efficient_algorithm
    payload = json.loads(suite.suite_json.read_text())
    assert "metric_stability_leaders" in payload
    assert "metric_win_leaders" in payload
    assert "background_mode_counts" in payload
    assert "regime_background_mode_counts" in payload
    assert payload["metric_stability_leaders"]["mean_rank_score"] == suite.overall_best_algorithm
    assert payload["metric_stability_leaders"]["mean_efficient_rank_score"] == suite.overall_best_efficient_algorithm

    with suite.regime_summary_csv.open(newline="") as fp:
        regime_rows = list(csv.DictReader(fp))
    assert len({row["regime"] for row in regime_rows}) == 2
    assert all("efficiency_pass" in row for row in regime_rows)

    with suite.stability_csv.open(newline="") as fp:
        stability_rows = list(csv.DictReader(fp))
    assert stability_rows
    assert float(stability_rows[0]["mean_rank_score"]) >= float(stability_rows[-1]["mean_rank_score"])
    assert all("efficiency_pass_rate" in row for row in stability_rows)


@pytest.mark.perf
def test_random_compression_benchmark_perf_sweep() -> None:
    if not _env_flag("CWIPSS_RUN_PERF", default=False):
        pytest.skip("set CWIPSS_RUN_PERF=1 to execute the large randomized compression sweep")

    output_dir_env = os.getenv("CWIPSS_PERF_OUTPUT_DIR", "").strip()
    output_dir = Path(output_dir_env) if output_dir_env else PROJECT_DIR / "runs" / "pytest_compression_perf"
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    background_modes = _requested_background_modes("CWIPSS_PERF_BACKGROUND_MODES", ["synthetic"])
    ce4_input = _env_optional_path("CWIPSS_PERF_CE4_INPUT")
    if ce4_input is not None and not ce4_input.is_absolute():
        ce4_input = PROJECT_DIR / ce4_input
    if "ce4" in background_modes and ce4_input is None and _ce4_input_or_none() is None:
        pytest.skip("CE4-backed random sweep requested but no complete CE4 .2C file is available")

    result = run_random_compression_benchmark(
        CompressionBenchmarkConfig(
            output_dir=output_dir,
            case_count=_env_int("CWIPSS_PERF_CASES", 100),
            seed=_env_int("CWIPSS_PERF_SEED", 12345),
            records_min=_env_int("CWIPSS_PERF_RECORDS_MIN", 512),
            records_max=_env_int("CWIPSS_PERF_RECORDS_MAX", 4096),
            channels_min=_env_int("CWIPSS_PERF_CHANNELS_MIN", 64),
            channels_max=_env_int("CWIPSS_PERF_CHANNELS_MAX", 2048),
            noise_std_min=_env_float("CWIPSS_PERF_NOISE_STD_MIN", 1.0),
            noise_std_max=_env_float("CWIPSS_PERF_NOISE_STD_MAX", 1.0),
            background_modes=background_modes,
            ce4_input_path=ce4_input,
            ce4_input_dir=Path(os.getenv("CWIPSS_PERF_CE4_INPUT_DIR", str(PROJECT_DIR / "data" / "CE4"))),
            ce4_f_start_mhz=_env_float("CWIPSS_PERF_CE4_F_START", 0.0)
            if os.getenv("CWIPSS_PERF_CE4_F_START", "").strip()
            else None,
            ce4_f_stop_mhz=_env_float("CWIPSS_PERF_CE4_F_STOP", 0.0)
            if os.getenv("CWIPSS_PERF_CE4_F_STOP", "").strip()
            else None,
            signal_models=tuple(
                _env_list(
                    "CWIPSS_PERF_SIGNAL_MODELS",
                    [
                        "single_channel_periodic",
                        "pulsed_periodic",
                        "intermittent_periodic",
                        "sinusoidal_narrowband",
                        "band_limited_periodic",
                        "drifting_ridge",
                    ],
                )
            ),
            bandwidth_channels_min=_env_float("CWIPSS_PERF_BANDWIDTH_MIN", 1.0),
            bandwidth_channels_max=_env_float("CWIPSS_PERF_BANDWIDTH_MAX", 9.0),
            drift_channels_min=_env_float("CWIPSS_PERF_DRIFT_MIN", 0.0),
            drift_channels_max=_env_float("CWIPSS_PERF_DRIFT_MAX", 6.0),
            injection_period_min_records=_env_float("CWIPSS_PERF_INJ_PERIOD_MIN", 12.0),
            injection_period_max_records=_env_float("CWIPSS_PERF_INJ_PERIOD_MAX", 256.0),
            injection_duration_fraction_min=_env_float("CWIPSS_PERF_DURATION_FRAC_MIN", 0.35),
            injection_duration_fraction_max=_env_float("CWIPSS_PERF_DURATION_FRAC_MAX", 1.0),
            amplitude_factor_min=_env_float("CWIPSS_PERF_AMP_FACTOR_MIN", 0.25),
            amplitude_factor_max=_env_float("CWIPSS_PERF_AMP_FACTOR_MAX", 1.50),
            duty_cycle_min=_env_float("CWIPSS_PERF_DUTY_MIN", 0.05),
            duty_cycle_max=_env_float("CWIPSS_PERF_DUTY_MAX", 0.20),
            period_min_records=_env_float("CWIPSS_PERF_PERIOD_MIN", 2.0),
            period_max_records=_env_float("CWIPSS_PERF_PERIOD_MAX", 512.0),
            period_count=_env_int("CWIPSS_PERF_PERIOD_COUNT", 96),
            candidate_period_min_records=_env_float("CWIPSS_PERF_CAND_PERIOD_MIN", 10.0),
            candidate_period_max_records=_env_float("CWIPSS_PERF_CAND_PERIOD_MAX", 200.0),
            structure_time_support_records=_env_int("CWIPSS_PERF_STRUCTURE_TIME_SUPPORT", 64),
            pelt_penalty=_env_float("CWIPSS_PERF_PELT_PENALTY", 16.0),
            pelt_min_size_records=_env_int("CWIPSS_PERF_PELT_MIN_SIZE", 384),
            window_min_duration_records=_env_int("CWIPSS_PERF_WINDOW_MIN_DURATION", 384),
            window_min_activity_mean=_env_float("CWIPSS_PERF_WINDOW_MIN_MEAN", 0.05),
            window_merge_gap_records=_env_int("CWIPSS_PERF_WINDOW_MERGE_GAP", 256),
            algorithms=tuple(_env_list("CWIPSS_PERF_ALGORITHMS", ["all"])),
            progress_every=_env_int("CWIPSS_PERF_PROGRESS_EVERY", 10),
        )
    )

    payload = json.loads(result.summary_json.read_text())
    leaders = payload["metric_leaders"]
    top_rows = payload["summary_rows"][:5]
    print("best algorithm:", payload["best_algorithm"])
    print("metric leaders:", leaders)
    print("top 5 algorithms:", [row["algorithm"] for row in top_rows])

    assert result.summary_csv.exists()
    assert result.summary_json.exists()
    assert result.algorithm_count >= 10
    assert result.case_count == _env_int("CWIPSS_PERF_CASES", 100)


@pytest.mark.perf
def test_stratified_compression_benchmark_perf_suite() -> None:
    if not _env_flag("CWIPSS_RUN_PERF", default=False):
        pytest.skip("set CWIPSS_RUN_PERF=1 to execute the large stratified compression suite")

    output_dir_env = os.getenv("CWIPSS_PERF_SUITE_OUTPUT_DIR", "").strip()
    output_dir = Path(output_dir_env) if output_dir_env else PROJECT_DIR / "runs" / "pytest_compression_perf_suite"
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    background_modes = _requested_background_modes("CWIPSS_PERF_SUITE_BACKGROUND_MODES", ["synthetic"])
    ce4_input = _env_optional_path("CWIPSS_PERF_SUITE_CE4_INPUT")
    if ce4_input is not None and not ce4_input.is_absolute():
        ce4_input = PROJECT_DIR / ce4_input
    if "ce4" in background_modes and ce4_input is None and _ce4_input_or_none() is None:
        pytest.skip("CE4-backed suite requested but no complete CE4 .2C file is available")

    base = CompressionBenchmarkConfig(
        output_dir=output_dir,
        case_count=_env_int("CWIPSS_PERF_SUITE_CASES", 24),
        seed=_env_int("CWIPSS_PERF_SUITE_SEED", 12345),
        noise_std_min=_env_float("CWIPSS_PERF_SUITE_NOISE_STD_MIN", 1.0),
        noise_std_max=_env_float("CWIPSS_PERF_SUITE_NOISE_STD_MAX", 1.0),
        background_modes=background_modes,
        ce4_input_path=ce4_input,
        ce4_input_dir=Path(os.getenv("CWIPSS_PERF_SUITE_CE4_INPUT_DIR", str(PROJECT_DIR / "data" / "CE4"))),
        ce4_f_start_mhz=_env_float("CWIPSS_PERF_SUITE_CE4_F_START", 0.0)
        if os.getenv("CWIPSS_PERF_SUITE_CE4_F_START", "").strip()
        else None,
        ce4_f_stop_mhz=_env_float("CWIPSS_PERF_SUITE_CE4_F_STOP", 0.0)
        if os.getenv("CWIPSS_PERF_SUITE_CE4_F_STOP", "").strip()
        else None,
        signal_models=tuple(
            _env_list(
                "CWIPSS_PERF_SUITE_SIGNAL_MODELS",
                [
                    "single_channel_periodic",
                    "pulsed_periodic",
                    "intermittent_periodic",
                    "sinusoidal_narrowband",
                    "band_limited_periodic",
                    "drifting_ridge",
                ],
            )
        ),
        bandwidth_channels_min=_env_float("CWIPSS_PERF_SUITE_BANDWIDTH_MIN", 1.0),
        bandwidth_channels_max=_env_float("CWIPSS_PERF_SUITE_BANDWIDTH_MAX", 9.0),
        drift_channels_min=_env_float("CWIPSS_PERF_SUITE_DRIFT_MIN", 0.0),
        drift_channels_max=_env_float("CWIPSS_PERF_SUITE_DRIFT_MAX", 6.0),
        injection_period_min_records=_env_float("CWIPSS_PERF_SUITE_INJ_PERIOD_MIN", 12.0),
        injection_duration_fraction_min=_env_float("CWIPSS_PERF_SUITE_DURATION_FRAC_MIN", 0.35),
        injection_duration_fraction_max=_env_float("CWIPSS_PERF_SUITE_DURATION_FRAC_MAX", 1.0),
        amplitude_factor_min=_env_float("CWIPSS_PERF_SUITE_AMP_FACTOR_MIN", 0.25),
        amplitude_factor_max=_env_float("CWIPSS_PERF_SUITE_AMP_FACTOR_MAX", 1.50),
        duty_cycle_min=_env_float("CWIPSS_PERF_SUITE_DUTY_MIN", 0.05),
        duty_cycle_max=_env_float("CWIPSS_PERF_SUITE_DUTY_MAX", 0.20),
        period_min_records=_env_float("CWIPSS_PERF_SUITE_PERIOD_MIN", 2.0),
        candidate_period_min_records=_env_float("CWIPSS_PERF_SUITE_CAND_PERIOD_MIN", 10.0),
        structure_baseline_quantile=_env_float("CWIPSS_PERF_SUITE_STRUCTURE_BG_Q", 0.10),
        structure_scale_quantile=_env_float("CWIPSS_PERF_SUITE_STRUCTURE_SCALE_Q", 0.20),
        structure_z_threshold=_env_float("CWIPSS_PERF_SUITE_STRUCTURE_Z", 1.0),
        pelt_penalty=_env_float("CWIPSS_PERF_SUITE_PELT_PENALTY", 16.0),
        window_min_activity_mean=_env_float("CWIPSS_PERF_SUITE_WINDOW_MIN_MEAN", 0.0),
        algorithms=tuple(_env_list("CWIPSS_PERF_SUITE_ALGORITHMS", ["all"])),
        progress_every=_env_int("CWIPSS_PERF_SUITE_PROGRESS_EVERY", 8),
    )

    include_xlarge = _env_flag("CWIPSS_PERF_SUITE_INCLUDE_XLARGE", default=False)
    defaults = {regime.name: regime for regime in default_regimes(include_xlarge=include_xlarge)}
    cases_per_regime = _env_int("CWIPSS_PERF_SUITE_CASES", 24)
    regimes_list = [
        replace(defaults["small"], case_count=_env_int("CWIPSS_PERF_SUITE_CASES_SMALL", cases_per_regime)),
        replace(defaults["medium"], case_count=_env_int("CWIPSS_PERF_SUITE_CASES_MEDIUM", cases_per_regime)),
        replace(defaults["large"], case_count=_env_int("CWIPSS_PERF_SUITE_CASES_LARGE", cases_per_regime)),
    ]
    if include_xlarge:
        regimes_list.append(replace(defaults["xlarge"], case_count=_env_int("CWIPSS_PERF_SUITE_CASES_XLARGE", cases_per_regime)))
    regimes = tuple(regimes_list)

    suite = run_stratified_compression_benchmark(
        base_config=base,
        regimes=regimes,
        output_dir=output_dir,
    )

    payload = json.loads(suite.suite_json.read_text())
    top_rows = payload["stability_rows"][:5]
    print("overall best algorithm:", payload["overall_best_algorithm"])
    print("top stability algorithms:", [row["algorithm"] for row in top_rows])

    assert suite.regime_summary_csv.exists()
    assert suite.stability_csv.exists()
    assert suite.suite_json.exists()
    assert suite.regime_count == len(regimes)


@pytest.mark.perf
def test_configured_real_ce4_compression_rank() -> None:
    if not _env_flag("CWIPSS_RUN_PERF", default=False):
        pytest.skip("set CWIPSS_RUN_PERF=1 to run configured CE4 compression rank")

    output_dir = Path(
        os.getenv(
            "CWIPSS_CONFIG_COMPRESSION_OUTPUT",
            "runs/pytest_compression_rank_ce4_lowfreq_config_100",
        )
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    input_value = os.getenv("CWIPSS_CONFIG_COMPRESSION_INPUT", "").strip()
    input_path = Path(input_value) if input_value else largest_complete_2c(PROJECT_DIR / "data" / "CE4")
    if not input_path.is_absolute():
        input_path = PROJECT_DIR / input_path

    selected_algorithms = tuple(
        _env_list(
            "CWIPSS_CONFIG_COMPRESSION_ALGORITHMS",
            list(DEFAULT_TOP10_ALGORITHMS),
        )
    )
    result = run_configured_compression_rank(
        ConfiguredCompressionRun(
            output_dir=output_dir,
            input_path=input_path,
            injection_config=PROJECT_DIR
            / os.getenv(
                "CWIPSS_CONFIG_COMPRESSION_INJECTIONS",
                "configs/injection_lowfreq_random_100.json",
            ),
            cwt_config=PROJECT_DIR
            / os.getenv(
                "CWIPSS_CONFIG_COMPRESSION_CWT",
                "configs/cwt_default.json",
            ),
            algorithms=selected_algorithms,
            cwt_backend=os.getenv("CWIPSS_CONFIG_COMPRESSION_BACKEND", "cpu"),
            candidate_period_max_records=_env_float(
                "CWIPSS_CONFIG_COMPRESSION_CANDIDATE_MAX",
                1000.0,
            ),
            progress_every=_env_int(
                "CWIPSS_CONFIG_COMPRESSION_PROGRESS_EVERY",
                10,
            ),
        )
    )

    assert result["component_count"] == 133
    assert result["group_count"] == 100
    expected_count = result["available_algorithm_count"] if "all" in selected_algorithms else len(selected_algorithms)
    assert result["algorithm_count"] == expected_count
    assert result["available_algorithm_count"] >= 72
    assert (output_dir / "component_cases.csv").exists()
    assert (output_dir / "group_cases.csv").exists()
    assert (output_dir / "compression_summary.csv").exists()
    assert (output_dir / "compression_algorithm_map.json").exists()


@pytest.mark.perf
def test_cwt_activity_real_ce4_rank() -> None:
    if not _env_flag("CWIPSS_RUN_PERF", default=False):
        pytest.skip("set CWIPSS_RUN_PERF=1 to run real CE4 post-CWT activity rank")

    output_dir = Path(
        os.getenv(
            "CWIPSS_ACTIVITY_OUTPUT",
            "runs/pytest_cwt_activity_rank_ce4_lowfreq_config_100",
        )
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    input_value = os.getenv("CWIPSS_ACTIVITY_INPUT", "").strip()
    input_path = Path(input_value) if input_value else largest_complete_2c(PROJECT_DIR / "data" / "CE4")
    if not input_path.is_absolute():
        input_path = PROJECT_DIR / input_path

    selected_algorithms = tuple(
        _env_list(
            "CWIPSS_ACTIVITY_ALGORITHMS",
            list(DEFAULT_CWT_ACTIVITY_ALGORITHMS),
        )
    )
    result = run_cwt_activity_rank(
        CWTActivityRun(
            output_dir=output_dir,
            input_path=input_path,
            injection_config=PROJECT_DIR
            / os.getenv(
                "CWIPSS_ACTIVITY_INJECTIONS",
                "configs/injection_lowfreq_random_100.json",
            ),
            cwt_config=PROJECT_DIR
            / os.getenv(
                "CWIPSS_ACTIVITY_CWT",
                "configs/cwt_default.json",
            ),
            algorithms=selected_algorithms,
            cwt_backend=os.getenv("CWIPSS_ACTIVITY_BACKEND", "cpu"),
            pelt_threads=_env_int("CWIPSS_ACTIVITY_PELT_THREADS", 0),
            candidate_period_max_records=_env_float(
                "CWIPSS_ACTIVITY_CANDIDATE_MAX",
                1000.0,
            ),
            progress_every=_env_int(
                "CWIPSS_ACTIVITY_PROGRESS_EVERY",
                10,
            ),
            negative_control=_env_flag("CWIPSS_ACTIVITY_NEGATIVE_CONTROL", True),
            negative_f_start_mhz=_env_float(
                "CWIPSS_ACTIVITY_NEGATIVE_F_START",
                0.15,
            ),
            negative_f_stop_mhz=_env_float(
                "CWIPSS_ACTIVITY_NEGATIVE_F_STOP",
                1.90,
            ),
            negative_max_channels=_env_int(
                "CWIPSS_ACTIVITY_NEGATIVE_MAX_CHANNELS",
                0,
            ),
            negative_window_method=os.getenv(
                "CWIPSS_ACTIVITY_NEGATIVE_WINDOW_METHOD",
                "pelt",
            ),
            strict_single_map=_env_flag("CWIPSS_ACTIVITY_STRICT_SINGLE_MAP", False),
            max_groups_per_family=_env_int("CWIPSS_ACTIVITY_MAX_GROUPS_PER_FAMILY", 0),
        )
    )

    assert result["component_count"] == 133
    assert result["group_count"] == 100
    expected_count = result["available_algorithm_count"] if "all" in selected_algorithms else len(selected_algorithms)
    assert result["algorithm_count"] == expected_count
    assert result["available_algorithm_count"] >= 10
    assert result["paradigm"]["constraint"].startswith("no shared post-CWT")
    assert "negative_control" in result
    assert (output_dir / "component_cases.csv").exists()
    assert (output_dir / "group_cases.csv").exists()
    assert (output_dir / "negative_control_cases.csv").exists()
    assert (output_dir / "negative_control_summary.csv").exists()
    assert (output_dir / "cwt_activity_summary.csv").exists()
    assert (output_dir / "cwt_activity_summary.json").exists()
    assert (output_dir / "cwt_activity_algorithm_map.json").exists()
