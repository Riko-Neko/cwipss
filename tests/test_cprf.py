from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy import ndimage

from cwipss.signal import cprf_cuda
from cwipss.signal.cprf import CPRFParameters, evaluate_cprf, normalize_cwt_power
from cwipss.signal.detection import build_channel_candidates


def _ridge_map() -> tuple[np.ndarray, np.ndarray]:
    periods = np.geomspace(10.0, 200.0, 24)
    values = np.full((periods.size, 256), 0.25, dtype=np.float32)
    values[10:15, :] = 12.0
    return values, periods


def test_cprf_defaults_match_selected_low_false_positive_working_point() -> None:
    params = CPRFParameters()
    assert params.min_band_persistence == 0.40
    assert params.min_band_concentration == 0.50
    assert params.min_local_contrast == 1.20
    assert params.min_integrated_strength == 0.0


def test_cprf_accepts_a_persistent_concentrated_period_band() -> None:
    normalized, periods = _ridge_map()
    result = evaluate_cprf(
        normalized,
        periods,
        normalization_threshold=32.0,
    )
    assert result.accepted
    assert result.band_persistence >= CPRFParameters().min_band_persistence
    assert result.band_concentration >= CPRFParameters().min_band_concentration
    assert result.local_contrast >= CPRFParameters().min_local_contrast
    assert 10 <= result.peak_index <= 14


def test_cprf_rejects_short_lived_period_texture() -> None:
    periods = np.geomspace(10.0, 200.0, 24)
    normalized = np.full((periods.size, 256), 0.25, dtype=np.float32)
    normalized[10:15, 100:110] = 12.0
    result = evaluate_cprf(
        normalized,
        periods,
        normalization_threshold=32.0,
    )
    assert not result.accepted


def test_cprf_absolute_normalization_is_noise_scale_invariant() -> None:
    power, _periods = _ridge_map()
    params = CPRFParameters(threshold_snr=1.0, texture_quantile=0.0)
    base, base_threshold = normalize_cwt_power(
        power,
        noise_std=1.0,
        noise_gain=np.ones(power.shape[0]),
        params=params,
    )
    scaled, scaled_threshold = normalize_cwt_power(
        4.0 * power,
        noise_std=2.0,
        noise_gain=np.ones(power.shape[0]),
        params=params,
    )
    np.testing.assert_allclose(base, scaled)
    assert base_threshold == scaled_threshold == 1.0


def test_candidate_builder_emits_only_cprf_accepted_windows() -> None:
    accepted_map, periods = _ridge_map()
    rejected_map = np.full_like(accepted_map, 0.25)
    rejected_map[10:15, 100:110] = 12.0
    accepted = evaluate_cprf(accepted_map, periods, normalization_threshold=32.0)
    rejected = evaluate_cprf(rejected_map, periods, normalization_threshold=32.0)

    candidates, windows = build_channel_candidates(
        shape_activity=np.ones(64, dtype=np.float32),
        windows=(
            {"record_start": 0, "record_stop": 32},
            {"record_start": 32, "record_stop": 64},
        ),
        noise_std=1.0,
        calibrated_threshold=32.0,
        freq_mhz=1.0,
        channel_idx=0,
        record_start=0,
        max_candidates_per_channel=8,
        cprf_getter=lambda start, _stop: accepted if start == 0 else rejected,
    )
    assert len(windows) == 2
    assert [row["accepted"] for row in windows] == [1, 0]
    assert len(candidates) == 1
    assert candidates[0]["band_conc"] == accepted.band_concentration
    assert candidates[0]["score"] == accepted.total_score
    assert {
        "shape_mean",
        "pelt_z_mean",
        "ridge_peak",
        "ridge_int",
        "band_persist",
        "local_contrast",
        "core_score",
        "score",
    }.issubset(candidates[0])
    assert {"peak_score", "mean_score", "integrated_score"}.isdisjoint(candidates[0])


def test_cprf_cuda_core_transfers_only_one_final_scalar_pack() -> None:
    source = inspect.getsource(cprf_cuda.evaluate_cprf_cuda)
    assert source.count("cp.asnumpy") == 1
    assert "cp.asnumpy(power" not in source
    assert "cp.asnumpy(profile" not in source


def test_cprf_cuda_vectorized_math_matches_cpu_without_device(monkeypatch) -> None:
    class NumpyCudaShim:
        def __getattr__(self, name):
            return getattr(np, name)

        @staticmethod
        def asnumpy(values):
            return np.asarray(values)

    power, periods = _ridge_map()
    params = CPRFParameters(threshold_snr=1.0, texture_quantile=0.0)
    normalized, threshold = normalize_cwt_power(
        power,
        noise_std=1.0,
        noise_gain=np.ones(power.shape[0]),
        params=params,
    )
    cpu = evaluate_cprf(
        normalized,
        periods,
        normalization_threshold=threshold,
        params=params,
    )
    monkeypatch.setattr(cprf_cuda, "_cupy_modules", lambda: (NumpyCudaShim(), ndimage))
    cuda_path = cprf_cuda.evaluate_cprf_cuda(
        power,
        periods,
        noise_std=1.0,
        noise_gain=np.ones(power.shape[0]),
        normalization_threshold=threshold,
        params=params,
    )
    assert cuda_path.accepted == cpu.accepted
    assert cuda_path.peak_index == cpu.peak_index
    assert cuda_path.band_start_index == cpu.band_start_index
    assert cuda_path.band_stop_index == cpu.band_stop_index
    np.testing.assert_allclose(
        [cuda_path.band_concentration, cuda_path.local_contrast, cuda_path.total_score],
        [cpu.band_concentration, cpu.local_contrast, cpu.total_score],
        rtol=2e-5,
        atol=2e-5,
    )


def test_cprf_cuda_matches_cpu_when_available() -> None:
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDevice()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device unavailable")
    power, periods = _ridge_map()
    params = CPRFParameters(threshold_snr=1.0, texture_quantile=0.0)
    normalized, threshold = normalize_cwt_power(
        power,
        noise_std=1.0,
        noise_gain=np.ones(power.shape[0]),
        params=params,
    )
    cpu = evaluate_cprf(
        normalized,
        periods,
        normalization_threshold=threshold,
        params=params,
    )
    cuda = cprf_cuda.evaluate_cprf_cuda(
        cp.asarray(power),
        periods,
        noise_std=cp.asarray(1.0),
        noise_gain=cp.ones(power.shape[0]),
        normalization_threshold=cp.asarray(threshold),
        params=params,
    )
    assert cuda.accepted == cpu.accepted
    assert cuda.peak_index == cpu.peak_index
    assert cuda.band_start_index == cpu.band_start_index
    assert cuda.band_stop_index == cpu.band_stop_index
    np.testing.assert_allclose(
        [cuda.band_concentration, cuda.local_contrast, cuda.total_score],
        [cpu.band_concentration, cpu.local_contrast, cpu.total_score],
        rtol=2e-5,
        atol=2e-5,
    )
