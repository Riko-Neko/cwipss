from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from cwipss.signal import cpro_cuda
from cwipss.signal.cpro import (
    CPROParameters,
    cpro_activity,
    cpro_continuity_features,
    cpro_continuity_map,
    difference_noise_std,
)
from cwipss.signal.cprf import CPRFParameters
from cwipss.signal.detection import (
    detect_block_periods,
    filter_ridge_continuity_windows,
    pelt_proposals_from_segments,
    resolve_channel_candidate_cap,
)
from cwipss.signal.windows import Segment, merge_close_windows


def _small_params() -> CPROParameters:
    return CPROParameters(
        threshold_snr=2.0,
        texture_quantile=0.0,
        period_center_bins=1,
        period_context_bins=1,
        min_period_contrast=0.0,
        period_support_bins=1,
    )


def test_cpro_shape_axis_responds_to_persistent_ridge() -> None:
    power = np.ones((3, 128), dtype=np.float32)
    power[1, 24:104] = 12.0
    result = cpro_activity(
        power,
        noise_std=1.0,
        noise_gain=np.ones(3),
        params=_small_params(),
    )
    assert np.max(result.shape_activity[40:90]) > np.max(result.shape_activity[:20])


def test_cpro_shape_axis_does_not_spread_a_short_bright_transient() -> None:
    power = np.ones((3, 128), dtype=np.float32)
    power[1, 60:62] = 100.0
    result = cpro_activity(
        power,
        noise_std=1.0,
        noise_gain=np.ones(3),
        params=_small_params(),
    )
    baseline = cpro_activity(
        np.ones((3, 128), dtype=np.float32),
        noise_std=1.0,
        noise_gain=np.ones(3),
        params=_small_params(),
    )
    np.testing.assert_allclose(result.shape_activity[:60], baseline.shape_activity[:60])
    np.testing.assert_allclose(result.shape_activity[62:], baseline.shape_activity[62:])
    assert np.min(result.shape_activity[60:62]) > np.max(baseline.shape_activity)


def test_cpro_activity_has_no_legacy_window_parameters() -> None:
    power = np.ones((3, 128), dtype=np.float32)
    power[1, 24:104] = 12.0
    params = _small_params()

    result = cpro_activity(power, noise_std=1.0, noise_gain=np.ones(3), params=params)

    assert "max_gap_records" not in params.__dict__
    assert "min_duration_records" not in params.__dict__
    assert "support_records" not in params.__dict__
    assert "min_occupancy" not in params.__dict__
    assert np.max(result.shape_activity) > 0.0


def test_cpro_activity_is_in_noise_calibrated_power_units() -> None:
    power = np.ones((3, 128), dtype=np.float32)
    power[1, 24:104] = 12.0

    base = cpro_activity(power, noise_std=1.0, noise_gain=np.ones(3), params=_small_params())
    scaled = cpro_activity(4.0 * power, noise_std=2.0, noise_gain=np.ones(3), params=_small_params())

    np.testing.assert_allclose(base.shape_activity, scaled.shape_activity)
    np.testing.assert_allclose(base.shape_map, scaled.shape_map)


def test_cpro_continuity_suppresses_time_isolated_ridge_energy() -> None:
    shape = np.zeros((3, 256), dtype=np.float32)
    shape[1, 32:160] = 10.0
    shape[2, 220:224] = 10.0

    evidence = cpro_continuity_map(shape, decay=0.95, power=2.0)
    persistent, _lock = cpro_continuity_features(evidence, 32, 160, threshold=10.0)
    transient, _lock = cpro_continuity_features(evidence, 220, 224, threshold=10.0)

    assert persistent > transient


def test_cpro_ridge_lock_requires_energy_on_one_period_row() -> None:
    locked = np.zeros((3, 32), dtype=np.float32)
    locked[1, :] = 1.0
    split = np.zeros_like(locked)
    split[0, :16] = 1.0
    split[2, 16:] = 1.0

    _mean, locked_score = cpro_continuity_features(locked, 0, 32, threshold=1.0)
    _mean, split_score = cpro_continuity_features(split, 0, 32, threshold=1.0)

    assert locked_score == pytest.approx(1.0)
    assert split_score == pytest.approx(0.5)


def test_cpro_cpu_cuda_math_is_identical_without_host_transfer(monkeypatch) -> None:
    rng = np.random.default_rng(123)
    power = rng.lognormal(mean=1.0, sigma=0.8, size=(8, 256)).astype(np.float32)
    power[3:6, 64:208] *= 20.0
    gain = np.linspace(0.8, 1.2, power.shape[0], dtype=np.float32)
    params = CPROParameters(
        threshold_snr=4.0,
        texture_quantile=0.8,
        period_support_bins=3,
    )
    cpu = cpro_activity(power, noise_std=1.5, noise_gain=gain, params=params)
    monkeypatch.setattr(cpro_cuda, "_cupy_modules", lambda: (np, ndimage))

    cuda = cpro_cuda.cpro_activity_cuda(
        power,
        noise_std=1.5,
        noise_gain=gain,
        params=params,
    )

    for field in (
        "shape_activity",
        "shape_map",
    ):
        np.testing.assert_allclose(getattr(cuda, field), getattr(cpu, field), rtol=2e-5, atol=2e-5)


def test_cpro_continuity_cuda_matches_cpu_on_real_device() -> None:
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.Device(0).compute_capability
    except Exception:
        pytest.skip("CUDA device is unavailable")
    rng = np.random.default_rng(44)
    shape = rng.lognormal(mean=0.0, sigma=0.8, size=(8, 257)).astype(np.float32)
    shape[3, 32:224] *= 8.0
    windows = [
        {"record_start": 0, "record_stop": 64},
        {"record_start": 32, "record_stop": 224},
        {"record_start": 224, "record_stop": 257},
    ]

    expected_map = cpro_continuity_map(shape, decay=0.995, power=2.0)
    actual_map = cpro_cuda.cpro_continuity_map_cuda(
        cp.asarray(shape),
        decay=0.995,
        power=2.0,
    )
    np.testing.assert_allclose(cp.asnumpy(actual_map), expected_map, rtol=2e-5, atol=2e-5)
    expected = np.asarray(
        [
            cpro_continuity_features(
                expected_map,
                int(window["record_start"]),
                int(window["record_stop"]),
                threshold=4.0,
            )
            for window in windows
        ]
    )
    means, locks = cpro_cuda.cpro_continuity_features_cuda(
        actual_map,
        windows,
        threshold=cp.asarray(4.0),
    )
    np.testing.assert_allclose(np.column_stack((means, locks)), expected, rtol=2e-5, atol=2e-5)


def test_noise_calibration_has_no_degenerate_fallback() -> None:
    with pytest.raises(ValueError, match="positive"):
        difference_noise_std(np.ones(32, dtype=np.float32))


def test_detector_excludes_zero_channel_without_noise_fallback() -> None:
    invalid_channels: list[dict] = []
    candidates, windows = detect_block_periods(
        power_cube=np.zeros((3, 32, 1), dtype=np.float32),
        raw_data=np.zeros((32, 1), dtype=np.float32),
        periods=np.array([10.0, 20.0, 40.0]),
        freqs_mhz=np.array([1.0]),
        noise_gain=np.ones(3),
        record_start=0,
        target_channel_start=0,
        target_channel_stop=1,
        candidate_period_min_records=None,
        candidate_period_max_records=None,
        cpro_threshold_snr=2.0,
        cpro_texture_quantile=0.0,
        cpro_period_center_bins=1,
        cpro_period_context_bins=1,
        cpro_min_period_contrast=0.0,
        cpro_period_support_bins=1,
        cpro_shape_power_softness=1.0,
        cpro_shape_contrast_softness=0.1,
        cpro_continuity_decay=0.995,
        cpro_continuity_power=2.0,
        cpro_min_continuity_mean=0.47,
        cpro_min_ridge_lock=0.94,
        pelt_penalty=1.0,
        pelt_min_size_records=4,
        pelt_jump_records=1,
        pelt_threads=1,
        window_min_activity_mean=0.0,
        window_merge_gap_records=0,
        cprf_params=CPRFParameters(),
        max_candidates_per_channel="auto",
        invalid_channels=invalid_channels,
    )

    assert candidates == []
    assert windows == []
    assert invalid_channels == [
        {
            "channel": 0,
            "freq_mhz": 1.0,
            "finite_records": 32,
            "data_min": 0.0,
            "data_max": 0.0,
            "reason": "all_zero",
        }
    ]


def test_candidate_cap_auto_scales_with_record_count() -> None:
    assert resolve_channel_candidate_cap("auto", 3.0 / 4096.0, 4096) == 3
    assert resolve_channel_candidate_cap("auto", 3.0 / 4096.0, 8192) == 6


def test_pelt_proposals_have_no_independent_duration_gate() -> None:
    shape_activity = np.ones(16, dtype=np.float32)
    activity_z = np.ones(16, dtype=np.float32)
    segments = [
        Segment(start=0, stop=8, cost=1.0, mean=1.0),
        Segment(start=8, stop=16, cost=2.0, mean=1.0),
    ]

    windows = pelt_proposals_from_segments(
        shape_activity,
        activity_z,
        segments,
        penalty=16.0,
        min_mean=0.05,
    )

    assert len(windows) == 2
    assert windows[0]["record_start"] == 0
    assert windows[0]["record_stop"] == 8
    assert windows[0]["duration_records"] == 8


def test_ridge_continuity_gate_selects_without_duration_condition() -> None:
    proposals = [
        {"record_start": 0, "record_stop": 8, "duration_records": 8},
        {"record_start": 8, "record_stop": 16, "duration_records": 8},
    ]
    windows = filter_ridge_continuity_windows(
        proposals,
        [0.5, 0.4],
        [0.95, 1.0],
        min_continuity_mean=0.47,
        min_ridge_lock=0.94,
        merge_gap=0,
    )

    assert [(row["record_start"], row["record_stop"]) for row in windows] == [(0, 8)]


def test_window_merge_rejects_unsorted_input_instead_of_sorting() -> None:
    with pytest.raises(ValueError, match="ordered"):
        merge_close_windows(
            [
                {"record_start": 8, "record_stop": 16},
                {"record_start": 0, "record_stop": 8},
            ]
        )
