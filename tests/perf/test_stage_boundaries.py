from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import period_profile_benchmark
from cwt_activity_algorithms import resolve_cwt_activity_algorithms
from stage_boundaries import (
    PELTWindowParameters,
    segment_activity_batch_with_pelt,
    segment_activity_with_pelt,
    stage3_windows,
)


def _parameters() -> PELTWindowParameters:
    return PELTWindowParameters(
        penalty=1.0,
        min_size_records=8,
        jump_records=1,
        min_duration_records=8,
        min_activity_mean=0.5,
        merge_gap_records=0,
        adapt_to_short_series=False,
    )


def test_native_pelt_is_the_only_activity_to_window_boundary() -> None:
    activity = np.concatenate(
        (np.zeros(32), np.full(32, 4.0), np.zeros(32))
    ).astype(np.float32)

    result = segment_activity_with_pelt(activity, _parameters(), activity_z=activity)

    assert [(segment.start, segment.stop) for segment in result.segments] == [
        (0, 32),
        (32, 64),
        (64, 96),
    ]
    assert [(window["record_start"], window["record_stop"]) for window in result.windows] == [
        (32, 64)
    ]


def test_stage3_duration_gate_only_removes_short_pelt_windows() -> None:
    windows = (
        {"record_start": 10, "record_stop": 30},
        {"record_start": 40, "record_stop": 90},
    )

    accepted = stage3_windows(windows, minimum_duration_records=32)

    assert accepted == ({"record_start": 40, "record_stop": 90},)


def test_batched_native_pelt_matches_independent_segmentation() -> None:
    first = np.concatenate((np.zeros(32), np.full(32, 4.0), np.zeros(32))).astype(np.float32)
    second = np.concatenate((np.zeros(16), np.full(48, 3.0), np.zeros(32))).astype(np.float32)

    batched = segment_activity_batch_with_pelt(
        np.stack((first, second)),
        _parameters(),
        activities_z=np.stack((first, second)),
        threads=2,
    )
    independent = (
        segment_activity_with_pelt(first, _parameters(), activity_z=first),
        segment_activity_with_pelt(second, _parameters(), activity_z=second),
    )

    assert tuple(result.segments for result in batched) == tuple(
        result.segments for result in independent
    )
    assert tuple(result.windows for result in batched) == tuple(
        result.windows for result in independent
    )


def test_period_rank_uses_activity_only_then_pelt(monkeypatch) -> None:
    algorithm = resolve_cwt_activity_algorithms(("sm_cpro_base",))[0]
    activity = np.concatenate(
        (np.zeros(32), np.full(32, 4.0), np.zeros(32))
    ).astype(np.float32)

    def fake_activity(*args, **kwargs):
        return SimpleNamespace(activity=activity, score_map=np.ones((3, activity.size), dtype=np.float32))

    monkeypatch.setattr(period_profile_benchmark, "compute_cwt_activity", fake_activity)
    pelt_windows, accepted = period_profile_benchmark._pelt_windows_for_power(
        np.zeros(activity.size, dtype=np.float32),
        np.ones((3, activity.size), dtype=np.float32),
        algorithm=algorithm,
        periods=np.asarray([10.0, 20.0, 30.0]),
        noise_std=1.0,
        noise_gain=np.ones(3, dtype=np.float32),
        pelt_parameters=_parameters(),
        stage3_min_window_records=24,
        wavelet="cmor1.5-1.0",
        method="fft",
        backend="cpu",
        cuda_device=0,
    )

    assert [(window["record_start"], window["record_stop"]) for window in pelt_windows] == [
        (32, 64)
    ]
    assert accepted == pelt_windows


def test_profile_normalization_is_independent_of_activity_algorithm() -> None:
    power = np.asarray([[32.0, 64.0], [64.0, 128.0]], dtype=np.float32)
    gain = np.asarray([1.0, 2.0], dtype=np.float32)

    threshold = period_profile_benchmark._profile_normalization_threshold(
        power,
        noise_std=1.0,
        noise_gain=gain,
        threshold_snr=16.0,
        texture_quantile=0.0,
    )
    normalized = period_profile_benchmark._normalized_cwt_power(
        power,
        noise_std=1.0,
        noise_gain=gain,
        calibrated_threshold=threshold,
    )

    assert threshold == 16.0
    np.testing.assert_allclose(normalized, [[2.0, 4.0], [2.0, 4.0]])
