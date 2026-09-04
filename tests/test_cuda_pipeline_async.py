from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from cwipss.signal import cpro_cuda, cwt_cuda, detection, detection_cuda
from cwipss.signal.windows import PeltCancellation, Segment, pelt_mean_shift_batch
from cwipss.workflows import search


def test_async_timing_accumulator_accepts_disabled_timing() -> None:
    search._timing_add(None, "pelt_seconds", 1.0)
    totals: dict[str, float] = {}
    search._timing_add(totals, "pelt_seconds", 1.0)
    assert totals == {"pelt_seconds": 1.0}


def test_block_timing_reports_native_pelt_candidate_diagnostics() -> None:
    message = search._timing_block_message(
        "run",
        "block_0001",
        (0, 8),
        2048,
        {"detect_seconds": 1.0},
        {
            "pelt_candidates_mean": 12.5,
            "pelt_candidates_max": 21.0,
            "pelt_short_circuit_channels": 3.0,
            "pelt_constant_channels": 1.0,
        },
        candidates=0,
        windows=0,
    )

    assert "pelt_candidates(mean=12.5 max=21)" in message
    assert "pelt_skip=3" in message
    assert "pelt_constant=1" in message


def test_invalid_channel_quality_is_compacted_per_file() -> None:
    summary = search._channel_quality_summary(
        selected_channel_count=8,
        invalid_channels=[
            {"channel": 5, "reason": "all_zero"},
            {"channel": 6, "reason": "all_zero"},
            {"channel": 7, "reason": "all_zero"},
        ],
    )

    assert summary["valid_channel_count"] == 5
    assert summary["quality_status"] == "invalid_channels_excluded"
    assert summary["invalid_ranges"] == [
        {"channel_start": 5, "channel_stop": 8, "count": 3, "reason": "all_zero"}
    ]


def test_cpro_core_has_no_host_array_transfer() -> None:
    activity_source = inspect.getsource(cpro_cuda.cpro_activity_cuda)
    continuity_source = inspect.getsource(cpro_cuda.cpro_continuity_map_cuda)
    feature_source = inspect.getsource(cpro_cuda.cpro_continuity_features_cuda)
    assert "asnumpy" not in activity_source
    assert "asnumpy" not in continuity_source
    assert feature_source.count("cp.asnumpy") == 1


def test_cuda_wavelet_integration_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    original = cwt_cuda.integrate_wavelet
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cwt_cuda, "integrate_wavelet", counted)
    cwt_cuda._cached_integrated_wavelet.cache_clear()
    first = cwt_cuda._integrated_wavelet("cmor1.5-1.0", np.dtype(np.float32))
    second = cwt_cuda._integrated_wavelet("cmor1.5-1.0", np.dtype(np.float32))

    assert calls == 1
    assert first[1] is second[1]
    assert first[2] is second[2]
    assert not first[1].flags.writeable
    cwt_cuda._cached_integrated_wavelet.cache_clear()


def test_cuda_orchestration_keeps_cwt_resident_through_cprf() -> None:
    source = inspect.getsource(detection_cuda.prepare_block_period_chunks_cuda_power)
    assert "cp.asnumpy(result.shape_activity)" in source
    assert "cpro_continuity_map_cuda" in source
    assert "cp.asnumpy(continuity_map)" not in source
    assert "cp.asnumpy(result.shape_map)" not in source
    assert "cp.asnumpy(result.occupancy_map)" not in source
    assert "cp.asnumpy(power" not in source
    assert "power_map=valid_power[:, :, target]" in source
    assert "cprf_normalization_threshold_cuda" in source

    pelt_source = inspect.getsource(detection_cuda.run_prepared_cuda_pelt)
    assert "pelt_mean_shift_batch" in pelt_source

    finalize_source = inspect.getsource(detection_cuda.finalize_prepared_cuda_period_chunks)
    assert "evaluate_cprf_cuda" in finalize_source
    assert "cpro_continuity_features_cuda" in finalize_source
    assert "cp.asnumpy" not in finalize_source
    assert "channel.power_map = None" in finalize_source

    workflow_source = inspect.getsource(search.run_cwt_search)
    assert "pelt_executor.submit(" in workflow_source
    assert "run_prepared_cuda_pelt," in workflow_source
    assert "pelt_cancellation.cancel()" in workflow_source
    assert "pending_blocks.popleft()" in workflow_source
    assert search.CWTSearchConfig().cuda_max_pending_blocks == 2


def test_native_batch_pelt_honors_preexisting_cancellation() -> None:
    cancellation = PeltCancellation()
    cancellation.cancel()

    with pytest.raises(RuntimeError, match="native PELT cancelled"):
        pelt_mean_shift_batch(
            np.zeros((2, 64), dtype=np.float64),
            penalty=1.0,
            min_size=8,
            threads=2,
            cancellation=cancellation,
        )


def test_native_batch_pelt_constant_fast_path_preserves_segments_and_reports_diagnostics() -> None:
    diagnostics: dict[str, float] = {}
    activity = np.stack(
        [
            np.zeros(4096, dtype=np.float64),
            np.full(4096, 0.1, dtype=np.float64),
        ]
    )

    segments = pelt_mean_shift_batch(
        activity,
        penalty=16.0,
        min_size=384,
        jump=1,
        threads=2,
        diagnostics=diagnostics,
    )

    assert [[(row.start, row.stop) for row in channel] for channel in segments] == [
        [(0, 4096)],
        [(0, 4096)],
    ]
    assert segments[0][0].cost == 0.0
    assert segments[0][0].mean == 0.0
    assert segments[1][0].cost == 0.0
    assert segments[1][0].mean == pytest.approx(0.1)
    assert diagnostics["pelt_constant_channels"] == 2.0
    assert diagnostics["pelt_candidates_mean"] == 0.0
    assert diagnostics["pelt_candidates_max"] == 0.0


def test_native_batch_pelt_reports_candidate_growth() -> None:
    diagnostics: dict[str, float] = {}
    activity = np.random.default_rng(123).normal(size=(1, 2048))

    pelt_mean_shift_batch(
        activity,
        penalty=16.0,
        min_size=128,
        jump=1,
        diagnostics=diagnostics,
    )

    assert diagnostics["pelt_candidate_observations"] > 0.0
    assert diagnostics["pelt_candidates_mean"] > 0.0
    assert diagnostics["pelt_candidates_max"] >= diagnostics["pelt_candidates_mean"]


def test_shared_window_pipeline_short_circuits_below_mean_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_pelt(*_args, **_kwargs):
        raise AssertionError("native PELT must not run below the activity-mean gate")

    monkeypatch.setattr(detection, "pelt_mean_shift", fail_pelt)
    windows, activity_z, segment_count = detection.pelt_windows_from_activity(
        np.zeros(2048, dtype=np.float32),
        np.zeros((3, 2048), dtype=np.float32),
        calibrated_threshold=32.0,
        penalty=16.0,
        min_size=384,
        jump=1,
        min_mean=0.05,
        min_continuity_mean=0.47,
        min_ridge_lock=0.94,
        merge_gap=256,
    )

    assert windows == []
    assert segment_count == 0
    assert np.array_equal(activity_z, np.zeros_like(activity_z))


def test_cuda_pelt_short_circuit_preserves_batch_order_and_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_batch(activity, **kwargs):
        captured["activity"] = np.asarray(activity)
        diagnostics = kwargs["diagnostics"]
        diagnostics.update(
            {
                "pelt_candidates_mean": 12.5,
                "pelt_candidates_max": 21.0,
                "pelt_candidate_observations": 10.0,
                "pelt_constant_channels": 0.0,
            }
        )
        return [[Segment(0, 3, 1.0, 2.0)]]

    monkeypatch.setattr(detection_cuda, "pelt_mean_shift_batch", fake_batch)
    common = {
        "window_min_activity_mean": 1.0,
        "pelt_penalty": 16.0,
        "pelt_min_size_records": 1,
        "pelt_jump_records": 1,
        "pelt_threads": 8,
    }
    prepared = [
        SimpleNamespace(
            activity_z=np.array([0.0, 0.5, 0.0]),
            **common,
        ),
        SimpleNamespace(
            activity_z=np.array([0.0, 2.0, 0.0]),
            **common,
        ),
    ]
    timing: dict[str, float] = {}

    segments, _seconds = detection_cuda.run_prepared_cuda_pelt(prepared, timing=timing)

    assert segments[0] == []
    assert segments[1] == [Segment(0, 3, 1.0, 2.0)]
    assert np.asarray(captured["activity"]).shape == (1, 3)
    assert timing["pelt_short_circuit_channels"] == 1.0
    assert timing["pelt_candidates_mean"] == 12.5
    assert timing["pelt_candidates_max"] == 21.0
