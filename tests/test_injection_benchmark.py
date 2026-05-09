from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ce4_period_search.benchmark import (
    MatchConfig,
    CWTBenchmarkConfig,
    aggregate_injection_performance,
    evaluate_injections,
    make_default_injections,
    run_injection_benchmark,
)
from ce4_period_search.injection import synthetic_background
from ce4_period_search.simulation import InjectionSpec, inject_periodic_signal
from ce4_period_search.validation import ValidationConfig
from ce4_period_search.veto import VetoConfig


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def test_injection_truth_clamps_channel_span() -> None:
    background = synthetic_background(records=32, channels=8, seed=1)
    spec = InjectionSpec(
        injection_id="edge",
        period_records=8,
        amplitude=3,
        channel_center=99,
        bandwidth_channels=4,
    )

    _matrix, truth = inject_periodic_signal(background.data, spec)

    assert truth["channel_start"] == 7
    assert truth["channel_stop"] == 8
    assert truth["record_stop"] == 32


def test_single_channel_periodic_injection_only_modulates_one_channel() -> None:
    data = np.zeros((32, 6), dtype=np.float32)
    spec = InjectionSpec(
        injection_id="single",
        signal_model="single_channel_periodic",
        period_records=8,
        amplitude=5,
        channel_center=3,
        bandwidth_channels=4,
    )

    matrix, truth = inject_periodic_signal(data, spec)

    active_channels = np.flatnonzero(np.nanmax(np.abs(matrix), axis=0) > 0)
    assert active_channels.tolist() == [3]
    assert truth["channel_start"] == 3
    assert truth["channel_stop"] == 4
    assert truth["bandwidth_channels"] == 1.0


def test_evaluate_injections_reports_validated_match() -> None:
    truth = {
        "injection_id": "inj",
        "signal_model": "pulsed_periodic",
        "period_records": 16,
        "amplitude": 5,
        "record_start": 10,
        "record_stop": 90,
        "freq_start_mhz": 3.0,
        "freq_stop_mhz": 5.0,
    }
    candidate = {
        "candidate_id": 1,
        "record_start": 12,
        "record_stop": 88,
        "freq_start_mhz": 4.0,
        "freq_stop_mhz": 4.0,
        "peak_score": 10,
        "candidate_status": "needs_validation",
    }
    validation = {
        "candidate_id": 1,
        "validation_status": "evaluated",
        "refined_period_records": 16,
        "p_value": 0.01,
        "q_value": 0.01,
        "global_q_value": 0.01,
        "evidence_rank": 1,
    }

    rows = evaluate_injections([truth], [candidate], [candidate], [validation], MatchConfig())

    assert rows[0]["validated"] is True
    assert rows[0]["failure_stage"] == "validated"
    assert rows[0]["matched_candidate_id"] == 1


def test_evaluate_injections_matches_after_veto_candidates_only() -> None:
    truth = {
        "injection_id": "inj",
        "signal_model": "pulsed_periodic",
        "period_records": 8,
        "amplitude": 5,
        "record_start": 10,
        "record_stop": 90,
        "freq_start_mhz": 3.0,
        "freq_stop_mhz": 5.0,
    }
    vetoed = {
        "candidate_id": 1,
        "record_start": 12,
        "record_stop": 88,
        "freq_start_mhz": 4.0,
        "freq_stop_mhz": 4.0,
        "peak_score": 99,
        "candidate_status": "vetoed",
    }
    kept = {
        "candidate_id": 2,
        "record_start": 12,
        "record_stop": 88,
        "freq_start_mhz": 4.0,
        "freq_stop_mhz": 4.0,
        "peak_score": 1,
        "candidate_status": "needs_validation",
    }
    validation = {
        "candidate_id": 2,
        "validation_status": "evaluated",
        "refined_period_records": 8,
    }

    rows = evaluate_injections([truth], [vetoed, kept], [vetoed, kept], [validation], MatchConfig())

    assert rows[0]["detected_after_veto"] is True
    assert rows[0]["matched_candidate_id"] == 2


def test_default_injections_can_build_period_amplitude_grid() -> None:
    specs = make_default_injections(
        periods=[8, 16],
        amplitudes=[4, 6],
        records=128,
        channels=16,
        grid=True,
        repeats=2,
    )

    assert len(specs) == 8
    assert {spec.period_records for spec in specs} == {8.0, 16.0}
    assert {spec.amplitude for spec in specs} == {4.0, 6.0}
    assert {spec.signal_model for spec in specs} == {"single_channel_periodic"}
    assert {spec.bandwidth_channels for spec in specs} == {1.0}


def test_aggregate_injection_performance_groups_recovery_rates() -> None:
    rows = aggregate_injection_performance(
        [
            {"signal_model": "pulsed_periodic", "period_records": 8, "amplitude": 4, "detected_raw": True, "detected_after_veto": True, "validated": True, "failure_stage": "validated"},
            {"signal_model": "pulsed_periodic", "period_records": 8, "amplitude": 4, "detected_raw": True, "detected_after_veto": False, "validated": False, "failure_stage": "vetoed"},
        ]
    )

    assert rows[0]["injection_count"] == 2
    assert rows[0]["detected_raw_rate"] == 1.0
    assert rows[0]["validated_rate"] == 0.5


def test_run_injection_benchmark_writes_expected_outputs(tmp_path: Path) -> None:
    background = synthetic_background(records=128, channels=16, seed=2)
    injections = [
        InjectionSpec(
            injection_id="inj_0001",
            period_records=8,
            amplitude=8,
            record_start=32,
            duration_records=64,
            channel_center=8,
            bandwidth_channels=4,
        )
    ]

    run_dir = run_injection_benchmark(
        background=background,
        injections=injections,
        output_dir=tmp_path / "bench",
        run_id="bench",
        search_config=CWTBenchmarkConfig(
            wavelet="cmor1.5-1.0",
            period_min_records=2,
            period_max_records=16,
            period_count=8,
            block_channels=16,
            threshold=1.5,
            min_pixels=1,
            local_period=5,
            local_freq=5,
            max_candidates_per_block=20,
        ),
        veto_config=VetoConfig(enabled=False),
        validation_config=ValidationConfig(
            include_vetoed=True,
            max_candidates=10,
            window_periods=16,
            min_window_records=64,
            max_window_records=128,
            period_search_radius=4.0,
            shuffle_trials=3,
            random_seed=2,
        ),
        match_config=MatchConfig(min_time_overlap=0.05, min_freq_overlap=0.05),
    )

    assert (run_dir / "injection_truth.csv").exists()
    assert (run_dir / "candidates_raw.csv").exists()
    assert (run_dir / "validation_reviewed.csv").exists()
    assert (run_dir / "injection_performance.csv").exists()
    assert _read_csv(run_dir / "injection_results.csv")[0]["injection_id"] == "inj_0001"
    summary = json.loads((run_dir / "injection_summary.json").read_text())
    assert summary["injection_count"] == 1
