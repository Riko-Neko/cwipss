from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from cwipss.analysis.benchmark import (
    MatchConfig,
    CWTBenchmarkConfig,
    aggregate_injection_performance,
    evaluate_injections,
    run_injection_benchmark,
)
from cwipss.analysis.injection import synthetic_background
from cwipss.analysis.injection_config import load_injection_config, make_injections_from_config
from cwipss.analysis.simulation import InjectionSpec, inject_periodic_signal
from cwipss.analysis.validation import ValidationConfig
from cwipss.analysis.veto import VetoConfig


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
        "t0_rec": 12,
        "t1_rec": 88,
        "freq_mhz": 4.0,
        "score": 10,
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
        "t0_rec": 12,
        "t1_rec": 88,
        "freq_mhz": 4.0,
        "score": 99,
        "candidate_status": "vetoed",
    }
    kept = {
        "candidate_id": 2,
        "t0_rec": 12,
        "t1_rec": 88,
        "freq_mhz": 4.0,
        "score": 1,
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


def test_injection_config_randomizes_long_time_spans_and_replicates() -> None:
    payload = {
        "seed": 7,
        "sets": [
            {
                "name": "weak",
                "count": 2,
                "signal_model": "single_channel_periodic",
                "period_records": {"values": [8, 16]},
                "amplitude": {"min": 0.1, "max": 0.2},
                "frequency_mhz": {"min": 0.0, "max": 15.0},
                "time": {"duration_fraction": {"min": 0.5, "max": 0.8}},
                "modulation": {
                    "phase": {"min": 0.0, "max": 1.0},
                    "duty_cycle": {"min": 0.08, "max": 0.2},
                },
                "replication": {"probability": 1.0, "max_copies": 2},
            }
        ],
    }

    specs = make_injections_from_config(
        payload,
        records=128,
        channels=16,
        freqs_mhz=np.arange(16, dtype=np.float64),
    )

    assert len(specs) == 4
    assert all(spec.duration_records is not None and spec.duration_records >= 64 for spec in specs)
    assert all(0 <= spec.record_start <= 128 - int(spec.duration_records or 0) for spec in specs)
    assert len({spec.channel_center for spec in specs}) > 1
    assert len({(spec.period_records, spec.record_start, spec.duration_records) for spec in specs}) == 2


def test_random_100_configs_expand_to_at_least_ten_cycles() -> None:
    records = 43794
    freqs = np.linspace(0.1, 40.0, 2048, dtype=np.float64)

    for path in (
        Path("configs/injection_fullband_random_100.json"),
        Path("configs/injection_lowfreq_random_100.json"),
    ):
        payload = load_injection_config(path)
        specs = make_injections_from_config(payload, records=records, channels=freqs.size, freqs_mhz=freqs)

        assert len(payload["sets"]) == 100
        assert len(specs) == 133
        base_periods = [float(item["period_records"]["value"]) for item in payload["sets"]]
        assert sum(period < 100.0 for period in base_periods) == 75
        assert sum(100.0 <= period < 1000.0 for period in base_periods) == 25
        assert all(float(spec.duration_records or 0) / float(spec.period_records) >= 10.0 for spec in specs)
        assert all(0 <= spec.record_start <= records - int(spec.duration_records or 0) for spec in specs)


def test_random_100_frequency_regions_match_config_names() -> None:
    freqs = np.linspace(0.1, 40.0, 2048, dtype=np.float64)

    fullband = make_injections_from_config(
        load_injection_config(Path("configs/injection_fullband_random_100.json")),
        records=43794,
        channels=freqs.size,
        freqs_mhz=freqs,
    )
    fullband_channels = {int(round(spec.channel_center)) for spec in fullband}
    assert len(fullband_channels) == len(fullband)
    assert min(fullband_channels) < 128
    assert max(fullband_channels) > 1920

    lowfreq = make_injections_from_config(
        load_injection_config(Path("configs/injection_lowfreq_random_100.json")),
        records=43794,
        channels=freqs.size,
        freqs_mhz=freqs,
    )
    lowfreq_values = [float(freqs[int(round(spec.channel_center))]) for spec in lowfreq]
    step = float(np.nanmedian(np.diff(freqs)))
    assert min(lowfreq_values) >= 0.15 - step
    assert max(lowfreq_values) <= 1.9 + step


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
            candidate_period_min_records=2.0,
            candidate_period_max_records=16.0,
            cpro_threshold_snr=2.0,
            cpro_texture_quantile=0.0,
            cpro_period_center_bins=1,
            cpro_period_context_bins=1,
            cpro_min_period_contrast=0.0,
            cpro_support_records=5,
            cpro_min_occupancy=0.4,
            cpro_period_support_bins=1,
            cpro_window_support_records=9,
            cpro_min_window_occupancy=0.2,
            cprf_threshold_snr=1.0,
            cprf_texture_quantile=0.0,
            cprf_smooth_bins=1,
            cprf_min_width_bins=1,
            cprf_min_peak_strength=0.0,
            cprf_min_band_persistence=0.0,
            cprf_min_band_concentration=0.0,
            cprf_min_local_contrast=0.0,
            max_candidates_per_channel=2,
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
    assert (run_dir / "time_windows.csv").exists()
    assert (run_dir / "candidates_raw.csv").exists()
    assert (run_dir / "validation_reviewed.csv").exists()
    assert (run_dir / "injection_performance.csv").exists()
    with (run_dir / "time_windows.csv").open(newline="") as stream:
        assert "accepted" in (csv.DictReader(stream).fieldnames or [])
    with (run_dir / "candidates_raw.csv").open(newline="") as stream:
        candidate_fields = csv.DictReader(stream).fieldnames or []
    assert "band_conc" in candidate_fields
    assert "local_contrast" in candidate_fields
    assert "integrated_score" not in candidate_fields
    assert _read_csv(run_dir / "injection_results.csv")[0]["injection_id"] == "inj_0001"
    summary = json.loads((run_dir / "injection_summary.json").read_text())
    assert summary["schema_version"] == 6
    assert summary["injection_count"] == 1
