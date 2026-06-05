from __future__ import annotations

import pytest

from cwipss.config import cwt_config_from_mapping, cwt_config_to_nested_dict
from cwipss.models import RAW_CANDIDATE_FIELDNAMES, normalize_candidate_row


def test_structured_config_maps_to_resolved_dataclass() -> None:
    config = cwt_config_from_mapping(
        {
            "schema_version": 1,
            "input": {"path": "data/example.2C"},
            "scan": {"f_start": 38.0, "f_stop": 39.0, "period_count": 40, "cwt_backend": "cuda", "cuda_device": 1},
            "detection": {
                "noise_floor_fraction": 0.15,
                "structure_baseline_quantile": 0.08,
                "structure_scale_quantile": 0.25,
                "structure_z_threshold": 1.2,
                "structure_time_support_records": 32,
                "structure_period_support_bins": 5,
                "structure_min_support_fraction": 0.2,
                "activity_smooth_records": 12,
                "pelt_penalty": 6.0,
                "pelt_jump_records": 4,
                "window_min_duration_records": 12,
                "window_min_activity_raw_mean": 11.0,
                "candidate_period_min_records": 10,
                "candidate_period_max_records": 200,
                "max_candidates_per_channel": "auto",
                "max_candidates_per_record": 0.01,
            },
            "veto": {"enabled": True, "max_bandwidth_fraction": 0.8},
            "validation": {"max_candidates": 12, "shuffle_trials": 20},
            "visualization": {"enabled": True, "max_blocks": 1, "max_channels": 2, "top_candidates": 10},
            "progress": {"enabled": False, "leave": True},
            "timing": {"enabled": True},
            "output": {"output_dir": "runs", "run_id": "smoke"},
        }
    )

    assert config.input == "data/example.2C"
    assert config.f_start == 38.0
    assert config.f_stop == 39.0
    assert config.period_count == 40
    assert config.cwt_backend == "cuda"
    assert config.cuda_device == 1
    assert config.noise_floor_fraction == 0.15
    assert config.structure_baseline_quantile == 0.08
    assert config.structure_scale_quantile == 0.25
    assert config.structure_z_threshold == 1.2
    assert config.structure_time_support_records == 32
    assert config.structure_period_support_bins == 5
    assert config.structure_min_support_fraction == 0.2
    assert config.activity_smooth_records == 12
    assert config.pelt_penalty == 6.0
    assert config.pelt_jump_records == 4
    assert config.window_min_duration_records == 12
    assert config.window_min_activity_raw_mean == 11.0
    assert config.candidate_period_min_records == 10
    assert config.candidate_period_max_records == 200
    assert config.max_candidates_per_channel == "auto"
    assert config.max_candidates_per_record == 0.01
    assert config.veto_enabled is True
    assert config.veto_max_bandwidth_fraction == 0.8
    assert config.validation_max_candidates == 12
    assert config.validation_shuffle_trials == 20
    assert config.visualization_enabled is True
    assert config.visualization_max_blocks == 1
    assert config.visualization_max_channels == 2
    assert config.visualization_top_candidates == 10
    assert config.progress_enabled is False
    assert config.progress_leave is True
    assert config.timing_enabled is True
    assert config.run_id == "smoke"

    nested = cwt_config_to_nested_dict(config)
    assert nested["input"]["path"] == "data/example.2C"
    assert nested["scan"]["period_count"] == 40
    assert nested["scan"]["cwt_backend"] == "cuda"
    assert nested["scan"]["cuda_device"] == 1
    assert nested["detection"]["noise_floor_fraction"] == 0.15
    assert nested["detection"]["structure_baseline_quantile"] == 0.08
    assert nested["detection"]["structure_scale_quantile"] == 0.25
    assert nested["detection"]["structure_z_threshold"] == 1.2
    assert nested["detection"]["structure_time_support_records"] == 32
    assert nested["detection"]["structure_period_support_bins"] == 5
    assert nested["detection"]["structure_min_support_fraction"] == 0.2
    assert nested["detection"]["window_min_duration_records"] == 12
    assert nested["detection"]["window_min_activity_raw_mean"] == 11.0
    assert nested["detection"]["candidate_period_min_records"] == 10
    assert nested["detection"]["candidate_period_max_records"] == 200
    assert nested["detection"]["pelt_jump_records"] == 4
    assert nested["detection"]["max_candidates_per_channel"] == "auto"
    assert nested["detection"]["max_candidates_per_record"] == 0.01
    assert nested["veto"]["max_bandwidth_fraction"] == 0.8
    assert nested["validation"]["shuffle_trials"] == 20
    assert nested["visualization"]["enabled"] is True
    assert nested["progress"]["enabled"] is False
    assert nested["progress"]["leave"] is True
    assert nested["timing"]["enabled"] is True


def test_flat_config_remains_supported_with_overrides() -> None:
    config = cwt_config_from_mapping(
        {"input": "data/example.2C", "period_count": 30},
        overrides={"period_count": 50, "pelt_penalty": 7.0},
    )

    assert config.input == "data/example.2C"
    assert config.period_count == 50
    assert config.pelt_penalty == 7.0


def test_unknown_config_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown config key"):
        cwt_config_from_mapping({"scan": {"typo": 1}})


def test_normalize_candidate_row_adds_schema_and_seconds() -> None:
    row = normalize_candidate_row(
        {
            "cwt_wavelet": "cmor1.5-1.0",
            "time_aggregation": "p95",
            "detection_method": "single_channel_lowfloor_pelt_profile",
            "window_id": "ch0003_w0001",
            "channel_index": 3,
            "region_pixels": 12,
            "record_start": 10,
            "record_stop": 16,
            "duration_records": 6,
            "period_start_records": 4,
            "period_stop_records": 4,
            "period_width_records": 0,
            "period_width_bins": 1,
            "peak_period_records": 4,
            "freq_start_mhz": 38.0,
            "freq_stop_mhz": 38.2,
            "bandwidth_mhz": 0.2,
            "peak_record": 12,
            "peak_freq_mhz": 38.1,
            "peak_score": 8.0,
            "mean_score": 6.0,
            "integrated_score": 12.0,
            "activity_mean": 1.0,
            "activity_max": 2.0,
            "activity_raw_mean": 3.0,
            "activity_raw_max": 4.0,
            "noise_floor": 0.5,
            "period_peak_prominence": 3.0,
            "block_channel_start": 0,
            "block_channel_stop": 8,
        },
        run_id="run-a",
        source_file="data/example.2C",
        block_id="block_0001",
        tsamp_seconds=0.5,
    )

    assert row["schema_version"] == 1
    assert row["run_id"] == "run-a"
    assert row["source_file"] == "data/example.2C"
    assert row["block_id"] == "block_0001"
    assert row["peak_period_seconds"] == 2.0
    assert row["duration_seconds"] == 3.0
    assert row["peak_time_seconds"] == 6.0
    assert set(RAW_CANDIDATE_FIELDNAMES).issubset(row.keys() | {"candidate_id"})
