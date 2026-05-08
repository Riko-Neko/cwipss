from __future__ import annotations

import pytest

from ce4_period_search.config import swt_config_from_mapping, swt_config_to_nested_dict
from ce4_period_search.models import RAW_CANDIDATE_FIELDNAMES, normalize_candidate_row


def test_structured_config_maps_to_resolved_dataclass() -> None:
    config = swt_config_from_mapping(
        {
            "schema_version": 1,
            "input": {"path": "data/example.2C"},
            "scan": {"f_start": 38.0, "f_stop": 39.0, "levels": 4},
            "detection": {"threshold": 6.5, "min_pixels": 8},
            "veto": {"enabled": True, "max_bandwidth_fraction": 0.8},
            "validation": {"max_candidates": 12, "shuffle_trials": 20},
            "output": {"output_dir": "runs", "run_id": "smoke"},
        }
    )

    assert config.input == "data/example.2C"
    assert config.f_start == 38.0
    assert config.f_stop == 39.0
    assert config.levels == 4
    assert config.threshold == 6.5
    assert config.min_pixels == 8
    assert config.veto_enabled is True
    assert config.veto_max_bandwidth_fraction == 0.8
    assert config.validation_max_candidates == 12
    assert config.validation_shuffle_trials == 20
    assert config.run_id == "smoke"

    nested = swt_config_to_nested_dict(config)
    assert nested["input"]["path"] == "data/example.2C"
    assert nested["scan"]["levels"] == 4
    assert nested["detection"]["threshold"] == 6.5
    assert nested["veto"]["max_bandwidth_fraction"] == 0.8
    assert nested["validation"]["shuffle_trials"] == 20


def test_flat_config_remains_supported_with_overrides() -> None:
    config = swt_config_from_mapping(
        {"input": "data/example.2C", "levels": 3},
        overrides={"levels": 5, "threshold": 7.0},
    )

    assert config.input == "data/example.2C"
    assert config.levels == 5
    assert config.threshold == 7.0


def test_unknown_config_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown config key"):
        swt_config_from_mapping({"scan": {"typo": 1}})


def test_normalize_candidate_row_adds_schema_and_seconds() -> None:
    row = normalize_candidate_row(
        {
            "swt_level": 2,
            "approx_scale_records": 4,
            "component_id": 1,
            "area_pixels": 12,
            "record_start": 10,
            "record_stop": 16,
            "duration_records": 6,
            "freq_start_mhz": 38.0,
            "freq_stop_mhz": 38.2,
            "bandwidth_mhz": 0.2,
            "peak_record": 12,
            "peak_freq_mhz": 38.1,
            "peak_score": 8.0,
            "mean_score": 6.0,
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
    assert row["approx_scale_seconds"] == 2.0
    assert row["duration_seconds"] == 3.0
    assert row["peak_time_seconds"] == 6.0
    assert set(RAW_CANDIDATE_FIELDNAMES).issubset(row.keys() | {"candidate_id"})
