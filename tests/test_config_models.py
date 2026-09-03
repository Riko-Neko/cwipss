from __future__ import annotations

import pytest

from cwipss.config import cwt_config_from_mapping, cwt_config_to_nested_dict
from cwipss.signal.cpro import CPRO_DETECTOR


def test_structured_cpro_config_round_trips() -> None:
    config = cwt_config_from_mapping(
        {
            "schema_version": 3,
            "input": {"path": "data/example.2C"},
            "scan": {"period_count": 40, "cwt_backend": "cuda", "cuda_device": 1},
            "detection": {
                "detector": CPRO_DETECTOR,
                "cpro_threshold_snr": 24.0,
                "cpro_texture_quantile": 0.9,
                "cpro_shape_contrast_softness": 0.2,
                "cpro_min_continuity_mean": 0.5,
                "cpro_min_ridge_lock": 0.9,
                "pelt_penalty": 6.0,
                "pelt_threads": 4,
                "cuda_max_pending_blocks": 2,
                "cprf_min_band_concentration": 0.55,
                "cprf_min_local_contrast": 3.6,
                "cprf_min_integrated_strength": 0.0,
            },
        }
    )
    assert config.detector == CPRO_DETECTOR
    assert config.cpro_threshold_snr == 24.0
    assert config.pelt_penalty == 6.0
    assert config.pelt_threads == 4
    assert config.cuda_max_pending_blocks == 2
    nested = cwt_config_to_nested_dict(config)
    assert nested["schema_version"] == 6
    assert nested["detection"]["cpro_texture_quantile"] == 0.9
    assert nested["detection"]["cpro_shape_contrast_softness"] == 0.2
    assert nested["detection"]["cpro_min_continuity_mean"] == 0.5
    assert nested["detection"]["cpro_min_ridge_lock"] == 0.9
    assert nested["detection"]["pelt_penalty"] == 6.0
    assert nested["detection"]["cuda_max_pending_blocks"] == 2
    assert nested["detection"]["cprf_min_band_concentration"] == 0.55
    assert nested["detection"]["cprf_min_local_contrast"] == 3.6
    assert nested["detection"]["cprf_min_integrated_strength"] == 0.0


def test_flat_config_remains_supported_with_overrides() -> None:
    config = cwt_config_from_mapping(
        {"input": "data/example.2C", "period_count": 30},
        overrides={"period_count": 50, "cpro_threshold_snr": 28.0},
    )
    assert config.period_count == 50
    assert config.cpro_threshold_snr == 28.0


def test_unknown_science_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown config key"):
        cwt_config_from_mapping({"detection": {"unknown_detector_parameter": 16.0}})


def test_non_cpro_detector_is_rejected_without_fallback() -> None:
    with pytest.raises(ValueError, match="Unsupported detector"):
        cwt_config_from_mapping({"detection": {"detector": "legacy"}})


def test_invalid_cpro_parameter_is_rejected() -> None:
    with pytest.raises(ValueError, match="cpro_shape_contrast_softness"):
        cwt_config_from_mapping({"detection": {"cpro_shape_contrast_softness": 0.0}})


def test_removed_duration_gate_is_rejected_as_unknown_science() -> None:
    with pytest.raises(ValueError, match="Unknown config key"):
        cwt_config_from_mapping(
            {"detection": {"pelt_segment_min_duration_records": 0}}
        )
