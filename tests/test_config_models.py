from __future__ import annotations

import numpy as np
import pytest

from cwipss.config import (
    CWTSearchConfig,
    cwt_config_from_mapping,
    cwt_config_to_nested_dict,
    resolve_cwt_period_domain,
)
from cwipss.signal.cpro import CPRO_DETECTOR, cpro_period_mask
from cwipss.signal.cwt import period_grid_records


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


@pytest.mark.parametrize("candidate_min,candidate_max", [(10.0, 200.0), (200.0, 4000.0)])
def test_automatic_cwt_domain_has_exact_minimum_safe_margin(
    candidate_min: float,
    candidate_max: float,
) -> None:
    config = cwt_config_from_mapping(
        {
            "detection": {
                "candidate_period_min_records": candidate_min,
                "candidate_period_max_records": candidate_max,
            }
        }
    )
    periods = period_grid_records(
        config.period_min_records,
        config.period_max_records,
        config.period_count,
        config.period_spacing,
    )
    selected = np.flatnonzero(cpro_period_mask(periods, candidate_min, candidate_max))

    assert config.period_count == 69
    assert selected[0] == 8
    assert periods.size - 1 - selected[-1] == 8
    assert periods[selected[0]] == pytest.approx(candidate_min)
    assert periods[selected[-1]] == pytest.approx(candidate_max)


def test_explicit_cwt_domain_is_preserved() -> None:
    config = cwt_config_from_mapping(
        {
            "scan": {
                "period_min_records": 2.0,
                "period_max_records": 512.0,
                "period_count": 96,
            }
        }
    )
    assert config.period_min_records == 2.0
    assert config.period_max_records == 512.0
    assert config.period_count == 96


def test_automatic_cwt_margin_tracks_cpro_context_widths() -> None:
    config = cwt_config_from_mapping(
        {
            "detection": {
                "cpro_period_context_bins": 31,
                "cpro_period_support_bins": 5,
            }
        }
    )
    periods = period_grid_records(
        config.period_min_records,
        config.period_max_records,
        config.period_count,
        config.period_spacing,
    )
    selected = np.flatnonzero(
        cpro_period_mask(
            periods,
            config.candidate_period_min_records,
            config.candidate_period_max_records,
        )
    )

    assert selected[0] == 17
    assert periods.size - 1 - selected[-1] == 17


def test_partial_explicit_cwt_domain_is_rejected() -> None:
    with pytest.raises(ValueError, match="must both be set"):
        cwt_config_from_mapping({"scan": {"period_min_records": 2.0}})


def test_direct_config_can_be_resolved() -> None:
    config = resolve_cwt_period_domain(CWTSearchConfig())
    assert config.period_min_records is not None
    assert config.period_max_records is not None
    assert config.period_count is not None
