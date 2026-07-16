from __future__ import annotations

import numpy as np
import pytest

from cwipss.signal.cpro import CPROParameters, cpro_activity, difference_noise_std
from cwipss.signal.detection import resolve_channel_candidate_cap


def _small_params() -> CPROParameters:
    return CPROParameters(
        threshold_snr=2.0,
        texture_quantile=0.0,
        period_center_bins=1,
        period_context_bins=1,
        min_period_contrast=0.0,
        support_records=5,
        min_occupancy=0.6,
        period_support_bins=1,
        window_support_records=5,
        min_window_occupancy=0.4,
    )


def test_cpro_retains_persistent_absolute_ridge() -> None:
    power = np.ones((3, 128), dtype=np.float32)
    power[1, 24:104] = 12.0
    result = cpro_activity(
        power,
        noise_std=1.0,
        noise_gain=np.ones(3),
        params=_small_params(),
    )
    assert np.max(result.activity[40:90]) > 0.0
    assert np.any(result.active_mask[40:90])


def test_cpro_rejects_short_bright_transient() -> None:
    power = np.ones((3, 128), dtype=np.float32)
    power[1, 60:62] = 100.0
    result = cpro_activity(
        power,
        noise_std=1.0,
        noise_gain=np.ones(3),
        params=_small_params(),
    )
    assert not np.any(result.active_mask)


def test_cpro_activity_has_no_legacy_window_parameters() -> None:
    power = np.ones((3, 128), dtype=np.float32)
    power[1, 24:104] = 12.0
    params = _small_params()

    result = cpro_activity(power, noise_std=1.0, noise_gain=np.ones(3), params=params)

    assert "max_gap_records" not in params.__dict__
    assert "min_duration_records" not in params.__dict__
    assert np.max(result.activity) > 0.0


def test_cpro_activity_is_in_noise_calibrated_power_units() -> None:
    power = np.ones((3, 128), dtype=np.float32)
    power[1, 24:104] = 12.0

    base = cpro_activity(power, noise_std=1.0, noise_gain=np.ones(3), params=_small_params())
    scaled = cpro_activity(4.0 * power, noise_std=2.0, noise_gain=np.ones(3), params=_small_params())

    np.testing.assert_allclose(base.activity, scaled.activity)
    np.testing.assert_allclose(base.score_map, scaled.score_map)


def test_noise_calibration_has_no_degenerate_fallback() -> None:
    with pytest.raises(ValueError, match="positive"):
        difference_noise_std(np.ones(32, dtype=np.float32))


def test_candidate_cap_auto_scales_with_record_count() -> None:
    assert resolve_channel_candidate_cap("auto", 3.0 / 4096.0, 4096) == 3
    assert resolve_channel_candidate_cap("auto", 3.0 / 4096.0, 8192) == 6
