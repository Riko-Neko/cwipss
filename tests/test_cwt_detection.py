from __future__ import annotations

import numpy as np

from ce4_period_search.cwt import aggregate_cwt_time, cwt_power_cube, period_grid_records
from ce4_period_search.detection import channel_period_peak_score, summarize_channel_period_peaks


def test_cwt_power_cube_shape() -> None:
    rng = np.random.default_rng(123)
    data = rng.normal(size=(64, 3)).astype(np.float32)
    periods = period_grid_records(2, 16, 8)

    power = cwt_power_cube(data, periods, wavelet="cmor1.5-1.0")

    assert power.shape == (8, 64, 3)
    assert np.all(np.isfinite(power))


def test_cwt_time_aggregation_returns_period_channel_map() -> None:
    power = np.ones((5, 10, 2), dtype=np.float32)
    power[2, 5, 1] = 10.0

    response = aggregate_cwt_time(power, method="max")

    assert response.shape == (5, 2)
    assert response[2, 1] == 10.0


def test_channel_period_peak_detector_finds_single_channel_short_band() -> None:
    periods = period_grid_records(2, 128, 48)
    response = np.ones((periods.size, 5), dtype=np.float32)
    target_period_idx = 24
    target_channel = 3
    response[target_period_idx - 1:target_period_idx + 2, target_channel] = 50.0
    power = np.ones((periods.size, 64, 5), dtype=np.float32)
    power[target_period_idx, 17, target_channel] = 100.0

    score = channel_period_peak_score(
        np.log10(response + 1e-12),
        sigma_peak=1.0,
        sigma_background=8.0,
    )
    rows = summarize_channel_period_peaks(
        score=score,
        power_cube=power,
        periods=periods,
        freqs_mhz=np.arange(5, dtype=np.float64),
        record_start=10,
        record_stop=74,
        threshold=2.0,
        min_prominence=2.0,
        min_width_bins=1.0,
        max_width_bins=8.0,
        min_distance_bins=2,
        max_candidates_per_channel=1,
        max_components=10,
    )

    assert len(rows) == 1
    assert rows[0]["detection_method"] == "channel_period_peak_dog"
    assert rows[0]["channel_index"] == target_channel
    assert rows[0]["peak_freq_mhz"] == float(target_channel)
    assert rows[0]["peak_period_records"] == float(periods[target_period_idx])
    assert rows[0]["peak_record"] == 27
    assert rows[0]["peak_prominence"] >= 2.0
