from __future__ import annotations

import numpy as np

from ce4_period_search.activity import (
    coherent_structure_map,
    low_fraction_noise_floor,
    relative_excess,
    signed_trimmed_period_activity,
)
from ce4_period_search.cwt import aggregate_cwt_time, cwt_power_cube, period_grid_records
from ce4_period_search.detection import detect_block_periods
from ce4_period_search.windows import pelt_mean_shift


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


def test_low_fraction_noise_floor_ignores_high_power_tail() -> None:
    values = np.ones((10, 100), dtype=np.float32)
    values[:, 80:] = 100.0

    floor = low_fraction_noise_floor(values, fraction=0.2)

    assert 0.9 <= floor <= 1.1


def test_signed_activity_keeps_negative_excess() -> None:
    excess = np.array([[-1.0, 2.0], [1.0, 4.0], [3.0, 6.0]], dtype=np.float32)

    activity = signed_trimmed_period_activity(excess, trim_low=0.0, trim_high=1.0)

    assert activity.tolist() == [1.0, 4.0]


def test_coherent_structure_map_suppresses_sparse_texture() -> None:
    texture = np.zeros((16, 128), dtype=np.float32)
    texture[3, 20] = 20.0
    texture[9, 90] = 20.0
    band = np.zeros((16, 128), dtype=np.float32)
    band[7:10, 40:100] = 5.0

    texture_structured = coherent_structure_map(
        texture,
        baseline_quantile=0.1,
        scale_quantile=0.2,
        z_threshold=1.0,
        time_support_records=16,
        period_support_bins=3,
        min_support_fraction=0.15,
    )
    band_structured = coherent_structure_map(
        band,
        baseline_quantile=0.1,
        scale_quantile=0.2,
        z_threshold=1.0,
        time_support_records=16,
        period_support_bins=3,
        min_support_fraction=0.15,
    )

    assert float(np.nanmax(texture_structured)) < 1e-3
    assert float(np.nanmean(band_structured[:, 50:90])) > 0.1


def test_pelt_mean_shift_finds_active_segment() -> None:
    activity = np.zeros(120, dtype=np.float32)
    activity[40:90] = 4.0

    segments = pelt_mean_shift(activity, penalty=5.0, min_size=10)
    bounds = [(segment.start, segment.stop) for segment in segments]

    assert any(start <= 40 and stop >= 90 for start, stop in bounds)


def test_lowfloor_pelt_detector_finds_windowed_period_peak() -> None:
    periods = period_grid_records(2, 128, 48)
    target_period_idx = int(np.argmin(np.abs(periods - 64.0)))
    target_channel = 3
    power = np.ones((periods.size, 128, 5), dtype=np.float32)
    power[target_period_idx - 2:target_period_idx + 3, 36:96, target_channel] = 50.0

    rows, windows = detect_block_periods(
        power_cube=power,
        periods=periods,
        freqs_mhz=np.arange(5, dtype=np.float64),
        record_start=10,
        candidate_period_min_records=10.0,
        candidate_period_max_records=200.0,
        noise_floor_fraction=0.2,
        excess_eps_fraction=1e-6,
        structure_baseline_quantile=0.1,
        structure_scale_quantile=0.2,
        structure_z_threshold=0.0,
        structure_time_support_records=3,
        structure_period_support_bins=1,
        structure_min_support_fraction=0.0,
        activity_trim_low=0.0,
        activity_trim_high=1.0,
        activity_smooth_records=3,
        pelt_penalty=5.0,
        pelt_min_size_records=8,
        window_min_duration_records=16,
        window_min_activity_mean=0.5,
        window_min_activity_raw_mean=0.0,
        window_merge_gap_records=4,
        profile_min_prominence=0.1,
        profile_max_peaks_per_window=2,
        max_candidates_per_channel=2,
        max_candidates=10,
    )

    assert len(windows) >= 1
    assert len(rows) >= 1
    assert rows[0]["detection_method"] == "single_channel_lowfloor_pelt_profile"
    assert rows[0]["channel_index"] == target_channel
    assert rows[0]["peak_freq_mhz"] == float(target_channel)
    assert abs(rows[0]["peak_period_records"] - float(periods[target_period_idx])) < 12.0
    assert rows[0]["record_start"] <= 50
    assert rows[0]["record_stop"] >= 90
    assert rows[0]["integrated_score"] > 0


def test_default_profile_keeps_one_period_family_per_window() -> None:
    periods = period_grid_records(2, 128, 48)
    first_idx = int(np.argmin(np.abs(periods - 32.0)))
    second_idx = int(np.argmin(np.abs(periods - 64.0)))
    power = np.ones((periods.size, 128, 1), dtype=np.float32)
    power[first_idx - 1:first_idx + 2, 32:96, 0] = 30.0
    power[second_idx - 1:second_idx + 2, 32:96, 0] = 60.0

    rows, windows = detect_block_periods(
        power_cube=power,
        periods=periods,
        freqs_mhz=np.array([1.0], dtype=np.float64),
        record_start=0,
        candidate_period_min_records=10.0,
        candidate_period_max_records=200.0,
        noise_floor_fraction=0.2,
        excess_eps_fraction=1e-6,
        structure_baseline_quantile=0.1,
        structure_scale_quantile=0.2,
        structure_z_threshold=0.0,
        structure_time_support_records=3,
        structure_period_support_bins=1,
        structure_min_support_fraction=0.0,
        activity_trim_low=0.0,
        activity_trim_high=1.0,
        activity_smooth_records=3,
        pelt_penalty=5.0,
        pelt_min_size_records=8,
        window_min_duration_records=16,
        window_min_activity_mean=0.5,
        window_min_activity_raw_mean=0.0,
        window_merge_gap_records=4,
        profile_min_prominence=0.1,
        profile_max_peaks_per_window=1,
        max_candidates_per_channel=3,
        max_candidates=10,
    )

    assert len(windows) >= 1
    assert len(rows) == len({row["window_id"] for row in rows})
    assert len(rows) == 1
    assert min(
        abs(rows[0]["peak_period_records"] - float(periods[first_idx])),
        abs(rows[0]["peak_period_records"] - float(periods[second_idx])),
    ) < 12.0


def test_relative_excess_is_zero_near_noise_floor() -> None:
    power = np.ones((4, 5), dtype=np.float32)
    z = relative_excess(power, 1.0)
    assert np.max(np.abs(z)) < 1e-5
