from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
for path in (PROJECT_DIR / "src", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from persistent_occupancy import (  # noqa: E402
    PersistentOccupancyParameters,
    difference_noise_std,
    impulse_cwt_noise_gain,
    persistent_occupancy_catalog,
    persistent_occupancy_windows,
    regularize_time_mask,
)


PARAMS = PersistentOccupancyParameters(
    name="test",
    threshold_snr=4.0,
    support_records=21,
    min_occupancy=0.5,
    min_duration_records=20,
    max_gap_records=5,
)


def test_short_bright_transient_is_rejected_but_long_band_survives() -> None:
    power = np.ones((3, 160), dtype=np.float32)
    power[1, 30:34] = 1000.0
    power[1, 80:130] = 10.0

    result = persistent_occupancy_windows(
        power,
        noise_std=1.0,
        noise_gain=np.ones(3, dtype=np.float32),
        params=PARAMS,
    )

    assert not np.any(result.active_mask[20:50])
    assert np.all(result.active_mask[90:120])
    assert len(result.windows) == 1


def test_dense_band_survives_bounded_gaps() -> None:
    power = np.ones((1, 180), dtype=np.float32)
    power[0, 40:150] = 8.0
    power[0, 70:74] = 1.0
    power[0, 100:104] = 1.0

    result = persistent_occupancy_windows(
        power,
        noise_std=1.0,
        noise_gain=np.ones(1, dtype=np.float32),
        params=PARAMS,
    )

    assert len(result.windows) == 1
    assert result.windows[0]["duration_records"] >= 100


def test_mask_regularization_does_not_merge_large_gaps() -> None:
    mask = np.zeros(100, dtype=bool)
    mask[5:30] = True
    mask[34:60] = True
    mask[75:95] = True

    result = regularize_time_mask(mask, max_gap=5, min_duration=20)

    assert np.all(result[5:60])
    assert not np.any(result[60:75])
    assert np.all(result[75:95])


def test_amplitude_scaling_preserves_mask_and_absolute_score() -> None:
    power = np.ones((2, 120), dtype=np.float32)
    power[:, 20:100] = 9.0
    base = persistent_occupancy_windows(
        power,
        noise_std=1.0,
        noise_gain=np.ones(2, dtype=np.float32),
        params=PARAMS,
    )
    scaled = persistent_occupancy_windows(
        25.0 * power,
        noise_std=5.0,
        noise_gain=np.ones(2, dtype=np.float32),
        params=PARAMS,
    )

    np.testing.assert_array_equal(scaled.active_mask, base.active_mask)
    np.testing.assert_allclose(scaled.activity, 25.0 * base.activity, rtol=2e-6)


def test_difference_noise_sigma_matches_gaussian_scale() -> None:
    rng = np.random.default_rng(20260715)
    values = rng.normal(0.0, 3.0, size=100_000)

    sigma = difference_noise_std(values)

    assert abs(sigma - 3.0) < 0.05


def test_impulse_gain_matches_white_noise_cwt_mean() -> None:
    from cwipss.signal.cwt import cwt_power_cube

    periods = np.array([10.0, 32.0, 100.0, 320.0], dtype=np.float64)
    gain = impulse_cwt_noise_gain(periods, wavelet="cmor1.5-1.0")
    rng = np.random.default_rng(11)
    noise = rng.normal(size=(65_536, 1)).astype(np.float32)
    power = cwt_power_cube(
        noise,
        periods,
        wavelet="cmor1.5-1.0",
        normalize_channels=False,
    )[:, 4096:-4096, 0]

    np.testing.assert_allclose(np.mean(power, axis=1), gain, rtol=0.12)


def test_second_time_consensus_rejects_isolated_brightness() -> None:
    params = PersistentOccupancyParameters(
        name="consensus",
        threshold_snr=4.0,
        support_records=1,
        min_occupancy=1.0,
        min_duration_records=1,
        max_gap_records=0,
        window_support_records=257,
        min_window_occupancy=0.5,
    )
    short = np.ones((3, 700), dtype=np.float32)
    short[1, 250:360] = 10.0
    long = np.ones((3, 700), dtype=np.float32)
    long[1, 250:420] = 10.0

    short_result = persistent_occupancy_windows(
        short, noise_std=1.0, noise_gain=np.ones(3, dtype=np.float32), params=params
    )
    long_result = persistent_occupancy_windows(
        long, noise_std=1.0, noise_gain=np.ones(3, dtype=np.float32), params=params
    )

    assert not np.any(short_result.active_mask)
    assert np.any(long_result.active_mask)


def test_cpro_catalog_is_focused_and_unique() -> None:
    catalog = persistent_occupancy_catalog()
    names = [params.name for params in catalog]

    assert len(catalog) == 432
    assert len(set(names)) == len(names)
    assert "cpro_e32_q825_r150_o70_b5_w385_v40_d096" in names


def test_final_cpro_preserves_full_duration_horizontal_ridge() -> None:
    params = next(
        item
        for item in persistent_occupancy_catalog()
        if item.name == "cpro_e32_q938_r150_o65_b3_w385_v40_d096"
    )
    power = np.ones((18, 1800), dtype=np.float32)
    power[7:10, :] = 100.0

    result = persistent_occupancy_windows(
        power, noise_std=1.0, noise_gain=np.ones(18, dtype=np.float32), params=params
    )

    assert float(np.mean(result.active_mask)) > 0.95


def test_final_cpro_keeps_absolute_score_under_scale_change() -> None:
    params = next(
        item
        for item in persistent_occupancy_catalog()
        if item.name == "cpro_e32_q938_r150_o65_b3_w385_v40_d096"
    )
    power = np.ones((18, 1800), dtype=np.float32)
    power[7:10, 300:1500] = 100.0
    base = persistent_occupancy_windows(
        power, noise_std=1.0, noise_gain=np.ones(18, dtype=np.float32), params=params
    )
    scaled = persistent_occupancy_windows(
        25.0 * power,
        noise_std=5.0,
        noise_gain=np.ones(18, dtype=np.float32),
        params=params,
    )

    np.testing.assert_array_equal(scaled.active_mask, base.active_mask)
    np.testing.assert_allclose(scaled.activity, 25.0 * base.activity, rtol=3e-6)


def test_final_cpro_rejects_period_switching_texture() -> None:
    params = next(
        item
        for item in persistent_occupancy_catalog()
        if item.name == "cpro_e32_q938_r150_o65_b3_w385_v40_d096"
    )
    power = np.ones((18, 1800), dtype=np.float32)
    centers = (3, 8, 13)
    for index, start in enumerate(range(0, power.shape[1], 100)):
        center = centers[index % len(centers)]
        power[center - 1 : center + 2, start : start + 100] = 100.0

    result = persistent_occupancy_windows(
        power, noise_std=1.0, noise_gain=np.ones(18, dtype=np.float32), params=params
    )

    assert float(np.mean(result.active_mask)) < 0.05
