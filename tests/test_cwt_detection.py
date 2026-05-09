from __future__ import annotations

import numpy as np

from ce4_period_search.cwt import aggregate_cwt_time, cwt_power_cube, period_grid_records


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
