from __future__ import annotations

import numpy as np

from ce4_period_search.validation import (
    ValidationConfig,
    best_acf_peak,
    best_fold_period,
    fft_periodogram_peak,
    period_grid,
    select_candidates,
    shuffle_null_pvalue,
    validation_period_bounds,
)


def _pulse_train(period: int = 8, size: int = 512) -> np.ndarray:
    values = np.zeros(size, dtype=np.float64)
    values[::period] = 10.0
    return values


def test_acf_periodogram_and_folding_recover_synthetic_period() -> None:
    values = _pulse_train(period=8)

    acf = best_acf_peak(values, min_lag=6, max_lag=10)
    periodogram = fft_periodogram_peak(values, min_period=6, max_period=10)
    folding = best_fold_period(values, period_grid(6, 10), fold_bins=16)

    assert acf["acf_best_lag_records"] == 8.0
    assert abs(periodogram["periodogram_best_period_records"] - 8.0) < 1e-6
    assert folding["folding_best_period_records"] == 8.0
    assert folding["fold_profile_snr"] > 1.0


def test_shuffle_null_pvalue_is_bounded_and_deterministic() -> None:
    values = _pulse_train(period=8)
    rng = np.random.default_rng(123)
    result = shuffle_null_pvalue(values, period_grid(6, 10), fold_bins=16, shuffle_trials=12, rng=rng)

    assert result["shuffle_trials"] == 12.0
    assert 0.0 < result["shuffle_pvalue"] <= 1.0
    assert result["observed_metric"] > 1.0


def test_select_candidates_skips_vetoed_by_default() -> None:
    rows = [
        {"candidate_id": "1", "candidate_status": "vetoed"},
        {"candidate_id": "2", "candidate_status": "needs_validation"},
    ]

    selected = select_candidates(rows, ValidationConfig(include_vetoed=False, max_candidates=10))

    assert [row["candidate_id"] for row in selected] == ["2"]


def test_validation_period_bounds_respect_series_and_config_limits() -> None:
    config = ValidationConfig(period_search_radius=2.0, min_period_records=2, max_period_records=100)

    assert validation_period_bounds(8, series_size=512, config=config) == (4, 16)
    assert validation_period_bounds(8, series_size=20, config=config) == (4, 10)
