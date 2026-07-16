from __future__ import annotations

import numpy as np

from period_profile_algorithms import (
    PeriodProfileAlgorithm,
    evaluate_period_profile,
    period_profile_catalog,
)


def _ridge_map(
    periods: np.ndarray,
    records: int,
    components: tuple[tuple[float, float, float], ...],
) -> np.ndarray:
    frequencies = 1.0 / periods
    values = np.zeros((periods.size, records), dtype=np.float32)
    for period, amplitude, fractional_width in components:
        center = 1.0 / period
        width = max(center * fractional_width, 1e-6)
        ridge = amplitude * np.exp(-0.5 * ((frequencies - center) / width) ** 2)
        values += ridge[:, None].astype(np.float32)
    return values


def _algorithm(**overrides) -> PeriodProfileAlgorithm:
    values = {
        "name": "test_filter",
        "reducer": "mean",
        "score_mode": "geometric",
        "smooth_bins": 1,
        "width_fraction": 0.50,
        "min_width_bins": 2,
        "min_peak_strength": 0.75,
        "min_integrated_strength": 1.25,
        "min_band_persistence": 0.35,
        "harmonic_weight": 0.20,
    }
    values.update(overrides)
    return PeriodProfileAlgorithm(**values)


def test_catalog_is_large_but_scientifically_focused() -> None:
    catalog = period_profile_catalog()
    assert len(catalog) >= 100
    assert len({item.name for item in catalog}) == len(catalog)
    cprf = next(item for item in catalog if item.name == "cprf_absolute_ridge_c35_r140")
    assert cprf.min_peak_strength > 0.0
    assert cprf.min_width_bins >= 2
    assert cprf.min_integrated_strength == 0.0
    assert cprf.min_band_persistence == 0.0
    assert {item.reducer for item in catalog} == {
        "mean",
        "rms",
        "occupied_mean",
        "persistence_sqrt",
        "stable_mean",
        "top_fraction_mean",
    }


def test_frozen_concentrated_cprf_matches_ranked_configuration() -> None:
    catalog = {item.name: item for item in period_profile_catalog()}
    frozen = catalog["cprf_concentrated_ridge_c45"]
    ranked = catalog["pbsf_focus_concentration_c45"]

    assert {key: value for key, value in frozen.to_dict().items() if key != "name"} == {
        key: value for key, value in ranked.to_dict().items() if key != "name"
    }
    periods = np.geomspace(10.0, 1000.0, 80)
    score_map = _ridge_map(periods, 320, ((64.0, 3.0, 0.055),))
    frozen_result = evaluate_period_profile(score_map, periods, frozen)
    ranked_result = evaluate_period_profile(score_map, periods, ranked)
    assert frozen_result.accepted == ranked_result.accepted
    assert frozen_result.peak_index == ranked_result.peak_index
    assert frozen_result.band_start_index == ranked_result.band_start_index
    assert frozen_result.band_stop_index == ranked_result.band_stop_index
    np.testing.assert_array_equal(frozen_result.profile, ranked_result.profile)


def test_main_peak_does_not_require_harmonics() -> None:
    periods = np.geomspace(10.0, 200.0, 96)
    score_map = _ridge_map(periods, 320, ((64.0, 3.0, 0.055),))
    result = evaluate_period_profile(score_map, periods, _algorithm())
    assert result.accepted
    assert abs(result.peak_period_records - 64.0) / 64.0 < 0.06
    assert result.harmonic_support_count == 0


def test_harmonics_are_found_in_frequency_not_period_multiples() -> None:
    periods = np.geomspace(10.0, 200.0, 96)
    score_map = _ridge_map(
        periods,
        320,
        (
            (72.0, 3.0, 0.055),
            (36.0, 1.1, 0.055),
            (24.0, 0.8, 0.055),
        ),
    )
    result = evaluate_period_profile(score_map, periods, _algorithm())
    assert result.accepted
    assert abs(result.peak_period_records - 72.0) / 72.0 < 0.06
    assert result.harmonic_2_score > 0.20
    assert result.harmonic_3_score > 0.15
    assert result.harmonic_support_count == 2


def test_single_bin_spike_fails_width_gate() -> None:
    periods = np.geomspace(10.0, 200.0, 96)
    score_map = np.zeros((periods.size, 320), dtype=np.float32)
    score_map[42, :] = 20.0
    result = evaluate_period_profile(
        score_map,
        periods,
        _algorithm(min_width_bins=3, harmonic_weight=0.0),
    )
    assert not result.accepted
    assert result.width_bins == 1
