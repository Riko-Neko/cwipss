from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PERF_DIR = Path(__file__).resolve().parent
if str(PERF_DIR) not in sys.path:
    sys.path.insert(0, str(PERF_DIR))

from single_map_activity_algorithms import (  # noqa: E402
    _horizontal_structure_weight,
    compute_absolute_persistent_energy,
    single_map_algorithm_names,
    single_map_absolute_persistence_names,
)
from cwt_activity_algorithms import resolve_cwt_activity_algorithms  # noqa: E402
from cwt_activity_rank import _window_standardize  # noqa: E402


def test_absolute_persistent_energy_preserves_constant_power_at_edges() -> None:
    power = np.full((3, 41), 7.5, dtype=np.float32)

    score = compute_absolute_persistent_energy(power, width=17, exponent=0.5)

    np.testing.assert_allclose(score, power, rtol=2e-6, atol=2e-6)


def test_absolute_persistent_energy_is_power_scale_equivariant() -> None:
    rng = np.random.default_rng(20260714)
    power = rng.lognormal(mean=-2.0, sigma=1.0, size=(7, 101)).astype(np.float32)

    base = compute_absolute_persistent_energy(power, width=33, exponent=0.5)
    scaled = compute_absolute_persistent_energy(9.0 * power, width=33, exponent=0.5)

    np.testing.assert_allclose(scaled, 9.0 * base, rtol=3e-6, atol=1e-7)


def test_absolute_persistent_energy_suppresses_sparse_bright_texture() -> None:
    persistent = np.ones((1, 65), dtype=np.float32)
    sparse = np.zeros((1, 65), dtype=np.float32)
    sparse[0, 32] = 65.0

    persistent_score = compute_absolute_persistent_energy(persistent, width=65, exponent=0.5)
    sparse_score = compute_absolute_persistent_energy(sparse, width=65, exponent=0.5)

    assert persistent_score[0, 32] == 1.0
    np.testing.assert_allclose(sparse_score[0, 32], 1.0 / 65.0, rtol=2e-6)


def test_absolute_persistence_catalog_is_unique() -> None:
    names = single_map_absolute_persistence_names()

    assert len(names) == 38
    assert len(set(names)) == len(names)
    assert not set(names).intersection(single_map_algorithm_names())


def test_rejected_absolute_algorithms_require_explicit_resolution() -> None:
    rejected_name = "sm_ape_l0065_r050_k1"

    assert rejected_name not in {item.name for item in resolve_cwt_activity_algorithms(("all",))}
    explicit = resolve_cwt_activity_algorithms((rejected_name,))[0]

    assert explicit.family.startswith("rejected_")


def test_absolute_window_standardization_handles_native_power_units() -> None:
    algorithm = resolve_cwt_activity_algorithms(("sm_ape_l0065_r050_k1",))[0]
    activity = np.array([1.0e-15, 1.1e-15, 3.0e-15], dtype=np.float32)

    standardized = _window_standardize(activity, algorithm)

    assert standardized[-1] > 1.0
    assert np.all(np.isfinite(standardized))


def test_horizontal_structure_weight_prefers_time_elongated_band() -> None:
    horizontal = np.zeros((31, 101), dtype=np.float32)
    horizontal[14:17, 10:91] = 1.0
    vertical = np.zeros_like(horizontal)
    vertical[3:28, 48:53] = 1.0

    horizontal_weight = _horizontal_structure_weight(horizontal, time_width=17, period_width=5)
    vertical_weight = _horizontal_structure_weight(vertical, time_width=17, period_width=5)

    assert float(np.mean(horizontal_weight[14:17, 20:81])) > 0.5
    assert float(np.mean(vertical_weight[5:26, 48:53])) < 0.1
