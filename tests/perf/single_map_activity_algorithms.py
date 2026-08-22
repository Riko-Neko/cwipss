"""Strict single-CWT-map linear activity algorithms for scientific ranking."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage


MIN_SCALE = 1e-6


def _code(value: float) -> str:
    return f"{int(round(100 * value)):02d}"


def single_map_catalog(algorithm_class) -> list[Any]:
    """Build a broad O(P*T) grid whose only numerical input is one CWT map."""
    algorithms: list[Any] = []
    transforms = (
        ("rmad", "row_mad", "row-wise median/MAD signed calibration"),
        ("lmad", "log_row_mad", "row-wise log-power median/MAD calibration"),
        ("ps13", "period_sideband_3_13", "row-MAD plus 3/13-bin period sideband contrast"),
        ("ps21", "period_sideband_5_21", "row-MAD plus 5/21-bin period sideband contrast"),
    )
    gates = (
        (0.0, 0.0),
        (0.50, 0.10),
        (0.70, 0.20),
    )
    for short, transform, description in transforms:
        for cycles in (2.0, 4.0, 6.0, 8.0, 12.0):
            for support, floor in gates:
                name = f"sm_{short}_c{int(cycles):02d}_q{_code(support)}_f{_code(floor)}_k3"
                algorithms.append(
                    algorithm_class(
                        name=name,
                        family=f"single_map_{transform}",
                        description=(
                            f"Strict single-map {description}; {cycles:g}-cycle signed coherence, "
                            f"support={support:.2f}, floor={floor:.2f}, Top-3 pooling."
                        ),
                        method="single_map_coherent",
                        source_refs=("robust_time_frequency_calibration", "anisotropic_ridge_filter"),
                        complexity="O(P*T)",
                        migratability="Separable filters, robust reductions, and fixed-k partition.",
                        input_denoiser="none",
                        params={
                            "transform": transform,
                            "time_support_cycles": cycles,
                            "min_positive_support": support,
                            "score_floor": floor,
                            "clip_score": 6.0,
                            "top_k": 3,
                        },
                    )
                )
    # Sparse-evidence variants first offset the calibrated map, so persistence
    # measures genuinely elevated row-wise evidence rather than sign alone.
    for short, transform, description in transforms:
        for cycles in (4.0, 8.0, 12.0):
            for activation in (0.5, 1.0, 1.5):
                for support in (0.50, 0.70):
                    name = (
                        f"sm_{short}_a{_code(activation)}_c{int(cycles):02d}_"
                        f"q{_code(support)}_f20_k3"
                    )
                    algorithms.append(
                        algorithm_class(
                            name=name,
                            family=f"single_map_sparse_{transform}",
                            description=(
                                f"Strict single-map {description}; activation={activation:.2f}, "
                                f"{cycles:g}-cycle coherence, support={support:.2f}, 0.20 floor."
                            ),
                            method="single_map_coherent",
                            source_refs=("robust_CFAR_threshold", "persistence_detector"),
                            complexity="O(P*T)",
                            migratability="Robust reductions, scalar offset, separable filters, Top-3 pooling.",
                            input_denoiser="none",
                            params={
                                "transform": transform,
                                "activation_threshold": activation,
                                "time_support_cycles": cycles,
                                "min_positive_support": support,
                                "score_floor": 0.20,
                                "clip_score": 6.0,
                                "top_k": 3,
                            },
                        )
                    )
    # Pooling ablations around the scientifically conservative settings.
    for short, transform, description in transforms:
        for cycles in (4.0, 8.0):
            for top_k in (1, 5):
                algorithms.append(
                    algorithm_class(
                        name=f"sm_{short}_c{int(cycles):02d}_q70_f20_k{top_k}",
                        family=f"single_map_{transform}_pooling",
                        description=(
                            f"Strict single-map {description}; {cycles:g}-cycle coherence, "
                            f"70% support, 0.20 floor, Top-{top_k} pooling."
                        ),
                        method="single_map_coherent",
                        source_refs=("robust_time_frequency_calibration", "top_k_pooling"),
                        complexity="O(P*T)",
                        migratability="Separable filters, robust reductions, and fixed-k partition.",
                        input_denoiser="none",
                        params={
                            "transform": transform,
                            "time_support_cycles": cycles,
                            "min_positive_support": 0.70,
                            "score_floor": 0.20,
                            "clip_score": 6.0,
                            "top_k": top_k,
                        },
                    )
                )
    algorithms.extend(single_map_ridge_refinement_catalog(algorithm_class))
    algorithms.extend(single_map_false_window_catalog(algorithm_class))
    algorithms.extend(single_map_false_window_refinement_catalog(algorithm_class))
    algorithms.extend(single_map_cpro_catalog(algorithm_class))
    return algorithms


def single_map_cpro_catalog(algorithm_class) -> list[Any]:
    """CPRO activity-only baseline and one-factor/interacting ablations."""
    base: dict[str, float | int] = {
        "threshold_snr": 32.0,
        "texture_quantile": 0.9375,
        "period_center_bins": 3,
        "period_context_bins": 15,
        "min_period_contrast": 1.5,
        "support_records": 65,
        "min_occupancy": 0.65,
        "period_support_bins": 3,
        "window_support_records": 769,
        "min_window_occupancy": 0.40,
    }
    configurations: dict[str, dict[str, float | int | str]] = {"sm_cpro_base": dict(base)}
    sweeps = {
        "e": ("threshold_snr", (8.0, 16.0, 24.0, 48.0, 64.0)),
        "q": ("texture_quantile", (0.80, 0.85, 0.90, 0.925, 0.95, 0.975)),
        "r": ("min_period_contrast", (0.0, 1.10, 1.25, 1.75, 2.0)),
        "s": ("support_records", (17, 33, 129, 257)),
        "o": ("min_occupancy", (0.40, 0.50, 0.60, 0.70, 0.80)),
        "b": ("period_support_bins", (1, 2, 5, 7)),
        "w": ("window_support_records", (129, 257, 513, 769)),
        "v": ("min_window_occupancy", (0.20, 0.30, 0.50, 0.60, 0.70)),
    }
    for code, (field, values) in sweeps.items():
        for value in values:
            params = dict(base)
            params[field] = value
            token = int(round(1000 * value)) if isinstance(value, float) and value < 1.0 else int(round(value))
            configurations[f"sm_cpro_{code}{token:03d}"] = params
    for support, occupancy in ((33, 0.50), (33, 0.70), (65, 0.50), (65, 0.70), (129, 0.50), (129, 0.70)):
        params = dict(base)
        params.update(support_records=support, min_occupancy=occupancy)
        configurations[f"sm_cpro_so{support:03d}_{int(round(100 * occupancy)):02d}"] = params
    for support, occupancy in ((257, 0.30), (257, 0.50), (385, 0.30), (385, 0.50), (513, 0.30), (513, 0.50)):
        params = dict(base)
        params.update(window_support_records=support, min_window_occupancy=occupancy)
        configurations[f"sm_cpro_wv{support:03d}_{int(round(100 * occupancy)):02d}"] = params
    return [
        algorithm_class(
            name=name,
            family="single_map_cpro_activity",
            description=(
                "CPRO used strictly as a single-map 2D-to-1D activity compressor; "
                "its internal window list is excluded and native PELT defines stage 2."
            ),
            method="single_map_cpro_activity",
            source_refs=("absolute_CWT_calibration", "persistent_ridge_occupancy"),
            complexity="O(P*T)",
            migratability="Separable filters, reductions, and masks; CPU/CUDA compatible.",
            input_denoiser="absolute",
            params=params,
        )
        for name, params in sorted(configurations.items())
    ]


def single_map_absolute_persistence_catalog(algorithm_class) -> list[Any]:
    """Rejected absolute-power projections retained only for reproduction."""
    algorithms: list[Any] = []
    for window in (65, 129, 257, 513, 1025):
        for exponent in (0.50, 0.75, 1.00):
            exponent_code = int(round(100.0 * exponent))
            for top_k in (1, 3):
                algorithms.append(
                    algorithm_class(
                        name=f"sm_ape_l{window:04d}_r{exponent_code:03d}_k{top_k}",
                        family="rejected_single_map_absolute_persistent_energy",
                        description=(
                            "REJECTED for dense fragmented CE4 false windows. Strict single-map "
                            "absolute CWT generalized-mean projection; "
                            f"fixed {window}-record support, exponent={exponent:g}, Top-{top_k}."
                        ),
                        method="single_map_absolute_persistence",
                        source_refs=("generalized_mean", "temporal_energy_occupancy"),
                        complexity="O(P*T)",
                        migratability="One elementwise power, one fixed-width time filter, and fixed-k pooling.",
                        input_denoiser="absolute",
                        params={
                            "time_support_records": window,
                            "power_exponent": exponent,
                            "top_k": top_k,
                        },
                    )
                )
    for window in (65, 129):
        for orientation_window in (33, 65):
            for weight_floor in (0.0, 0.25):
                algorithms.append(
                    algorithm_class(
                        name=(
                            f"sm_apeh_l{window:04d}_o{orientation_window:03d}_"
                            f"f{int(round(100 * weight_floor)):02d}_k3"
                        ),
                        family="rejected_single_map_absolute_horizontal_persistent_energy",
                        description=(
                            "REJECTED for dense fragmented CE4 false windows. Absolute persistent energy "
                            "with fixed-scale horizontal structure weighting; "
                            f"support={window} records, orientation={orientation_window} records, "
                            f"weight floor={weight_floor:g}, Top-3."
                        ),
                        method="single_map_absolute_persistence",
                        source_refs=("generalized_mean", "structure_tensor_orientation"),
                        complexity="O(P*T)",
                        migratability="Separable time filter, Sobel gradients, local tensor filters, and Top-3 pooling.",
                        input_denoiser="absolute",
                        params={
                            "time_support_records": window,
                            "power_exponent": 0.5,
                            "top_k": 3,
                            "horizontal_orientation_records": orientation_window,
                            "horizontal_orientation_periods": 5,
                            "horizontal_weight_floor": weight_floor,
                        },
                    )
                )
    return algorithms


def single_map_ridge_refinement_catalog(algorithm_class) -> list[Any]:
    """One-factor refinement around the broad-screen period-CFAR leader."""
    configurations: set[tuple[int, int, float, float, int]] = set()
    for inner, outer in ((1, 9), (3, 13), (3, 17), (5, 13), (5, 17), (5, 21), (7, 17), (7, 21), (7, 25)):
        configurations.add((inner, outer, 4.0, 1.5, 3))
    for cycles in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        configurations.add((5, 17, cycles, 1.5, 3))
    for top_k in (1, 2, 3, 5, 7):
        configurations.add((5, 17, 4.0, 1.5, top_k))
    for clip in (1.0, 1.5, 2.0, 3.0):
        configurations.add((5, 17, 4.0, clip, 3))
    return [
        algorithm_class(
            name=f"sm_pcfar_i{inner:02d}_o{outer:02d}_c{int(cycles):02d}_l{_code(clip)}_k{top_k}",
            family="single_map_period_sideband_cfar_refinement",
            description=(
                f"Strict single-map period-sideband CFAR, inner={inner}, outer={outer}, "
                f"cycles={cycles:g}, clip={clip:g}, Top-{top_k}."
            ),
            method="ridge_cfar",
            source_refs=("cell_averaging_CFAR", "anisotropic_ridge_filter", "top_k_pooling"),
            complexity="O(P*T)",
            migratability="Two period-axis box filters, period-scaled time filters, fixed-k partition.",
            input_denoiser="none",
            params={
                "inner_width": inner,
                "outer_width": outer,
                "time_support_cycles": cycles,
                "clip_log_ratio": clip,
                "top_k": top_k,
            },
        )
        for inner, outer, cycles, clip, top_k in sorted(configurations)
    ]


def single_map_false_window_catalog(algorithm_class) -> list[Any]:
    """Linear single-map suppressors for multi-ridge natural-texture activity."""
    algorithms: list[Any] = []
    pooling_configs = (
        ("band_share", 2, 1.0),
        ("band_share", 2, 2.0),
        ("band_share", 4, 1.0),
        ("band_share", 4, 2.0),
        ("normalized_ipr", 2, 0.5),
        ("normalized_ipr", 2, 1.0),
        ("normalized_ipr", 2, 2.0),
        ("outside_ratio", 2, 1.0),
        ("outside_ratio", 2, 2.0),
        ("outside_ratio", 4, 1.0),
        ("outside_ratio", 4, 2.0),
        ("outside_ratio", 6, 1.0),
        ("share_outside", 2, 1.0),
        ("share_outside", 4, 1.0),
        ("share_outside", 6, 1.0),
        ("ipr_outside", 2, 1.0),
        ("ipr_outside", 4, 1.0),
    )
    for pooling, guard, power in pooling_configs:
        power_code = int(round(100.0 * power))
        algorithms.append(
            algorithm_class(
                name=f"sm_pscr_{pooling}_g{guard}_p{power_code:03d}",
                family="single_map_pscr_concentration",
                description=(
                    f"PSCR weighted by {pooling}, independent-ridge guard={guard}, "
                    f"concentration exponent={power:g}."
                ),
                method="ridge_cfar",
                source_refs=("time_frequency_concentration", "ridge_exclusivity"),
                complexity="O(P*T)",
                migratability="Elementwise reductions and fixed-width masks after separable PSCR filters.",
                input_denoiser="none",
                params={
                    "inner_width": 5,
                    "outer_width": 17,
                    "time_support_cycles": 4.0,
                    "clip_log_ratio": 1.5,
                    "top_k": 1,
                    "period_pooling": pooling,
                    "period_guard_bins": guard,
                    "concentration_power": power,
                },
            )
        )
    for clip in (1.5, 3.0):
        for support in (0.50, 0.70):
            for floor in (0.0, 0.20):
                algorithms.append(
                    algorithm_class(
                        name=(
                            f"sm_pscr_support_q{_code(support)}_f{_code(floor)}_"
                            f"l{_code(clip)}"
                        ),
                        family="single_map_pscr_support_gate",
                        description=(
                            f"PSCR with positive-support floor={support:.2f}, native floor={floor:.2f}, "
                            f"clip={clip:g}."
                        ),
                        method="ridge_cfar",
                        source_refs=("ridge_persistence", "native_evidence_gate"),
                        complexity="O(P*T)",
                        migratability="Additional separable binary-support filter and elementwise floor.",
                        input_denoiser="none",
                        params={
                            "inner_width": 5,
                            "outer_width": 17,
                            "time_support_cycles": 4.0,
                            "clip_log_ratio": clip,
                            "top_k": 1,
                            "min_positive_support": support,
                            "support_weighting": "fraction",
                            "score_floor": floor,
                        },
                    )
                )
    return algorithms


def single_map_false_window_refinement_catalog(algorithm_class) -> list[Any]:
    """Refine concentration strength and strongest-ridge temporal stability."""
    algorithms: list[Any] = []
    pooling_configs: set[tuple[str, int, float, float]] = set()
    for guard in (0, 1, 2, 3, 4):
        for power in (0.25, 0.40, 0.50, 0.60, 0.70, 0.75, 1.00, 1.25, 1.50):
            pooling_configs.add(("band_share", guard, power, 4.0))
    for pooling in ("outside_ratio", "share_outside", "ipr_outside"):
        for guard in (2, 4, 6):
            for power in (0.50, 1.00, 1.50):
                pooling_configs.add((pooling, guard, power, 4.0))
    for pooling in ("winner_persistence", "winner_band_share"):
        for guard in (1, 2, 4):
            for cycles in (2.0, 4.0, 8.0):
                for power in (1.0, 2.0):
                    pooling_configs.add((pooling, guard, power, cycles))
    for pooling, guard, power, winner_cycles in sorted(pooling_configs):
        algorithms.append(
            algorithm_class(
                name=(
                    f"sm_pscr_ref_{pooling}_g{guard}_p{int(round(100 * power)):03d}_"
                    f"w{int(winner_cycles):02d}"
                ),
                family="single_map_pscr_false_window_refinement",
                description=(
                    f"PSCR {pooling} refinement, period guard={guard}, weight exponent={power:g}, "
                    f"winner support={winner_cycles:g} cycles."
                ),
                method="ridge_cfar",
                source_refs=("Renyi_order_2_concentration", "ridge_temporal_continuity"),
                complexity="O(P*T)",
                migratability="Period reductions and separable period-scaled temporal filters.",
                input_denoiser="none",
                params={
                    "inner_width": 5,
                    "outer_width": 17,
                    "time_support_cycles": 4.0,
                    "clip_log_ratio": 1.5,
                    "top_k": 1,
                    "period_pooling": pooling,
                    "period_guard_bins": guard,
                    "concentration_power": power,
                    "winner_support_cycles": winner_cycles,
                },
            )
        )
    return algorithms


def single_map_algorithm_names() -> tuple[str, ...]:
    class _Algorithm:
        def __init__(self, **kwargs):
            self.name = kwargs["name"]

    return tuple(item.name for item in single_map_catalog(_Algorithm))


def single_map_cpro_names() -> tuple[str, ...]:
    class _Algorithm:
        def __init__(self, **kwargs):
            self.name = kwargs["name"]

    return tuple(item.name for item in single_map_cpro_catalog(_Algorithm))


def single_map_absolute_persistence_names() -> tuple[str, ...]:
    class _Algorithm:
        def __init__(self, **kwargs):
            self.name = kwargs["name"]

    return tuple(item.name for item in single_map_absolute_persistence_catalog(_Algorithm))


def single_map_sparse_algorithm_names() -> tuple[str, ...]:
    return tuple(name for name in single_map_algorithm_names() if "_a" in name)


def single_map_ridge_refinement_names() -> tuple[str, ...]:
    class _Algorithm:
        def __init__(self, **kwargs):
            self.name = kwargs["name"]

    return tuple(item.name for item in single_map_ridge_refinement_catalog(_Algorithm))


def single_map_false_window_names() -> tuple[str, ...]:
    class _Algorithm:
        def __init__(self, **kwargs):
            self.name = kwargs["name"]

    return tuple(item.name for item in single_map_false_window_catalog(_Algorithm))


def single_map_false_window_refinement_names() -> tuple[str, ...]:
    class _Algorithm:
        def __init__(self, **kwargs):
            self.name = kwargs["name"]

    return tuple(item.name for item in single_map_false_window_refinement_catalog(_Algorithm))


def _finite_power(power: np.ndarray) -> np.ndarray:
    values = np.asarray(power, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("single-map activity requires power with shape (periods, records)")
    return np.maximum(np.where(np.isfinite(values), values, 0.0), 0.0).astype(np.float32, copy=False)


def _edge_corrected_time_mean(values: np.ndarray, width: int) -> np.ndarray:
    records = int(values.shape[1])
    size = max(1, min(int(width), records))
    if size <= 1:
        return values.astype(np.float32, copy=False)
    summed = ndimage.uniform_filter1d(values, size=size, axis=1, mode="constant", cval=0.0) * float(size)
    counts = ndimage.uniform_filter1d(
        np.ones(records, dtype=np.float32), size=size, mode="constant", cval=0.0
    ) * float(size)
    return (summed / np.maximum(counts[None, :], 1.0)).astype(np.float32, copy=False)


def compute_absolute_persistent_energy(power: np.ndarray, *, width: int, exponent: float) -> np.ndarray:
    """Return a scale-equivariant persistent-energy map in native CWT-power units."""
    values = _finite_power(power)
    r = min(1.0, max(0.05, float(exponent)))
    transformed = np.power(values, r).astype(np.float32, copy=False)
    averaged = _edge_corrected_time_mean(transformed, width)
    score = np.power(np.maximum(averaged, 0.0), 1.0 / r)
    score[~np.isfinite(score)] = 0.0
    return score.astype(np.float32, copy=False)


def _horizontal_structure_weight(score: np.ndarray, *, time_width: int, period_width: int) -> np.ndarray:
    """Return zero-to-one evidence for structures elongated along the time axis."""
    amplitude = np.sqrt(np.maximum(score, 0.0)).astype(np.float32, copy=False)
    gradient_time = ndimage.sobel(amplitude, axis=1, mode="nearest")
    gradient_period = ndimage.sobel(amplitude, axis=0, mode="nearest")
    size = (max(1, int(period_width)), max(1, int(time_width)))
    jtt = ndimage.uniform_filter(gradient_time * gradient_time, size=size, mode="nearest")
    jpp = ndimage.uniform_filter(gradient_period * gradient_period, size=size, mode="nearest")
    total = jtt + jpp
    eps = max(float(np.nanmax(total)) * 1e-12, float(np.finfo(np.float32).tiny))
    weight = np.clip((jpp - jtt) / (total + eps), 0.0, 1.0)
    weight[~np.isfinite(weight)] = 0.0
    return weight.astype(np.float32, copy=False)


def compute_absolute_persistence_activity(power: np.ndarray, algorithm: Any, result_class):
    params = algorithm.params
    score = compute_absolute_persistent_energy(
        power,
        width=int(params["time_support_records"]),
        exponent=float(params["power_exponent"]),
    )
    orientation_width = int(params.get("horizontal_orientation_records", 0))
    if orientation_width > 0:
        weight = _horizontal_structure_weight(
            score,
            time_width=orientation_width,
            period_width=int(params.get("horizontal_orientation_periods", 5)),
        )
        floor = min(1.0, max(0.0, float(params.get("horizontal_weight_floor", 0.0))))
        score *= floor + (1.0 - floor) * weight
    k = max(1, min(int(params.get("top_k", 1)), score.shape[0]))
    top = np.partition(score, score.shape[0] - k, axis=0)[-k:]
    activity = np.mean(top, axis=0).astype(np.float32, copy=False)
    return result_class(activity=activity, score_map=score)


def _row_mad(values: np.ndarray) -> np.ndarray:
    center = np.median(values, axis=1, keepdims=True)
    centered = values - center
    scale = 1.4826 * np.median(np.abs(centered), axis=1, keepdims=True)
    fallback = np.quantile(np.abs(centered), 0.75, axis=1, keepdims=True) / 0.67448975
    scale = np.where(np.isfinite(scale) & (scale > MIN_SCALE), scale, fallback)
    scale = np.maximum(np.where(np.isfinite(scale), scale, MIN_SCALE), MIN_SCALE)
    calibrated = centered / scale
    calibrated[~np.isfinite(calibrated)] = 0.0
    return calibrated.astype(np.float32, copy=False)


def _period_sideband(values: np.ndarray, inner: int, outer: int) -> np.ndarray:
    inner_mean = ndimage.uniform_filter1d(values, size=inner, axis=0, mode="nearest")
    outer_mean = ndimage.uniform_filter1d(values, size=outer, axis=0, mode="nearest")
    side = (outer_mean * float(outer) - inner_mean * float(inner)) / float(outer - inner)
    contrast = inner_mean - side
    margin = outer // 2
    if margin:
        contrast[:margin] = 0.0
        contrast[-margin:] = 0.0
    return contrast.astype(np.float32, copy=False)


def _calibrated_map(power: np.ndarray, transform: str, clip_score: float) -> np.ndarray:
    values = _finite_power(power)
    if transform == "row_mad":
        score = _row_mad(values)
    elif transform == "log_row_mad":
        row_level = np.median(values, axis=1, keepdims=True)
        positive = values[values > 0.0]
        global_level = float(np.median(positive)) if positive.size else MIN_SCALE
        scaled = np.log1p(values / np.maximum(row_level, global_level * 1e-6))
        score = _row_mad(scaled)
    elif transform == "period_sideband_3_13":
        score = _period_sideband(_row_mad(values), 3, 13)
    elif transform == "period_sideband_5_21":
        score = _period_sideband(_row_mad(values), 5, 21)
    else:
        raise ValueError(f"Unknown single-map transform: {transform}")
    limit = max(0.1, float(clip_score))
    return np.clip(score, -limit, limit).astype(np.float32, copy=False)


def _coherent_map(
    signed: np.ndarray,
    periods: np.ndarray,
    *,
    cycles: float,
    min_support: float,
    score_floor: float,
) -> np.ndarray:
    period_values = np.asarray(periods, dtype=np.float64)
    if signed.shape[0] != period_values.size:
        raise ValueError("period count must match the single CWT map")
    records = int(signed.shape[1])
    coherent = np.zeros_like(signed, dtype=np.float32)
    for row, period in enumerate(period_values):
        width = max(3, int(round(float(cycles) * max(float(period), 1.0))))
        width = min(width, max(3, records if records % 2 else records - 1))
        if width % 2 == 0:
            width -= 1
        filtered = ndimage.uniform_filter1d(signed[row], size=width, mode="constant", cval=0.0)
        if min_support > 0.0:
            support = ndimage.uniform_filter1d(
                (signed[row] > 0.0).astype(np.float32), size=width, mode="constant", cval=0.0
            )
            filtered *= np.clip((support - min_support) / max(1e-6, 1.0 - min_support), 0.0, 1.0)
        margin = min(records // 2, max(int(math.ceil(max(float(period), 1.0))), width // 2))
        if margin:
            filtered[:margin] = 0.0
            filtered[-margin:] = 0.0
        coherent[row] = filtered
    coherent = np.maximum(coherent, 0.0)
    if score_floor > 0.0:
        coherent[coherent < score_floor] = 0.0
    coherent[~np.isfinite(coherent)] = 0.0
    return coherent.astype(np.float32, copy=False)


def compute_single_map_activity(power: np.ndarray, periods: np.ndarray, algorithm: Any, result_class):
    params = algorithm.params
    signed = _calibrated_map(
        power,
        str(params["transform"]),
        float(params.get("clip_score", 6.0)),
    )
    signed = signed - float(params.get("activation_threshold", 0.0))
    score = _coherent_map(
        signed,
        periods,
        cycles=float(params["time_support_cycles"]),
        min_support=float(params["min_positive_support"]),
        score_floor=float(params["score_floor"]),
    )
    k = max(1, min(int(params.get("top_k", 3)), score.shape[0]))
    top = np.partition(score, score.shape[0] - k, axis=0)[-k:]
    activity = np.mean(top, axis=0).astype(np.float32, copy=False)
    return result_class(activity=activity, score_map=score)
