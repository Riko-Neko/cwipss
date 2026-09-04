from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from math import ceil, isfinite, log2
from pathlib import Path
from typing import Any


AUTO_PERIOD_BINS_PER_OCTAVE = 12.0


@dataclass
class CWTSearchConfig:
    """Resolved single-file CWT period-candidate-generation config."""

    input: str | None = None
    f_start: float | None = None
    f_stop: float | None = None
    t_start: int | None = None
    t_stop: int | None = None
    wavelet: str = "cmor1.5-1.0"
    cwt_method: str = "fft"
    cwt_backend: str = "cpu"
    cuda_device: int = 0
    period_min_records: float | None = None
    period_max_records: float | None = None
    period_count: int | None = None
    period_spacing: str = "log"
    block_channels: int = 128
    time_aggregation: str = "p95"
    aggregation_percentile: float = 95.0
    detector: str = "calibrated_period_ridge_observation"
    candidate_period_min_records: float = 10.0
    candidate_period_max_records: float = 200.0
    cpro_threshold_snr: float = 32.0
    cpro_texture_quantile: float = 0.9375
    cpro_period_center_bins: int = 3
    cpro_period_context_bins: int = 15
    cpro_min_period_contrast: float = 1.5
    cpro_period_support_bins: int = 3
    cpro_shape_power_softness: float = 1.0
    cpro_shape_contrast_softness: float = 0.10
    cpro_continuity_decay: float = 0.995
    cpro_continuity_power: float = 2.0
    cpro_min_continuity_mean: float = 0.47
    cpro_min_ridge_lock: float = 0.94
    pelt_penalty: float = 16.0
    pelt_min_size_records: int = 64
    pelt_jump_records: int = 8
    pelt_threads: int = 1
    cuda_max_pending_blocks: int = 2
    window_min_activity_mean: float = 0.05
    window_merge_gap_records: int = 0
    cprf_threshold_snr: float = 32.0
    cprf_texture_quantile: float = 0.9375
    cprf_smooth_bins: int = 3
    cprf_peak_band_fraction: float = 0.50
    cprf_min_width_bins: int = 3
    cprf_min_peak_strength: float = 1.25
    cprf_min_integrated_strength: float = 0.0
    cprf_min_band_persistence: float = 0.40
    cprf_min_band_concentration: float = 0.50
    cprf_min_local_contrast: float = 1.20
    cprf_harmonic_weight: float = 0.20
    cprf_harmonic_min_relative: float = 0.12
    cprf_harmonic_window_scale: float = 1.25
    cprf_max_peak_hypotheses: int = 8
    max_candidates_per_channel: int | str = "auto"
    max_candidates_per_record: float = 3.0 / 4096.0
    output_dir: str = "runs"
    veto_enabled: bool = True
    veto_edge_time_records: int = -1
    veto_edge_freq_mhz: float = 0.0
    veto_max_bandwidth_fraction: float = 0.75
    veto_max_fixed_channel_bandwidth_fraction: float = -1.0
    veto_min_fixed_channel_duration_fraction: float = 0.25
    veto_max_burst_duration_fraction: float = 0.02
    veto_min_burst_bandwidth_fraction: float = 0.25
    validation_include_vetoed: bool = False
    validation_max_candidates: int = 25
    validation_window_periods: int = 128
    validation_min_window_records: int = 256
    validation_max_window_records: int = 4096
    validation_period_search_radius: float = 2.0
    validation_min_period_records: int = 2
    validation_max_period_records: int = 2048
    validation_fold_bins: int = 16
    validation_shuffle_trials: int = 100
    validation_random_seed: int = 12345
    visualization_enabled: bool = False
    visualization_max_blocks: int = 2
    visualization_max_channels: int = 4
    visualization_top_candidates: int = 50
    visualization_dpi: int = 140
    progress_enabled: bool = True
    progress_leave: bool = False
    timing_enabled: bool = False
    run_id: str | None = None
    save_legacy_candidates_csv: bool = False


_KNOWN_FIELDS = {field.name for field in fields(CWTSearchConfig)}


_SECTION_KEY_MAP: dict[str, dict[str, str]] = {
    "input": {
        "path": "input",
        "file": "input",
        "input": "input",
    },
    "scan": {
        "f_start": "f_start",
        "f_stop": "f_stop",
        "t_start": "t_start",
        "t_stop": "t_stop",
        "wavelet": "wavelet",
        "cwt_method": "cwt_method",
        "cwt_backend": "cwt_backend",
        "cuda_device": "cuda_device",
        "period_min_records": "period_min_records",
        "period_max_records": "period_max_records",
        "period_count": "period_count",
        "period_spacing": "period_spacing",
        "block_channels": "block_channels",
        "time_aggregation": "time_aggregation",
        "aggregation_percentile": "aggregation_percentile",
    },
    "detection": {
        "detector": "detector",
        "candidate_period_min_records": "candidate_period_min_records",
        "candidate_period_max_records": "candidate_period_max_records",
        "cpro_threshold_snr": "cpro_threshold_snr",
        "cpro_texture_quantile": "cpro_texture_quantile",
        "cpro_period_center_bins": "cpro_period_center_bins",
        "cpro_period_context_bins": "cpro_period_context_bins",
        "cpro_min_period_contrast": "cpro_min_period_contrast",
        "cpro_period_support_bins": "cpro_period_support_bins",
        "cpro_shape_power_softness": "cpro_shape_power_softness",
        "cpro_shape_contrast_softness": "cpro_shape_contrast_softness",
        "cpro_continuity_decay": "cpro_continuity_decay",
        "cpro_continuity_power": "cpro_continuity_power",
        "cpro_min_continuity_mean": "cpro_min_continuity_mean",
        "cpro_min_ridge_lock": "cpro_min_ridge_lock",
        "pelt_penalty": "pelt_penalty",
        "pelt_min_size_records": "pelt_min_size_records",
        "pelt_jump_records": "pelt_jump_records",
        "pelt_threads": "pelt_threads",
        "cuda_max_pending_blocks": "cuda_max_pending_blocks",
        "window_min_activity_mean": "window_min_activity_mean",
        "window_merge_gap_records": "window_merge_gap_records",
        "cprf_threshold_snr": "cprf_threshold_snr",
        "cprf_texture_quantile": "cprf_texture_quantile",
        "cprf_smooth_bins": "cprf_smooth_bins",
        "cprf_peak_band_fraction": "cprf_peak_band_fraction",
        "cprf_min_width_bins": "cprf_min_width_bins",
        "cprf_min_peak_strength": "cprf_min_peak_strength",
        "cprf_min_integrated_strength": "cprf_min_integrated_strength",
        "cprf_min_band_persistence": "cprf_min_band_persistence",
        "cprf_min_band_concentration": "cprf_min_band_concentration",
        "cprf_min_local_contrast": "cprf_min_local_contrast",
        "cprf_harmonic_weight": "cprf_harmonic_weight",
        "cprf_harmonic_min_relative": "cprf_harmonic_min_relative",
        "cprf_harmonic_window_scale": "cprf_harmonic_window_scale",
        "cprf_max_peak_hypotheses": "cprf_max_peak_hypotheses",
        "max_candidates_per_channel": "max_candidates_per_channel",
        "max_candidates_per_record": "max_candidates_per_record",
    },
    "veto": {
        "enabled": "veto_enabled",
        "edge_time_records": "veto_edge_time_records",
        "edge_freq_mhz": "veto_edge_freq_mhz",
        "max_bandwidth_fraction": "veto_max_bandwidth_fraction",
        "max_fixed_channel_bandwidth_fraction": "veto_max_fixed_channel_bandwidth_fraction",
        "min_fixed_channel_duration_fraction": "veto_min_fixed_channel_duration_fraction",
        "max_burst_duration_fraction": "veto_max_burst_duration_fraction",
        "min_burst_bandwidth_fraction": "veto_min_burst_bandwidth_fraction",
    },
    "validation": {
        "include_vetoed": "validation_include_vetoed",
        "max_candidates": "validation_max_candidates",
        "window_periods": "validation_window_periods",
        "min_window_records": "validation_min_window_records",
        "max_window_records": "validation_max_window_records",
        "period_search_radius": "validation_period_search_radius",
        "min_period_records": "validation_min_period_records",
        "max_period_records": "validation_max_period_records",
        "fold_bins": "validation_fold_bins",
        "shuffle_trials": "validation_shuffle_trials",
        "random_seed": "validation_random_seed",
    },
    "output": {
        "dir": "output_dir",
        "output_dir": "output_dir",
        "run_id": "run_id",
        "save_legacy_candidates_csv": "save_legacy_candidates_csv",
    },
    "progress": {
        "enabled": "progress_enabled",
        "leave": "progress_leave",
    },
    "timing": {
        "enabled": "timing_enabled",
    },
    "visualization": {
        "enabled": "visualization_enabled",
        "max_blocks": "visualization_max_blocks",
        "max_channels": "visualization_max_channels",
        "top_candidates": "visualization_top_candidates",
        "dpi": "visualization_dpi",
    },
}


def _coerce_section(section_name: str, value: object, flat: dict[str, Any]) -> None:
    if section_name == "input" and not isinstance(value, Mapping):
        flat["input"] = value
        return
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section {section_name!r} must be an object.")
    mapping = _SECTION_KEY_MAP[section_name]
    for key, item in value.items():
        if key not in mapping:
            raise ValueError(f"Unknown config key {section_name}.{key}")
        flat[mapping[key]] = item


def flatten_cwt_config_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _SECTION_KEY_MAP:
            _coerce_section(key, value, flat)
        elif key in _KNOWN_FIELDS:
            flat[key] = value
        elif key == "schema_version":
            continue
        else:
            raise ValueError(f"Unknown config key {key!r}")
    return flat


def cwt_config_from_mapping(
    payload: Mapping[str, Any] | None,
    overrides: Mapping[str, Any] | None = None,
) -> CWTSearchConfig:
    flat = flatten_cwt_config_mapping(payload or {})
    if overrides:
        flat.update({key: value for key, value in overrides.items() if value is not None})
    config = resolve_cwt_period_domain(CWTSearchConfig(**flat))
    validate_cwt_config(config)
    return config


def _automatic_period_margin_bins(config: CWTSearchConfig) -> int:
    contrast_radius = int(config.cpro_period_context_bins) // 2
    support_radius = int(config.cpro_period_support_bins) // 2
    return contrast_radius + support_radius


def resolve_cwt_period_domain(config: CWTSearchConfig) -> CWTSearchConfig:
    """Resolve the smallest safe CWT grid around the candidate period domain."""
    candidate_min = float(config.candidate_period_min_records)
    candidate_max = float(config.candidate_period_max_records)
    if (
        not isfinite(candidate_min)
        or not isfinite(candidate_max)
        or candidate_min <= 0.0
        or candidate_max <= candidate_min
    ):
        raise ValueError("candidate period bounds must be finite, positive, and increasing")

    period_min = config.period_min_records
    period_max = config.period_max_records
    period_count = config.period_count
    if (period_min is None) != (period_max is None):
        raise ValueError("period_min_records and period_max_records must both be set or both be null")

    if period_min is not None and period_max is not None:
        lo = float(period_min)
        hi = float(period_max)
        if not isfinite(lo) or not isfinite(hi) or lo <= 0.0 or hi <= lo:
            raise ValueError("explicit CWT period bounds must be finite, positive, and increasing")
        if lo > candidate_min or hi < candidate_max:
            raise ValueError("explicit CWT period bounds must contain the candidate period domain")
        if period_count is None:
            period_count = int(ceil(log2(hi / lo) * AUTO_PERIOD_BINS_PER_OCTAVE)) + 1
        return replace(
            config,
            period_min_records=lo,
            period_max_records=hi,
            period_count=int(period_count),
        )

    if config.period_spacing != "log":
        raise ValueError("automatic CWT period bounds require logarithmic spacing")
    margin = _automatic_period_margin_bins(config)
    target_octaves = log2(candidate_max / candidate_min)
    if period_count is None:
        target_intervals = max(1, int(ceil(target_octaves * AUTO_PERIOD_BINS_PER_OCTAVE)))
        period_count = target_intervals + 2 * margin + 1
    else:
        period_count = int(period_count)
        target_intervals = period_count - 1 - 2 * margin
        if target_intervals < 1:
            raise ValueError(
                f"period_count must exceed {2 * margin + 1} for the automatic safe margin"
            )
    step_octaves = target_octaves / float(target_intervals)
    padding = 2.0 ** (float(margin) * step_octaves)
    return replace(
        config,
        period_min_records=candidate_min / padding,
        period_max_records=candidate_max * padding,
        period_count=period_count,
    )


def cprf_parameters_from_config(config: CWTSearchConfig):
    """Resolve the single production CPRF parameter set."""
    from .signal.cprf import CPRFParameters

    return CPRFParameters(
        threshold_snr=config.cprf_threshold_snr,
        texture_quantile=config.cprf_texture_quantile,
        smooth_bins=config.cprf_smooth_bins,
        peak_band_fraction=config.cprf_peak_band_fraction,
        min_width_bins=config.cprf_min_width_bins,
        min_peak_strength=config.cprf_min_peak_strength,
        min_integrated_strength=config.cprf_min_integrated_strength,
        min_band_persistence=config.cprf_min_band_persistence,
        min_band_concentration=config.cprf_min_band_concentration,
        min_local_contrast=config.cprf_min_local_contrast,
        harmonic_weight=config.cprf_harmonic_weight,
        harmonic_min_relative=config.cprf_harmonic_min_relative,
        harmonic_window_scale=config.cprf_harmonic_window_scale,
        max_peak_hypotheses=config.cprf_max_peak_hypotheses,
    )


def validate_cwt_config(config: CWTSearchConfig) -> None:
    """Reject unsupported scientific configurations instead of falling back."""
    from .signal.cpro import CPRO_DETECTOR, CPROParameters

    if config.detector != CPRO_DETECTOR:
        raise ValueError(f"Unsupported detector {config.detector!r}; required detector is {CPRO_DETECTOR!r}")
    CPROParameters(
        threshold_snr=config.cpro_threshold_snr,
        texture_quantile=config.cpro_texture_quantile,
        period_center_bins=config.cpro_period_center_bins,
        period_context_bins=config.cpro_period_context_bins,
        min_period_contrast=config.cpro_min_period_contrast,
        period_support_bins=config.cpro_period_support_bins,
        shape_power_softness=config.cpro_shape_power_softness,
        shape_contrast_softness=config.cpro_shape_contrast_softness,
        continuity_decay=config.cpro_continuity_decay,
        continuity_power=config.cpro_continuity_power,
        min_continuity_mean=config.cpro_min_continuity_mean,
        min_ridge_lock=config.cpro_min_ridge_lock,
    ).validate()
    cprf_parameters_from_config(config).validate()
    if config.period_min_records is None or config.period_max_records is None or config.period_count is None:
        raise ValueError("CWT period domain must be resolved before validation")
    if config.period_count < 1:
        raise ValueError("period_count must be positive")
    if config.pelt_penalty < 0.0:
        raise ValueError("pelt_penalty must be non-negative")
    if config.pelt_min_size_records < 1 or config.pelt_jump_records < 1:
        raise ValueError("PELT record parameters must be positive")
    if config.pelt_threads < 1:
        raise ValueError("pelt_threads must be positive")
    if config.cuda_max_pending_blocks < 1:
        raise ValueError("cuda_max_pending_blocks must be positive")
    if config.window_min_activity_mean < 0.0:
        raise ValueError("PELT window activity threshold must be non-negative")
    if config.window_merge_gap_records < 0:
        raise ValueError("window_merge_gap_records must be non-negative")


def load_cwt_config(path: str | Path | None, overrides: Mapping[str, Any] | None = None) -> CWTSearchConfig:
    payload: dict[str, Any] = {}
    if path is not None:
        loaded = json.loads(Path(path).read_text())
        if not isinstance(loaded, dict):
            raise ValueError("Config JSON must contain an object.")
        payload = loaded
    return cwt_config_from_mapping(payload, overrides=overrides)


def resolve_output_dir(config: CWTSearchConfig, project_dir: str | Path) -> CWTSearchConfig:
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(project_dir) / output_dir
    return replace(config, output_dir=str(output_dir))


def cwt_config_to_nested_dict(config: CWTSearchConfig) -> dict[str, Any]:
    return {
        "schema_version": 6,
        "input": {
            "path": config.input,
        },
        "scan": {
            "f_start": config.f_start,
            "f_stop": config.f_stop,
            "t_start": config.t_start,
            "t_stop": config.t_stop,
            "wavelet": config.wavelet,
            "cwt_method": config.cwt_method,
            "cwt_backend": config.cwt_backend,
            "cuda_device": config.cuda_device,
            "period_min_records": config.period_min_records,
            "period_max_records": config.period_max_records,
            "period_count": config.period_count,
            "period_spacing": config.period_spacing,
            "block_channels": config.block_channels,
            "time_aggregation": config.time_aggregation,
            "aggregation_percentile": config.aggregation_percentile,
        },
        "detection": {
            "detector": config.detector,
            "candidate_period_min_records": config.candidate_period_min_records,
            "candidate_period_max_records": config.candidate_period_max_records,
            "cpro_threshold_snr": config.cpro_threshold_snr,
            "cpro_texture_quantile": config.cpro_texture_quantile,
            "cpro_period_center_bins": config.cpro_period_center_bins,
            "cpro_period_context_bins": config.cpro_period_context_bins,
            "cpro_min_period_contrast": config.cpro_min_period_contrast,
            "cpro_period_support_bins": config.cpro_period_support_bins,
            "cpro_shape_power_softness": config.cpro_shape_power_softness,
            "cpro_shape_contrast_softness": config.cpro_shape_contrast_softness,
            "cpro_continuity_decay": config.cpro_continuity_decay,
            "cpro_continuity_power": config.cpro_continuity_power,
            "cpro_min_continuity_mean": config.cpro_min_continuity_mean,
            "cpro_min_ridge_lock": config.cpro_min_ridge_lock,
            "pelt_penalty": config.pelt_penalty,
            "pelt_min_size_records": config.pelt_min_size_records,
            "pelt_jump_records": config.pelt_jump_records,
            "pelt_threads": config.pelt_threads,
            "cuda_max_pending_blocks": config.cuda_max_pending_blocks,
            "window_min_activity_mean": config.window_min_activity_mean,
            "window_merge_gap_records": config.window_merge_gap_records,
            "cprf_threshold_snr": config.cprf_threshold_snr,
            "cprf_texture_quantile": config.cprf_texture_quantile,
            "cprf_smooth_bins": config.cprf_smooth_bins,
            "cprf_peak_band_fraction": config.cprf_peak_band_fraction,
            "cprf_min_width_bins": config.cprf_min_width_bins,
            "cprf_min_peak_strength": config.cprf_min_peak_strength,
            "cprf_min_integrated_strength": config.cprf_min_integrated_strength,
            "cprf_min_band_persistence": config.cprf_min_band_persistence,
            "cprf_min_band_concentration": config.cprf_min_band_concentration,
            "cprf_min_local_contrast": config.cprf_min_local_contrast,
            "cprf_harmonic_weight": config.cprf_harmonic_weight,
            "cprf_harmonic_min_relative": config.cprf_harmonic_min_relative,
            "cprf_harmonic_window_scale": config.cprf_harmonic_window_scale,
            "cprf_max_peak_hypotheses": config.cprf_max_peak_hypotheses,
            "max_candidates_per_channel": config.max_candidates_per_channel,
            "max_candidates_per_record": config.max_candidates_per_record,
        },
        "veto": {
            "enabled": config.veto_enabled,
            "edge_time_records": config.veto_edge_time_records,
            "edge_freq_mhz": config.veto_edge_freq_mhz,
            "max_bandwidth_fraction": config.veto_max_bandwidth_fraction,
            "max_fixed_channel_bandwidth_fraction": config.veto_max_fixed_channel_bandwidth_fraction,
            "min_fixed_channel_duration_fraction": config.veto_min_fixed_channel_duration_fraction,
            "max_burst_duration_fraction": config.veto_max_burst_duration_fraction,
            "min_burst_bandwidth_fraction": config.veto_min_burst_bandwidth_fraction,
        },
        "validation": {
            "include_vetoed": config.validation_include_vetoed,
            "max_candidates": config.validation_max_candidates,
            "window_periods": config.validation_window_periods,
            "min_window_records": config.validation_min_window_records,
            "max_window_records": config.validation_max_window_records,
            "period_search_radius": config.validation_period_search_radius,
            "min_period_records": config.validation_min_period_records,
            "max_period_records": config.validation_max_period_records,
            "fold_bins": config.validation_fold_bins,
            "shuffle_trials": config.validation_shuffle_trials,
            "random_seed": config.validation_random_seed,
        },
        "output": {
            "output_dir": config.output_dir,
            "run_id": config.run_id,
            "save_legacy_candidates_csv": config.save_legacy_candidates_csv,
        },
        "progress": {
            "enabled": config.progress_enabled,
            "leave": config.progress_leave,
        },
        "timing": {
            "enabled": config.timing_enabled,
        },
        "visualization": {
            "enabled": config.visualization_enabled,
            "max_blocks": config.visualization_max_blocks,
            "max_channels": config.visualization_max_channels,
            "top_candidates": config.visualization_top_candidates,
            "dpi": config.visualization_dpi,
        },
    }


def cwt_config_to_flat_dict(config: CWTSearchConfig) -> dict[str, Any]:
    return asdict(config)
