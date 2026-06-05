from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any


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
    period_min_records: float = 2.0
    period_max_records: float = 512.0
    period_count: int = 96
    period_spacing: str = "log"
    block_channels: int = 128
    time_aggregation: str = "p95"
    aggregation_percentile: float = 95.0
    detector: str = "single_channel_lowfloor_pelt"
    candidate_period_min_records: float = 10.0
    candidate_period_max_records: float = 200.0
    noise_floor_fraction: float = 0.20
    excess_eps_fraction: float = 1e-6
    structure_baseline_quantile: float = 0.10
    structure_scale_quantile: float = 0.20
    structure_z_threshold: float = 1.0
    structure_time_support_records: int = 64
    structure_period_support_bins: int = 3
    structure_min_support_fraction: float = 0.10
    activity_trim_low: float = 0.05
    activity_trim_high: float = 0.95
    activity_smooth_records: int = 16
    pelt_penalty: float = 16.0
    pelt_min_size_records: int = 384
    pelt_jump_records: int = 1
    pelt_threads: int = 1
    cuda_structure_batch: bool = False
    cuda_structure_batch_channels: int = 16
    window_min_duration_records: int = 384
    window_min_activity_mean: float = 0.05
    window_min_activity_raw_mean: float = 25.0
    window_merge_gap_records: int = 256
    profile_min_prominence: float = 0.5
    profile_max_peaks_per_window: int = 1
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
        "noise_floor_fraction": "noise_floor_fraction",
        "excess_eps_fraction": "excess_eps_fraction",
        "structure_baseline_quantile": "structure_baseline_quantile",
        "structure_scale_quantile": "structure_scale_quantile",
        "structure_z_threshold": "structure_z_threshold",
        "structure_time_support_records": "structure_time_support_records",
        "structure_period_support_bins": "structure_period_support_bins",
        "structure_min_support_fraction": "structure_min_support_fraction",
        "activity_trim_low": "activity_trim_low",
        "activity_trim_high": "activity_trim_high",
        "activity_smooth_records": "activity_smooth_records",
        "pelt_penalty": "pelt_penalty",
        "pelt_min_size_records": "pelt_min_size_records",
        "pelt_jump_records": "pelt_jump_records",
        "pelt_threads": "pelt_threads",
        "cuda_structure_batch": "cuda_structure_batch",
        "cuda_structure_batch_channels": "cuda_structure_batch_channels",
        "window_min_duration_records": "window_min_duration_records",
        "window_min_activity_mean": "window_min_activity_mean",
        "window_min_activity_raw_mean": "window_min_activity_raw_mean",
        "window_merge_gap_records": "window_merge_gap_records",
        "profile_min_prominence": "profile_min_prominence",
        "profile_max_peaks_per_window": "profile_max_peaks_per_window",
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
    return CWTSearchConfig(**flat)


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
        "schema_version": 1,
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
            "noise_floor_fraction": config.noise_floor_fraction,
            "excess_eps_fraction": config.excess_eps_fraction,
            "structure_baseline_quantile": config.structure_baseline_quantile,
            "structure_scale_quantile": config.structure_scale_quantile,
            "structure_z_threshold": config.structure_z_threshold,
            "structure_time_support_records": config.structure_time_support_records,
            "structure_period_support_bins": config.structure_period_support_bins,
            "structure_min_support_fraction": config.structure_min_support_fraction,
            "activity_trim_low": config.activity_trim_low,
            "activity_trim_high": config.activity_trim_high,
            "activity_smooth_records": config.activity_smooth_records,
            "pelt_penalty": config.pelt_penalty,
            "pelt_min_size_records": config.pelt_min_size_records,
            "pelt_jump_records": config.pelt_jump_records,
            "pelt_threads": config.pelt_threads,
            "cuda_structure_batch": config.cuda_structure_batch,
            "cuda_structure_batch_channels": config.cuda_structure_batch_channels,
            "window_min_duration_records": config.window_min_duration_records,
            "window_min_activity_mean": config.window_min_activity_mean,
            "window_min_activity_raw_mean": config.window_min_activity_raw_mean,
            "window_merge_gap_records": config.window_merge_gap_records,
            "profile_min_prominence": config.profile_min_prominence,
            "profile_max_peaks_per_window": config.profile_max_peaks_per_window,
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
