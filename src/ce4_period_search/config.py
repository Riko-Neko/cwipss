from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any


@dataclass
class SWTScanConfig:
    """Resolved single-file SWT candidate-generation config.

    The public dataclass stays flat for compatibility with the original
    prototype. JSON config files may use either this flat layout or the
    structured sections handled by `swt_config_from_mapping`.
    """

    input: str | None = None
    f_start: float | None = None
    f_stop: float | None = None
    t_start: int | None = None
    t_stop: int | None = None
    wavelet: str = "db4"
    levels: int = 5
    block_channels: int = 128
    threshold: float = 5.0
    min_pixels: int = 12
    local_time: int = 513
    local_freq: int = 9
    output_dir: str = "runs"
    max_candidates_per_block: int = 200
    veto_enabled: bool = True
    veto_edge_time_records: int = 0
    veto_edge_freq_mhz: float = 0.0
    veto_max_bandwidth_fraction: float = 0.75
    veto_max_fixed_channel_bandwidth_fraction: float = 0.01
    veto_min_fixed_channel_duration_fraction: float = 0.25
    veto_max_burst_duration_fraction: float = 0.02
    veto_min_burst_bandwidth_fraction: float = 0.25
    validation_include_vetoed: bool = False
    validation_max_candidates: int = 50
    validation_window_periods: int = 128
    validation_min_window_records: int = 256
    validation_max_window_records: int = 4096
    validation_period_search_radius: float = 2.0
    validation_min_period_records: int = 2
    validation_max_period_records: int = 2048
    validation_fold_bins: int = 16
    validation_shuffle_trials: int = 100
    validation_random_seed: int = 12345
    run_id: str | None = None
    save_legacy_candidates_csv: bool = True


_KNOWN_FIELDS = {field.name for field in fields(SWTScanConfig)}


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
        "levels": "levels",
        "block_channels": "block_channels",
    },
    "detection": {
        "threshold": "threshold",
        "min_pixels": "min_pixels",
        "local_time": "local_time",
        "local_freq": "local_freq",
        "max_candidates_per_block": "max_candidates_per_block",
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


def flatten_swt_config_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return flat SWTScanConfig kwargs from flat or structured config input."""
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


def swt_config_from_mapping(
    payload: Mapping[str, Any] | None,
    overrides: Mapping[str, Any] | None = None,
) -> SWTScanConfig:
    flat = flatten_swt_config_mapping(payload or {})
    if overrides:
        flat.update({key: value for key, value in overrides.items() if value is not None})
    return SWTScanConfig(**flat)


def load_swt_config(path: str | Path | None, overrides: Mapping[str, Any] | None = None) -> SWTScanConfig:
    payload: dict[str, Any] = {}
    if path is not None:
        loaded = json.loads(Path(path).read_text())
        if not isinstance(loaded, dict):
            raise ValueError("Config JSON must contain an object.")
        payload = loaded
    return swt_config_from_mapping(payload, overrides=overrides)


def resolve_output_dir(config: SWTScanConfig, project_dir: str | Path) -> SWTScanConfig:
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(project_dir) / output_dir
    return replace(config, output_dir=str(output_dir))


def swt_config_to_nested_dict(config: SWTScanConfig) -> dict[str, Any]:
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
            "levels": config.levels,
            "block_channels": config.block_channels,
        },
        "detection": {
            "threshold": config.threshold,
            "min_pixels": config.min_pixels,
            "local_time": config.local_time,
            "local_freq": config.local_freq,
            "max_candidates_per_block": config.max_candidates_per_block,
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
    }


def swt_config_to_flat_dict(config: SWTScanConfig) -> dict[str, Any]:
    return asdict(config)
