from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VetoConfig:
    enabled: bool = True
    edge_time_records: int = -1
    edge_freq_mhz: float = 0.0
    max_bandwidth_fraction: float = 0.75
    max_fixed_channel_bandwidth_fraction: float = -1.0
    min_fixed_channel_duration_fraction: float = 0.25
    max_burst_duration_fraction: float = 0.02
    min_burst_bandwidth_fraction: float = 0.25


@dataclass(frozen=True)
class VetoContext:
    record_start: int
    record_stop: int
    freq_start_mhz: float
    freq_stop_mhz: float

    @property
    def duration_records(self) -> int:
        return max(1, int(self.record_stop) - int(self.record_start))

    @property
    def bandwidth_mhz(self) -> float:
        return max(1e-12, abs(float(self.freq_stop_mhz) - float(self.freq_start_mhz)))


def veto_config_from_scan_config(config: object) -> VetoConfig:
    return VetoConfig(
        enabled=bool(getattr(config, "veto_enabled")),
        edge_time_records=int(getattr(config, "veto_edge_time_records")),
        edge_freq_mhz=float(getattr(config, "veto_edge_freq_mhz")),
        max_bandwidth_fraction=float(getattr(config, "veto_max_bandwidth_fraction")),
        max_fixed_channel_bandwidth_fraction=float(
            getattr(config, "veto_max_fixed_channel_bandwidth_fraction")
        ),
        min_fixed_channel_duration_fraction=float(
            getattr(config, "veto_min_fixed_channel_duration_fraction")
        ),
        max_burst_duration_fraction=float(getattr(config, "veto_max_burst_duration_fraction")),
        min_burst_bandwidth_fraction=float(getattr(config, "veto_min_burst_bandwidth_fraction")),
    )


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value == "":
        return default
    return float(value)


def _int(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value == "":
        return default
    return int(value)


def candidate_metrics(row: Mapping[str, Any], context: VetoContext) -> dict[str, float]:
    freq0 = _float(row, "freq_start_mhz")
    freq1 = _float(row, "freq_stop_mhz")
    freq_lo, freq_hi = sorted([freq0, freq1])
    ctx_lo, ctx_hi = sorted([context.freq_start_mhz, context.freq_stop_mhz])
    duration_records = max(0, _int(row, "duration_records"))
    bandwidth_mhz = max(0.0, abs(freq1 - freq0))

    return {
        "bandwidth_fraction": bandwidth_mhz / context.bandwidth_mhz,
        "duration_fraction": duration_records / context.duration_records,
        "time_edge_distance_records": float(
            min(
                max(0, _int(row, "record_start") - context.record_start),
                max(0, context.record_stop - _int(row, "record_stop")),
            )
        ),
        "freq_edge_distance_mhz": float(
            min(max(0.0, freq_lo - ctx_lo), max(0.0, ctx_hi - freq_hi))
        ),
    }


def evaluate_vetoes(
    row: Mapping[str, Any],
    context: VetoContext,
    config: VetoConfig | None = None,
) -> dict[str, Any]:
    config = config or VetoConfig()
    if not config.enabled:
        return {
            "candidate_status": "needs_validation",
            "veto_flags": "",
            "veto_reason": "",
            "veto_rule_count": 0,
            "veto_details_json": "{}",
        }

    metrics = candidate_metrics(row, context)
    flags: list[str] = []
    details: dict[str, dict[str, float | str]] = {}

    if metrics["bandwidth_fraction"] >= config.max_bandwidth_fraction:
        flags.append("broadband")
        details["broadband"] = {
            "reason": "candidate spans too much of the scanned frequency band",
            "bandwidth_fraction": metrics["bandwidth_fraction"],
            "threshold": config.max_bandwidth_fraction,
        }

    if config.edge_time_records >= 0 and metrics["time_edge_distance_records"] <= config.edge_time_records:
        flags.append("time_edge")
        details["time_edge"] = {
            "reason": "candidate touches or lies too close to the scanned time boundary",
            "edge_distance_records": metrics["time_edge_distance_records"],
            "threshold": float(max(0, config.edge_time_records)),
        }

    if metrics["freq_edge_distance_mhz"] <= max(0.0, config.edge_freq_mhz):
        flags.append("freq_edge")
        details["freq_edge"] = {
            "reason": "candidate touches or lies too close to the scanned frequency boundary",
            "edge_distance_mhz": metrics["freq_edge_distance_mhz"],
            "threshold": float(max(0.0, config.edge_freq_mhz)),
        }

    if (
        config.max_fixed_channel_bandwidth_fraction >= 0.0
        and metrics["bandwidth_fraction"] <= config.max_fixed_channel_bandwidth_fraction
        and metrics["duration_fraction"] >= config.min_fixed_channel_duration_fraction
    ):
        flags.append("fixed_channel")
        details["fixed_channel"] = {
            "reason": "candidate is narrow in frequency but long in time",
            "bandwidth_fraction": metrics["bandwidth_fraction"],
            "bandwidth_threshold": config.max_fixed_channel_bandwidth_fraction,
            "duration_fraction": metrics["duration_fraction"],
            "duration_threshold": config.min_fixed_channel_duration_fraction,
        }

    if (
        metrics["duration_fraction"] <= config.max_burst_duration_fraction
        and metrics["bandwidth_fraction"] >= config.min_burst_bandwidth_fraction
    ):
        flags.append("burst_train")
        details["burst_train"] = {
            "reason": "candidate is short in time and broad in frequency",
            "duration_fraction": metrics["duration_fraction"],
            "duration_threshold": config.max_burst_duration_fraction,
            "bandwidth_fraction": metrics["bandwidth_fraction"],
            "bandwidth_threshold": config.min_burst_bandwidth_fraction,
        }

    return {
        "candidate_status": "vetoed" if flags else "needs_validation",
        "veto_flags": "|".join(flags),
        "veto_reason": "; ".join(details[flag]["reason"] for flag in flags),
        "veto_rule_count": len(flags),
        "veto_details_json": json.dumps(details, sort_keys=True, ensure_ascii=True),
    }


def apply_vetoes_to_candidate(
    row: Mapping[str, Any],
    context: VetoContext,
    config: VetoConfig | None = None,
) -> dict[str, Any]:
    reviewed = dict(row)
    reviewed.update(evaluate_vetoes(row, context, config))
    return reviewed


def review_candidates(
    rows: Iterable[Mapping[str, Any]],
    context: VetoContext,
    config: VetoConfig | None = None,
) -> list[dict[str, Any]]:
    return [apply_vetoes_to_candidate(row, context, config) for row in rows]
