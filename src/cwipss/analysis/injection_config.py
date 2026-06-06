"""Configuration-driven injection sampling."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .simulation import InjectionSpec


def load_injection_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Injection config JSON must contain an object.")
    return payload


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "set"


def _rng(payload: Mapping[str, Any], default_seed: int) -> np.random.Generator:
    return np.random.default_rng(int(payload.get("seed", default_seed)))


def _sample(spec: Any, rng: np.random.Generator, *, integer: bool = False) -> float | str:
    if isinstance(spec, list):
        if not spec:
            raise ValueError("Sampler list cannot be empty.")
        return spec[int(rng.integers(0, len(spec)))]
    if not isinstance(spec, Mapping):
        return int(spec) if integer else spec
    if "value" in spec:
        value = spec["value"]
        return int(value) if integer else value
    if "values" in spec:
        values = list(spec["values"])
        if not values:
            raise ValueError("Sampler values cannot be empty.")
        value = values[int(rng.integers(0, len(values)))]
        return int(value) if integer else value
    if "min" not in spec or "max" not in spec:
        raise ValueError(f"Sampler requires value, values, or min/max: {spec}")
    lo = float(spec["min"])
    hi = float(spec["max"])
    if hi < lo:
        lo, hi = hi, lo
    distribution = str(spec.get("distribution", "uniform"))
    if integer or distribution == "integer_uniform":
        return int(rng.integers(int(math.floor(lo)), int(math.floor(hi)) + 1))
    if distribution == "log_uniform":
        if lo <= 0 or hi <= 0:
            raise ValueError("log_uniform sampler requires positive min and max.")
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    if distribution in {"uniform", "integer_uniform"}:
        return float(rng.uniform(lo, hi))
    raise ValueError(f"Unknown sampler distribution: {distribution}")


def _sample_float(spec: Any, rng: np.random.Generator) -> float:
    return float(_sample(spec, rng))


def _sample_int(spec: Any, rng: np.random.Generator) -> int:
    return int(_sample(spec, rng, integer=True))


def _sample_string(spec: Any, rng: np.random.Generator) -> str:
    return str(_sample(spec, rng))


def _time_window(time_config: Mapping[str, Any], records: int, rng: np.random.Generator) -> tuple[int, int]:
    if "duration_records" in time_config:
        duration = _sample_int(time_config["duration_records"], rng)
    else:
        duration_fraction = _sample_float(time_config.get("duration_fraction", {"min": 0.5, "max": 1.0}), rng)
        duration = int(math.ceil(max(0.0, duration_fraction) * records))
    duration = min(max(1, duration), records)
    max_start = max(0, records - duration)
    if "record_start" in time_config:
        start = min(max(0, _sample_int(time_config["record_start"], rng)), max_start)
    else:
        start = int(rng.integers(0, max_start + 1)) if max_start else 0
    return start, duration


def _channel_from_frequency(freq_mhz: float, freqs_mhz: np.ndarray) -> int:
    if freqs_mhz.size == 0:
        raise ValueError("frequency_mhz sampler requires a non-empty frequency axis.")
    return int(np.nanargmin(np.abs(freqs_mhz - float(freq_mhz))))


def _sample_channel(set_config: Mapping[str, Any], freqs_mhz: np.ndarray, channels: int, rng: np.random.Generator) -> int:
    if "frequency_mhz" in set_config:
        return _channel_from_frequency(_sample_float(set_config["frequency_mhz"], rng), freqs_mhz)
    if "channel_center" in set_config:
        return min(max(0, _sample_int(set_config["channel_center"], rng)), channels - 1)
    return int(rng.integers(0, channels))


def _sample_copy_channel(
    set_config: Mapping[str, Any],
    replication: Mapping[str, Any],
    freqs_mhz: np.ndarray,
    channels: int,
    rng: np.random.Generator,
    used_channels: set[int],
) -> int:
    source = replication if "frequency_mhz" in replication or "channel_center" in replication else set_config
    for _ in range(max(8, channels * 2)):
        channel = _sample_channel(source, freqs_mhz, channels, rng)
        if channel not in used_channels:
            return channel
    return _sample_channel(source, freqs_mhz, channels, rng)


def _copy_count(replication: Mapping[str, Any], rng: np.random.Generator) -> int:
    max_copies = max(1, int(replication.get("max_copies", 1)))
    probability = min(max(float(replication.get("probability", 0.0)), 0.0), 1.0)
    count = 1
    for _ in range(max_copies - 1):
        if rng.random() < probability:
            count += 1
    return count


def make_injections_from_config(
    payload: Mapping[str, Any],
    *,
    records: int,
    channels: int,
    freqs_mhz: np.ndarray | None = None,
    default_seed: int = 12345,
) -> list[InjectionSpec]:
    """Create injection specs from a declarative simulation config."""
    records = max(1, int(records))
    channels = max(1, int(channels))
    freqs = np.arange(channels, dtype=np.float64) if freqs_mhz is None else np.asarray(freqs_mhz, dtype=np.float64)
    if freqs.size != channels:
        raise ValueError("freqs_mhz must have one value per channel.")
    rng = _rng(payload, default_seed)
    sets = payload.get("sets")
    if not isinstance(sets, list) or not sets:
        raise ValueError("Injection config requires a non-empty sets list.")

    specs: list[InjectionSpec] = []
    for set_index, set_config in enumerate(sets, start=1):
        if not isinstance(set_config, Mapping):
            raise ValueError("Each injection set must be an object.")
        name = _slug(set_config.get("name", f"set_{set_index}"))
        count = max(1, int(set_config.get("count", 1)))
        replication = set_config.get("replication", {})
        if not isinstance(replication, Mapping):
            raise ValueError("replication must be an object when present.")
        time_config = set_config.get("time", {})
        if not isinstance(time_config, Mapping):
            raise ValueError("time must be an object when present.")
        modulation_config = set_config.get("modulation", {})
        if not isinstance(modulation_config, Mapping):
            raise ValueError("modulation must be an object when present.")

        for base_idx in range(1, count + 1):
            model = _sample_string(set_config.get("signal_model", "single_channel_periodic"), rng)
            period = _sample_float(set_config.get("period_records", 16.0), rng)
            amplitude = _sample_float(set_config.get("amplitude", 5.0), rng)
            start, duration = _time_window(time_config, records, rng)
            phase = _sample_float(
                modulation_config.get("phase", set_config.get("phase", {"min": 0.0, "max": 1.0})),
                rng,
            )
            duty = _sample_float(
                modulation_config.get("duty_cycle", set_config.get("duty_cycle", 0.15)),
                rng,
            )
            bandwidth = _sample_float(set_config.get("bandwidth_channels", 1.0), rng)
            drift = _sample_float(set_config.get("drift_channels", 0.0), rng)
            used_channels: set[int] = set()
            copies = _copy_count(replication, rng)
            for copy_idx in range(1, copies + 1):
                if copy_idx == 1:
                    channel = _sample_channel(set_config, freqs, channels, rng)
                else:
                    channel = _sample_copy_channel(set_config, replication, freqs, channels, rng, used_channels)
                used_channels.add(channel)
                specs.append(
                    InjectionSpec(
                        injection_id=f"inj_{len(specs) + 1:04d}_{name}_b{base_idx:03d}_c{copy_idx:02d}",
                        signal_model=model,
                        period_records=period,
                        amplitude=amplitude,
                        record_start=start,
                        duration_records=duration,
                        channel_center=float(channel),
                        bandwidth_channels=bandwidth,
                        duty_cycle=duty,
                        phase=phase,
                        drift_channels=drift,
                    )
                )
    return specs
