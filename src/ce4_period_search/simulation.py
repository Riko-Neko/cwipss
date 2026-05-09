from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class InjectionSpec:
    injection_id: str
    signal_model: str = "single_channel_periodic"
    period_records: float = 16.0
    amplitude: float = 5.0
    record_start: int = 0
    duration_records: int | None = None
    channel_center: float = 16.0
    bandwidth_channels: float = 1.0
    duty_cycle: float = 0.15
    phase: float = 0.0
    drift_channels: float = 0.0


def make_noise_background(
    records: int,
    channels: int,
    noise_std: float = 1.0,
    seed: int = 12345,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, float(noise_std), size=(int(records), int(channels))).astype(np.float32)


def default_frequency_axis(channels: int, f_start_mhz: float = 0.0, f_stop_mhz: float | None = None) -> np.ndarray:
    stop = float(channels - 1) if f_stop_mhz is None else float(f_stop_mhz)
    return np.linspace(float(f_start_mhz), stop, int(channels), dtype=np.float64)


def _channel_envelope(channels: int, center: float, bandwidth: float) -> np.ndarray:
    if channels <= 0:
        return np.zeros(0, dtype=np.float64)
    if float(bandwidth) <= 1.0:
        envelope = np.zeros(channels, dtype=np.float64)
        idx = min(max(int(round(float(center))), 0), channels - 1)
        envelope[idx] = 1.0
        return envelope
    x = np.arange(channels, dtype=np.float64)
    width = max(float(bandwidth), 1.0)
    sigma = max(width / 2.355, 0.5)
    envelope = np.exp(-0.5 * ((x - float(center)) / sigma) ** 2)
    peak = np.nanmax(envelope)
    return envelope / peak if peak > 0 else envelope


def _time_indices(records: int, spec: InjectionSpec) -> tuple[np.ndarray, int, int]:
    start = max(0, int(spec.record_start))
    duration = records - start if spec.duration_records is None else int(spec.duration_records)
    stop = max(start, min(records, start + max(0, duration)))
    return np.arange(start, stop, dtype=np.int64), start, stop


def _periodic_wave(times: np.ndarray, spec: InjectionSpec) -> np.ndarray:
    period = max(float(spec.period_records), 1.0)
    phase = ((times.astype(np.float64) - float(spec.record_start)) / period + float(spec.phase)) % 1.0
    model = spec.signal_model
    if model in {"sinusoidal_narrowband", "band_limited_periodic", "drifting_ridge"}:
        return np.sin(2.0 * np.pi * phase)
    if model in {"single_channel_periodic", "pulsed_periodic", "intermittent_periodic"}:
        duty = min(max(float(spec.duty_cycle), 1e-3), 0.95)
        distance = np.minimum(phase, 1.0 - phase)
        sigma = max(duty / 2.355, 1e-3)
        return np.exp(-0.5 * (distance / sigma) ** 2)
    raise ValueError(f"Unknown injection signal_model: {model}")


def inject_periodic_signal(data: np.ndarray, spec: InjectionSpec) -> tuple[np.ndarray, dict]:
    matrix = np.asarray(data, dtype=np.float32).copy()
    if matrix.ndim != 2:
        raise ValueError("data must have shape (records, channels)")
    records, channels = matrix.shape
    times, start, stop = _time_indices(records, spec)
    if times.size == 0:
        return matrix, injection_truth(spec, channels, start, stop)

    wave = _periodic_wave(times, spec)
    bandwidth = 1.0 if spec.signal_model == "single_channel_periodic" else spec.bandwidth_channels
    if spec.signal_model == "drifting_ridge" and spec.drift_channels != 0:
        span = max(times.size - 1, 1)
        for row_idx, time_value in enumerate(times):
            frac = row_idx / span
            center = float(spec.channel_center) + (frac - 0.5) * float(spec.drift_channels)
            matrix[time_value, :] += (float(spec.amplitude) * wave[row_idx] * _channel_envelope(
                channels, center, bandwidth
            )).astype(np.float32)
    else:
        envelope = _channel_envelope(channels, spec.channel_center, bandwidth)
        matrix[times, :] += (float(spec.amplitude) * wave[:, None] * envelope[None, :]).astype(np.float32)
    return matrix, injection_truth(spec, channels, start, stop)


def injection_truth(spec: InjectionSpec, channels: int, start: int | None = None, stop: int | None = None) -> dict:
    start = int(spec.record_start) if start is None else int(start)
    if stop is None:
        duration = 0 if spec.duration_records is None else int(spec.duration_records)
        stop = start + max(0, duration)
    center = float(spec.channel_center)
    bandwidth = 1.0 if spec.signal_model == "single_channel_periodic" else max(float(spec.bandwidth_channels), 1.0)
    half_width = bandwidth / 2.0
    if channels <= 0:
        channel_start = 0
        channel_stop = 0
    elif spec.signal_model == "single_channel_periodic":
        channel_start = min(max(int(round(center)), 0), int(channels) - 1)
        channel_stop = channel_start + 1
    else:
        channel_start = min(max(0, int(np.floor(center - half_width))), int(channels) - 1)
        channel_stop = min(int(channels), max(channel_start + 1, int(np.ceil(center + half_width)) + 1))
    payload = asdict(spec)
    payload.update(
        {
            "record_start": start,
            "record_stop": int(stop),
            "duration_records": int(max(0, stop - start)),
            "channel_start": channel_start,
            "channel_stop": channel_stop,
            "bandwidth_channels": bandwidth,
        }
    )
    return payload
