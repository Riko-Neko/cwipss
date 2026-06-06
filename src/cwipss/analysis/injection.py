"""Synthetic and instrument-background signal injection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.readers import CE4Reader
from .simulation import InjectionSpec, default_frequency_axis, inject_periodic_signal, make_noise_background


@dataclass(frozen=True)
class BackgroundData:
    data: np.ndarray
    freqs_mhz: np.ndarray
    source_name: str
    tsamp_seconds: float = 1.0


def synthetic_background(
    records: int,
    channels: int,
    noise_std: float = 1.0,
    seed: int = 12345,
    f_start_mhz: float = 0.0,
    f_stop_mhz: float | None = None,
) -> BackgroundData:
    return BackgroundData(
        data=make_noise_background(records, channels, noise_std=noise_std, seed=seed),
        freqs_mhz=default_frequency_axis(channels, f_start_mhz=f_start_mhz, f_stop_mhz=f_stop_mhz),
        source_name="synthetic",
        tsamp_seconds=1.0,
    )


def ce4_background(
    input_path: str | Path,
    f_start: float | None = None,
    f_stop: float | None = None,
    t_start: int | None = None,
    t_stop: int | None = None,
) -> BackgroundData:
    reader = CE4Reader(input_path)
    block = reader.read_block(reader.record_slice(t_start, t_stop), reader.freq_slice(f_start, f_stop))
    return BackgroundData(
        data=block.data,
        freqs_mhz=block.freqs_mhz,
        source_name=str(input_path),
        tsamp_seconds=reader.tsamp_seconds,
    )


def inject_many(background: BackgroundData, specs: list[InjectionSpec]) -> tuple[BackgroundData, list[dict]]:
    data = background.data
    truths: list[dict] = []
    for spec in specs:
        data, truth = inject_periodic_signal(data, spec)
        truth = dict(truth)
        if background.freqs_mhz.size:
            start = int(truth["channel_start"])
            stop = int(truth["channel_stop"])
            stop_idx = min(max(stop - 1, start), background.freqs_mhz.size - 1)
            center_idx = min(max(int(round(float(truth["channel_center"]))), 0), background.freqs_mhz.size - 1)
            truth["freq_start_mhz"] = float(background.freqs_mhz[start])
            truth["freq_stop_mhz"] = float(background.freqs_mhz[stop_idx])
            truth["freq_center_mhz"] = float(background.freqs_mhz[center_idx])
        truths.append(truth)
    return (
        BackgroundData(
            data=data,
            freqs_mhz=background.freqs_mhz,
            source_name=background.source_name,
            tsamp_seconds=background.tsamp_seconds,
        ),
        truths,
    )
