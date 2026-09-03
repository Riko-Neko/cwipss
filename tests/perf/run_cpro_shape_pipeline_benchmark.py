#!/usr/bin/env python3
"""Benchmark the production shape-only CPRO -> PELT -> CPRF boundary."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np


PERF_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PERF_DIR.parents[1]
SRC_DIR = PROJECT_DIR / "src"
for path in (PERF_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compression_config_rank import injection_group_id  # noqa: E402
from cwt_activity_rank import (  # noqa: E402
    CWTActivityRun,
    activity_config_from_cwt,
    prepare_activity_component,
)
from cwipss.analysis.injection_config import (  # noqa: E402
    load_injection_config,
    make_injections_from_config,
)
from cwipss.config import (  # noqa: E402
    CWTSearchConfig,
    cprf_parameters_from_config,
    load_cwt_config,
)
from cwipss.data.readers import open_spectrum_reader  # noqa: E402
from cwipss.signal.cpro import (  # noqa: E402
    CPROParameters,
    cpro_activity,
    cpro_continuity_map,
    cpro_period_mask,
    difference_noise_std,
    impulse_cwt_noise_gain,
)
from cwipss.signal.cprf import evaluate_cprf, normalize_cwt_power  # noqa: E402
from cwipss.signal.cwt import cwt_power_cube, period_grid_records  # noqa: E402
from cwipss.signal.detection import pelt_windows_from_activity  # noqa: E402


def _cpro_parameters(config: CWTSearchConfig) -> CPROParameters:
    return CPROParameters(
        threshold_snr=config.cpro_threshold_snr,
        texture_quantile=config.cpro_texture_quantile,
        period_center_bins=config.cpro_period_center_bins,
        period_context_bins=config.cpro_period_context_bins,
        min_period_contrast=config.cpro_min_period_contrast,
        period_support_bins=config.cpro_period_support_bins,
        shape_power_softness=config.cpro_shape_power_softness,
        shape_contrast_softness=config.cpro_shape_contrast_softness,
    )


def _pelt_windows(cpro, config: CWTSearchConfig) -> list[dict]:
    continuity = cpro_continuity_map(
        cpro.shape_map,
        decay=config.cpro_continuity_decay,
        power=config.cpro_continuity_power,
    )
    return pelt_windows_from_activity(
        cpro.shape_activity,
        continuity,
        calibrated_threshold=cpro.threshold,
        penalty=config.pelt_penalty,
        min_size=config.pelt_min_size_records,
        jump=config.pelt_jump_records,
        min_mean=config.window_min_activity_mean,
        min_continuity_mean=config.cpro_min_continuity_mean,
        min_ridge_lock=config.cpro_min_ridge_lock,
        merge_gap=config.window_merge_gap_records,
    )[0]


def _overlaps(window: dict, start: int, stop: int) -> bool:
    return int(window["record_start"]) < stop and int(window["record_stop"]) > start


FEATURE_FIELDS = (
    "kind",
    "case_id",
    "group_id",
    "truth_overlap",
    "period_match_50pct",
    "accepted_default",
    "width_bins",
    "peak_strength",
    "integrated_strength",
    "band_persistence",
    "band_concentration",
    "local_contrast",
    "peak_period_records",
    "truth_period_records",
)


def _evaluated_windows(power, periods, noise_std, noise_gain, windows, cprf_params):
    normalized, threshold = normalize_cwt_power(
        power,
        noise_std=noise_std,
        noise_gain=noise_gain,
        params=cprf_params,
    )
    evaluated = []
    for window in windows:
        start = int(window["record_start"])
        stop = int(window["record_stop"])
        result = evaluate_cprf(
            normalized[:, start:stop],
            periods,
            normalization_threshold=threshold,
            params=cprf_params,
        )
        evaluated.append((window, result))
    return evaluated


def _feature_row(*, kind, case_id, group_id, overlap, period_match, result, truth_period):
    return {
        "kind": kind,
        "case_id": case_id,
        "group_id": group_id,
        "truth_overlap": int(overlap),
        "period_match_50pct": int(period_match),
        "accepted_default": int(result.accepted),
        "width_bins": int(result.width_bins),
        "peak_strength": float(result.peak_strength),
        "integrated_strength": float(result.integrated_strength),
        "band_persistence": float(result.band_persistence),
        "band_concentration": float(result.band_concentration),
        "local_contrast": float(result.local_contrast),
        "peak_period_records": float(result.peak_period_records),
        "truth_period_records": truth_period,
    }


def _injection_metrics(reader, specs, periods, noise_gain, config, cpro_params, cprf_params, run):
    group_hits: dict[str, dict[str, bool]] = defaultdict(
        lambda: {"pelt": False, "cprf": False, "period": False}
    )
    totals = {"components": 0, "pelt_windows": 0, "cprf_windows": 0}
    timing = {"prepare_seconds": 0.0, "cpro_seconds": 0.0, "pelt_seconds": 0.0, "cprf_seconds": 0.0}
    feature_rows = []
    benchmark_config = activity_config_from_cwt(run)
    for spec in specs:
        started = perf_counter()
        prepared, _baseline, _injected = prepare_activity_component(
            reader=reader,
            spec=spec,
            config=benchmark_config,
            periods=periods,
            input_denoisers=("absolute",),
            noise_gain=noise_gain,
        )
        timing["prepare_seconds"] += perf_counter() - started
        power = prepared.cwt_power["absolute"]
        started = perf_counter()
        cpro = cpro_activity(
            power,
            noise_std=prepared.noise_std,
            noise_gain=noise_gain,
            params=cpro_params,
        )
        timing["cpro_seconds"] += perf_counter() - started
        started = perf_counter()
        windows = _pelt_windows(cpro, config)
        timing["pelt_seconds"] += perf_counter() - started
        started = perf_counter()
        evaluated = _evaluated_windows(
            power,
            periods,
            prepared.noise_std,
            noise_gain,
            windows,
            cprf_params,
        )
        timing["cprf_seconds"] += perf_counter() - started
        truth_start = int(prepared.truth["record_start"])
        truth_stop = int(prepared.truth["record_stop"])
        truth_period = float(prepared.truth["period_records"])
        group_id = injection_group_id(spec.injection_id)
        group = group_hits[group_id]
        for window, result in evaluated:
            overlap = _overlaps(window, truth_start, truth_stop)
            period_match = (
                overlap
                and abs(float(result.peak_period_records) - truth_period) / truth_period <= 0.5
            )
            feature_rows.append(
                _feature_row(
                    kind="injection",
                    case_id=spec.injection_id,
                    group_id=group_id,
                    overlap=overlap,
                    period_match=period_match,
                    result=result,
                    truth_period=truth_period,
                )
            )
        accepted = [(window, result) for window, result in evaluated if result.accepted]
        group["pelt"] |= any(_overlaps(window, truth_start, truth_stop) for window in windows)
        group["cprf"] |= any(_overlaps(window, truth_start, truth_stop) for window, _result in accepted)
        group["period"] |= any(
            _overlaps(window, truth_start, truth_stop)
            and abs(float(result.peak_period_records) - truth_period) / truth_period <= 0.5
            for window, result in accepted
        )
        totals["components"] += 1
        totals["pelt_windows"] += len(windows)
        totals["cprf_windows"] += len(accepted)
    groups = len(group_hits)
    return {
        **totals,
        "groups": groups,
        "pelt_group_hits": sum(row["pelt"] for row in group_hits.values()),
        "cprf_group_hits": sum(row["cprf"] for row in group_hits.values()),
        "period_group_hits": sum(row["period"] for row in group_hits.values()),
        "pelt_group_recall": sum(row["pelt"] for row in group_hits.values()) / max(groups, 1),
        "cprf_group_recall": sum(row["cprf"] for row in group_hits.values()) / max(groups, 1),
        "period_group_recall": sum(row["period"] for row in group_hits.values()) / max(groups, 1),
        "timing": timing,
    }, feature_rows


def _negative_metrics(reader, periods, noise_gain, config, cpro_params, cprf_params, args):
    frequencies = np.asarray(reader.freqs_mhz)
    selected = np.flatnonzero(
        (frequencies >= args.negative_f_start) & (frequencies <= args.negative_f_stop)
    )
    totals = {
        "channels": 0,
        "channels_with_pelt_windows": 0,
        "channels_with_cprf_windows": 0,
        "pelt_windows": 0,
        "cprf_windows": 0,
    }
    timing = {"cwt_seconds": 0.0, "cpro_seconds": 0.0, "pelt_seconds": 0.0, "cprf_seconds": 0.0}
    feature_rows = []
    width = max(1, int(args.negative_block_channels))
    for offset in range(0, selected.size, width):
        channels = selected[offset : offset + width]
        raw = np.asarray(
            reader.read_block(
                slice(0, reader.n_records),
                slice(int(channels[0]), int(channels[-1]) + 1),
            ).data,
            dtype=np.float32,
        )
        started = perf_counter()
        power_cube = cwt_power_cube(
            raw,
            periods,
            wavelet=config.wavelet,
            normalize_channels=False,
            method=config.cwt_method,
            backend="cpu",
        )
        timing["cwt_seconds"] += perf_counter() - started
        for local in range(channels.size):
            try:
                noise_std = difference_noise_std(raw[:, local])
            except ValueError:
                continue
            power = np.asarray(power_cube[:, :, local], dtype=np.float32)
            started = perf_counter()
            cpro = cpro_activity(
                power,
                noise_std=noise_std,
                noise_gain=noise_gain,
                params=cpro_params,
            )
            timing["cpro_seconds"] += perf_counter() - started
            started = perf_counter()
            windows = _pelt_windows(cpro, config)
            timing["pelt_seconds"] += perf_counter() - started
            started = perf_counter()
            evaluated = _evaluated_windows(
                power,
                periods,
                noise_std,
                noise_gain,
                windows,
                cprf_params,
            )
            timing["cprf_seconds"] += perf_counter() - started
            accepted = [(window, result) for window, result in evaluated if result.accepted]
            channel_id = f"negative_ch{int(channels[local]):04d}"
            feature_rows.extend(
                _feature_row(
                    kind="negative",
                    case_id=channel_id,
                    group_id=channel_id,
                    overlap=False,
                    period_match=False,
                    result=result,
                    truth_period="",
                )
                for _window, result in evaluated
            )
            totals["channels"] += 1
            totals["channels_with_pelt_windows"] += int(bool(windows))
            totals["channels_with_cprf_windows"] += int(bool(accepted))
            totals["pelt_windows"] += len(windows)
            totals["cprf_windows"] += len(accepted)
        del power_cube
    totals["cprf_window_retention"] = totals["cprf_windows"] / max(totals["pelt_windows"], 1)
    return {**totals, "timing": timing}, feature_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/cwt_default.json")
    parser.add_argument(
        "--injections",
        type=Path,
        default=PROJECT_DIR / "configs/injection_lowfreq_random_100.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-f-start", type=float, default=0.15)
    parser.add_argument("--negative-f-stop", type=float, default=1.90)
    parser.add_argument("--negative-block-channels", type=int, default=8)
    parser.add_argument("--pelt-jump", type=int, default=None)
    args = parser.parse_args()

    overrides = {"cwt_backend": "cpu"}
    if args.pelt_jump is not None:
        overrides["pelt_jump_records"] = args.pelt_jump
    config = load_cwt_config(args.config, overrides=overrides)
    reader = open_spectrum_reader(args.input)
    full_periods = period_grid_records(
        config.period_min_records,
        config.period_max_records,
        config.period_count,
        config.period_spacing,
    )
    period_mask = cpro_period_mask(
        full_periods,
        config.candidate_period_min_records,
        config.candidate_period_max_records,
    )
    periods = full_periods[period_mask]
    noise_gain = impulse_cwt_noise_gain(periods, wavelet=config.wavelet, method=config.cwt_method)
    specs = make_injections_from_config(
        load_injection_config(args.injections),
        records=reader.n_records,
        channels=reader.n_channels,
        freqs_mhz=reader.freqs_mhz,
    )
    run = CWTActivityRun(
        output_dir=args.output,
        input_path=args.input,
        injection_config=args.injections,
        cwt_config=args.config,
        cwt_backend="cpu",
        pelt_threads=config.pelt_threads,
        candidate_period_max_records=config.candidate_period_max_records,
        progress_every=0,
    )
    cpro_params = _cpro_parameters(config)
    cprf_params = cprf_parameters_from_config(config)
    injection_metrics, injection_features = _injection_metrics(
        reader,
        specs,
        periods,
        noise_gain,
        config,
        cpro_params,
        cprf_params,
        run,
    )
    negative_metrics, negative_features = _negative_metrics(
        reader,
        periods,
        noise_gain,
        config,
        cpro_params,
        cprf_params,
        args,
    )
    payload = {
        "method": "shape-only CPRO -> native PELT -> CPRF on original CWT2D",
        "input": str(args.input),
        "config": str(args.config),
        "injections": str(args.injections),
        "injection": injection_metrics,
        "negative": negative_metrics,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "window_features.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_FIELDS)
        writer.writeheader()
        writer.writerows(injection_features + negative_features)
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
