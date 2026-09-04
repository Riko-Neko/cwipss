"""Single-file CWT search workflow."""

from __future__ import annotations

import csv
import json
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..config import (
    CWTSearchConfig,
    cprf_parameters_from_config,
    cwt_config_to_nested_dict,
    resolve_cwt_period_domain,
    validate_cwt_config,
)
from ..signal.cpro import impulse_cwt_noise_gain
from ..signal.cwt import cwt_power_cube, period_grid_records
from ..signal.detection import add_candidate_ids, detect_block_periods
from ..signal.windows import PeltCancellation, require_native_pelt
from ..data.readers import SpectrumReader, open_spectrum_reader
from ..data.schemas import (
    MANIFEST_FIELDNAMES,
    RAW_CANDIDATE_SCHEMA_VERSION,
    RAW_CANDIDATE_FIELDNAMES,
    REVIEWED_CANDIDATE_FIELDNAMES,
    TIME_WINDOW_FIELDNAMES,
    make_manifest_row,
    normalize_candidate_row,
)
from ..runtime import runtime_info
from ..analysis.veto import VetoContext, review_candidates, veto_config_from_scan_config


@dataclass
class _PendingCudaBlock:
    block_id: str
    channel_range: tuple[int, int]
    records: int
    read_seconds: float
    cwt_seconds: float
    prepare_seconds: float
    detection_timing: dict[str, float] | None
    prepared_channels: list[Any]
    pelt_future: Future


def _token(value: object) -> str:
    text = "full" if value is None else str(value)
    return text.replace(".", "p").replace("/", "_").replace(" ", "_")


def build_run_id(config: CWTSearchConfig, reader: SpectrumReader) -> str:
    if config.run_id:
        return _token(config.run_id)
    source = Path(config.input).stem
    return "__".join(
        [
            source,
            f"f_{_token(config.f_start)}_{_token(config.f_stop)}",
            f"t_{_token(config.t_start)}_{_token(config.t_stop)}",
            f"cwt_{_token(config.period_min_records)}_{_token(config.period_max_records)}_{config.period_count}",
        ]
    )


def build_run_dir(config: CWTSearchConfig, reader: SpectrumReader) -> Path:
    return Path(config.output_dir) / build_run_id(config, reader)


def write_rows_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_candidates_csv(path: Path, candidates: list[dict]) -> None:
    write_rows_csv(path, candidates, RAW_CANDIDATE_FIELDNAMES)


def write_reviewed_candidates_csv(path: Path, candidates: list[dict]) -> None:
    write_rows_csv(path, candidates, REVIEWED_CANDIDATE_FIELDNAMES)


def write_time_windows_csv(path: Path, rows: list[dict]) -> None:
    write_rows_csv(path, rows, TIME_WINDOW_FIELDNAMES)


def write_manifest_csv(path: Path, rows: list[dict]) -> None:
    write_rows_csv(path, rows, MANIFEST_FIELDNAMES)


def write_summary_json(
    path: Path,
    config: CWTSearchConfig,
    reader: SpectrumReader,
    run_id: str,
    candidates: list[dict],
    reviewed_candidates: list[dict],
    channel_quality: dict[str, Any],
) -> None:
    vetoed_count = sum(1 for row in reviewed_candidates if row.get("candidate_status") == "vetoed")
    payload = {
        "schema_version": RAW_CANDIDATE_SCHEMA_VERSION,
        "run_id": run_id,
        "config": cwt_config_to_nested_dict(config),
        "runtime": runtime_info(),
        "source": reader.info(),
        "candidate_count": len(candidates),
        "reviewed_candidate_count": len(reviewed_candidates),
        "vetoed_candidate_count": vetoed_count,
        "channel_quality": channel_quality,
        "visualization": {
            "enabled": bool(config.visualization_enabled),
            "dir": "visualization" if config.visualization_enabled else "",
        },
        "top_candidates": candidates[:20],
        "notes": [
            "CPRO activity, native PELT windows, and CPRF outputs are candidates, not signal claims.",
            "Each physical frequency channel is detected independently.",
            "Absolute CWT power is calibrated by raw first-difference noise and wavelet gain.",
            "Native C++ PELT defines time windows from each channel's CPRO activity axis.",
            "CPRF evaluates unmasked absolute CWT inside each PELT window.",
            "CPRF uses the single configured concentration/contrast gate without a scientific fallback.",
            "Candidate periods require validation in the original time series.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def _compact_invalid_channel_ranges(rows: list[dict]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: int(row["channel"]))
    ranges: list[dict[str, Any]] = []
    for row in ordered:
        channel = int(row["channel"])
        reason = str(row["reason"])
        if ranges and ranges[-1]["reason"] == reason and ranges[-1]["channel_stop"] == channel:
            ranges[-1]["channel_stop"] = channel + 1
            ranges[-1]["count"] += 1
        else:
            ranges.append(
                {
                    "channel_start": channel,
                    "channel_stop": channel + 1,
                    "count": 1,
                    "reason": reason,
                }
            )
    return ranges


def _channel_quality_summary(
    *,
    selected_channel_count: int,
    invalid_channels: list[dict],
) -> dict[str, Any]:
    invalid_count = len(invalid_channels)
    valid_count = max(0, int(selected_channel_count) - invalid_count)
    if invalid_count == 0:
        status = "valid"
    elif valid_count == 0:
        status = "no_valid_channels"
    else:
        status = "invalid_channels_excluded"
    reason_counts = dict(sorted(Counter(str(row["reason"]) for row in invalid_channels).items()))
    ranges = _compact_invalid_channel_ranges(invalid_channels)
    return {
        "selected_channel_count": int(selected_channel_count),
        "valid_channel_count": valid_count,
        "invalid_channel_count": invalid_count,
        "quality_status": status,
        "invalid_reason_counts": reason_counts,
        "invalid_ranges": ranges,
    }


def _channel_progress(total: int, run_id: str, enabled: bool, leave: bool):
    if not enabled:
        return None
    from tqdm.auto import tqdm

    return tqdm(
        total=max(0, int(total)),
        desc=f"CWT channels {run_id}",
        unit="ch",
        leave=bool(leave),
        dynamic_ncols=True,
    )


def _emit(message: str, progress=None) -> None:
    if progress is not None:
        progress.write(message)
    else:
        print(message, flush=True)


def _timing_add(totals: dict[str, float] | None, key: str, seconds: float) -> None:
    if totals is not None:
        totals[key] = float(totals.get(key, 0.0)) + float(seconds)


def _timing_value(values: dict[str, float], key: str) -> float:
    return float(values.get(key, 0.0))


def _timing_block_message(
    run_id: str,
    block_id: str,
    block_channels: tuple[int, int],
    records: int,
    timings: dict[str, float],
    detection_timing: dict[str, float],
    candidates: int,
    windows: int,
) -> str:
    detect = _timing_value(timings, "detect_seconds")
    detail = (
        f"cpro={_timing_value(detection_timing, 'cpro_seconds'):.3f}s "
        f"pelt={_timing_value(detection_timing, 'pelt_seconds'):.3f}s "
        f"pelt_wait={_timing_value(detection_timing, 'pelt_wait_seconds'):.3f}s "
        f"pelt_candidates(mean={_timing_value(detection_timing, 'pelt_candidates_mean'):.1f} "
        f"max={_timing_value(detection_timing, 'pelt_candidates_max'):.0f}) "
        f"pelt_skip={_timing_value(detection_timing, 'pelt_short_circuit_channels'):.0f} "
        f"pelt_constant={_timing_value(detection_timing, 'pelt_constant_channels'):.0f} "
        f"cprf={_timing_value(detection_timing, 'cprf_seconds'):.3f}s"
    )
    return (
        f"[CWT TIMING] run_id={run_id} block={block_id} "
        f"ch={block_channels[0]}:{block_channels[1]} records={records} "
        f"read={_timing_value(timings, 'read_seconds'):.3f}s "
        f"cwt={_timing_value(timings, 'cwt_seconds'):.3f}s "
        f"detect={detect:.3f}s ({detail}) "
        f"windows={windows} candidates={candidates}"
    )


def _timing_summary_message(run_id: str, totals: dict[str, float], blocks: int) -> str:
    return (
        f"[CWT TIMING] SUMMARY run_id={run_id} blocks={blocks} "
        f"read={_timing_value(totals, 'read_seconds'):.3f}s "
        f"cwt={_timing_value(totals, 'cwt_seconds'):.3f}s "
        f"detect={_timing_value(totals, 'detect_seconds'):.3f}s "
        f"pelt_wait={_timing_value(totals, 'pelt_wait_seconds'):.3f}s "
        f"write={_timing_value(totals, 'write_seconds'):.3f}s "
        f"veto={_timing_value(totals, 'veto_seconds'):.3f}s "
        f"visualization={_timing_value(totals, 'visualization_seconds'):.3f}s "
        f"total={_timing_value(totals, 'total_seconds'):.3f}s"
    )


def _use_cuda_block_backend(backend: str, method: str, cuda_device: int) -> bool:
    backend_name = str(backend or "cpu").lower()
    if backend_name == "cuda":
        return True
    if backend_name != "auto" or method != "fft":
        return False
    try:
        from ..signal.cwt_cuda import cuda_available
    except ImportError:
        return False
    return cuda_available(device=int(cuda_device))


def run_cwt_search(config: CWTSearchConfig) -> Path:
    run_start = perf_counter()
    require_native_pelt()
    config = resolve_cwt_period_domain(config)
    validate_cwt_config(config)
    cprf_params = cprf_parameters_from_config(config)
    if not config.input:
        raise ValueError("config.input is required")
    reader = open_spectrum_reader(config.input)
    run_id = build_run_id(config, reader)
    run_dir = build_run_dir(config, reader)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.resolved.json").write_text(
        json.dumps(cwt_config_to_nested_dict(config), indent=2, ensure_ascii=True)
    )

    periods = period_grid_records(
        config.period_min_records,
        config.period_max_records,
        config.period_count,
        config.period_spacing,
    )
    noise_gain = impulse_cwt_noise_gain(
        periods,
        wavelet=config.wavelet,
        method=config.cwt_method,
    )
    selected_records = reader.record_slice(config.t_start, config.t_stop)
    selected_channels = reader.freq_slice(config.f_start, config.f_stop)
    selected_freqs = reader.freqs_mhz[selected_channels]
    progress = _channel_progress(
        total=int(selected_channels.stop - selected_channels.start),
        run_id=run_id,
        enabled=config.progress_enabled,
        leave=config.progress_leave,
    )
    all_candidates: list[dict] = []
    all_windows: list[dict] = []
    all_invalid_channels: list[dict] = []
    timing_enabled = bool(config.timing_enabled)
    timing_totals: dict[str, float] = {}
    block_count = 0
    use_cuda_block_backend = _use_cuda_block_backend(config.cwt_backend, config.cwt_method, config.cuda_device)
    if use_cuda_block_backend:
        from ..signal.cwt_cuda import _cupy, cwt_power_cube_cuda_gpu
        from ..signal.detection_cuda import (
            finalize_prepared_cuda_period_chunks,
            prepare_block_period_chunks_cuda_power,
            run_prepared_cuda_pelt,
        )
    pending_blocks: deque[_PendingCudaBlock] = deque()
    pelt_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="cwipss-pelt")
        if use_cuda_block_backend
        else None
    )
    pelt_cancellation = PeltCancellation() if pelt_executor is not None else None
    max_pending_blocks = max(1, int(config.cuda_max_pending_blocks))

    def record_block_results(
        *,
        block_id: str,
        channel_range: tuple[int, int],
        records: int,
        read_seconds: float,
        cwt_seconds: float,
        detect_seconds: float,
        detection_timing: dict[str, float] | None,
        candidates: list[dict],
        windows: list[dict],
    ) -> None:
        if timing_enabled and detection_timing is not None:
            _emit(
                _timing_block_message(
                    run_id,
                    block_id,
                    channel_range,
                    records,
                    {
                        "read_seconds": read_seconds,
                        "cwt_seconds": cwt_seconds,
                        "detect_seconds": detect_seconds,
                    },
                    detection_timing,
                    candidates=len(candidates),
                    windows=len(windows),
                ),
                progress=progress,
            )
        for row in windows:
            row["channel"] = channel_range[0] + int(row["channel"])
            row["schema_version"] = RAW_CANDIDATE_SCHEMA_VERSION
            row["run_id"] = run_id
            row["source_file"] = str(config.input)
            row["block_id"] = block_id
            row["block_ch0"] = channel_range[0]
            row["block_ch1"] = channel_range[1]
            all_windows.append(row)
        for row in candidates:
            row["channel"] = channel_range[0] + int(row["channel"])
            row["wavelet"] = config.wavelet
            row["time_agg"] = config.time_aggregation
            row["block_ch0"] = channel_range[0]
            row["block_ch1"] = channel_range[1]
            all_candidates.append(
                normalize_candidate_row(
                    row,
                    run_id=run_id,
                    source_file=config.input,
                    block_id=block_id,
                    tsamp_seconds=reader.tsamp_seconds,
                )
            )
        if progress is not None:
            progress.update(int(channel_range[1] - channel_range[0]))

    def finalize_pending_block(pending: _PendingCudaBlock) -> None:
        wait_start = perf_counter()
        segments_batch, pelt_seconds = pending.pelt_future.result()
        pelt_wait_seconds = perf_counter() - wait_start
        _timing_add(pending.detection_timing, "pelt_seconds", pelt_seconds)
        _timing_add(pending.detection_timing, "pelt_wait_seconds", pelt_wait_seconds)
        _timing_add(timing_totals, "pelt_wait_seconds", pelt_wait_seconds)
        finalize_start = perf_counter()
        candidates, windows = finalize_prepared_cuda_period_chunks(
            pending.prepared_channels,
            segments_batch,
            timing=pending.detection_timing,
        )
        finalize_seconds = perf_counter() - finalize_start
        detect_seconds = pending.prepare_seconds + pelt_wait_seconds + finalize_seconds
        _timing_add(timing_totals, "detect_seconds", detect_seconds)
        record_block_results(
            block_id=pending.block_id,
            channel_range=pending.channel_range,
            records=pending.records,
            read_seconds=pending.read_seconds,
            cwt_seconds=pending.cwt_seconds,
            detect_seconds=detect_seconds,
            detection_timing=pending.detection_timing,
            candidates=candidates,
            windows=windows,
        )

    try:
        block_channels = max(1, int(config.block_channels))
        for block_index, block_start in enumerate(range(int(selected_channels.start), int(selected_channels.stop), block_channels), start=1):
            read_start = perf_counter()
            block_stop = min(block_start + block_channels, int(selected_channels.stop))
            halo = slice(block_start, block_stop)
            block = reader.read_block(selected_records, halo)
            read_seconds = perf_counter() - read_start
            _timing_add(timing_totals, "read_seconds", read_seconds)
            block_count += 1
            block_id = f"block_{block_index:04d}"
            all_zero_mask = np.all(block.data == 0.0, axis=0)
            for local_channel in np.flatnonzero(all_zero_mask):
                all_invalid_channels.append(
                    {
                        "channel": block_start + int(local_channel),
                        "freq_mhz": float(block.freqs_mhz[local_channel]),
                        "finite_records": int(block.data.shape[0]),
                        "data_min": 0.0,
                        "data_max": 0.0,
                        "reason": "all_zero",
                    }
                )
            if bool(np.all(all_zero_mask)):
                if progress is not None:
                    progress.update(block_stop - block_start)
                continue
            cwt_start = perf_counter()
            if use_cuda_block_backend:
                cp = _cupy()
                cp.cuda.Device(int(config.cuda_device)).use()
                raw_device = cp.asarray(block.data, dtype=cp.float32)
                power = cwt_power_cube_cuda_gpu(
                    raw_device,
                    periods,
                    wavelet=config.wavelet,
                    method=config.cwt_method,
                    device=config.cuda_device,
                    normalize_channels=False,
                )
                if timing_enabled:
                    cp.cuda.Stream.null.synchronize()
            else:
                raw_device = block.data
                power = cwt_power_cube(
                    block.data,
                    periods,
                    wavelet=config.wavelet,
                    method=config.cwt_method,
                    backend=config.cwt_backend,
                    cuda_device=config.cuda_device,
                    normalize_channels=False,
                )
            cwt_seconds = perf_counter() - cwt_start
            _timing_add(timing_totals, "cwt_seconds", cwt_seconds)
            detection_timing: dict[str, float] | None = {} if timing_enabled else None
            block_invalid_channels: list[dict] = []
            detector_kwargs = dict(
                periods=periods,
                freqs_mhz=block.freqs_mhz,
                record_start=block.record_range[0],
                target_channel_start=block_start - int(halo.start),
                target_channel_stop=block_stop - int(halo.start),
                candidate_period_min_records=config.candidate_period_min_records,
                candidate_period_max_records=config.candidate_period_max_records,
                noise_gain=noise_gain,
                cpro_threshold_snr=config.cpro_threshold_snr,
                cpro_texture_quantile=config.cpro_texture_quantile,
                cpro_period_center_bins=config.cpro_period_center_bins,
                cpro_period_context_bins=config.cpro_period_context_bins,
                cpro_min_period_contrast=config.cpro_min_period_contrast,
                cpro_period_support_bins=config.cpro_period_support_bins,
                cpro_shape_power_softness=config.cpro_shape_power_softness,
                cpro_shape_contrast_softness=config.cpro_shape_contrast_softness,
                cpro_continuity_decay=config.cpro_continuity_decay,
                cpro_continuity_power=config.cpro_continuity_power,
                cpro_min_continuity_mean=config.cpro_min_continuity_mean,
                cpro_min_ridge_lock=config.cpro_min_ridge_lock,
                pelt_penalty=config.pelt_penalty,
                pelt_min_size_records=config.pelt_min_size_records,
                pelt_jump_records=config.pelt_jump_records,
                pelt_threads=config.pelt_threads,
                window_min_activity_mean=config.window_min_activity_mean,
                window_merge_gap_records=config.window_merge_gap_records,
                cuda_device=config.cuda_device,
                cprf_params=cprf_params,
                max_candidates_per_channel=config.max_candidates_per_channel,
                max_candidates_per_record=config.max_candidates_per_record,
                invalid_channel_mask=all_zero_mask,
                invalid_channels=block_invalid_channels,
            )
            if use_cuda_block_backend:
                prepare_start = perf_counter()
                prepared_channels = prepare_block_period_chunks_cuda_power(
                    power_cube=power,
                    raw_data=raw_device,
                    timing=detection_timing,
                    **detector_kwargs,
                )
                prepare_seconds = perf_counter() - prepare_start
                for row in block_invalid_channels:
                    row["channel"] = block_start + int(row["channel"])
                    all_invalid_channels.append(row)
                del power, raw_device
                pending_blocks.append(
                    _PendingCudaBlock(
                        block_id=block_id,
                        channel_range=(block_start, block_stop),
                        records=int(block.data.shape[0]),
                        read_seconds=read_seconds,
                        cwt_seconds=cwt_seconds,
                        prepare_seconds=prepare_seconds,
                        detection_timing=detection_timing,
                        prepared_channels=prepared_channels,
                        pelt_future=pelt_executor.submit(
                            run_prepared_cuda_pelt,
                            prepared_channels,
                            pelt_cancellation,
                            detection_timing,
                        ),
                    )
                )
                if len(pending_blocks) >= max_pending_blocks:
                    finalize_pending_block(pending_blocks.popleft())
                continue

            detect_start = perf_counter()
            candidates, windows = detect_block_periods(
                power_cube=power,
                raw_data=raw_device,
                timing=detection_timing,
                **detector_kwargs,
            )
            for row in block_invalid_channels:
                row["channel"] = block_start + int(row["channel"])
                all_invalid_channels.append(row)
            detect_seconds = perf_counter() - detect_start
            del power, raw_device
            _timing_add(timing_totals, "detect_seconds", detect_seconds)
            record_block_results(
                block_id=block_id,
                channel_range=(block_start, block_stop),
                records=int(block.data.shape[0]),
                read_seconds=read_seconds,
                cwt_seconds=cwt_seconds,
                detect_seconds=detect_seconds,
                detection_timing=detection_timing,
                candidates=candidates,
                windows=windows,
            )
        while pending_blocks:
            finalize_pending_block(pending_blocks.popleft())
    except KeyboardInterrupt:
        if pelt_cancellation is not None:
            pelt_cancellation.cancel()
        pending_blocks.clear()
        raise
    finally:
        if pelt_executor is not None:
            pelt_executor.shutdown(wait=True, cancel_futures=True)
        if progress is not None:
            progress.close()

    write_start = perf_counter()
    final_candidates = add_candidate_ids(all_candidates)
    quality_summary = _channel_quality_summary(
        selected_channel_count=int(selected_channels.stop - selected_channels.start),
        invalid_channels=all_invalid_channels,
    )
    write_time_windows_csv(run_dir / "time_windows.csv", all_windows)
    write_candidates_csv(run_dir / "candidates_raw.csv", final_candidates)
    if config.save_legacy_candidates_csv:
        write_candidates_csv(run_dir / "candidates.csv", final_candidates)
    _timing_add(timing_totals, "write_seconds", perf_counter() - write_start)

    veto_start = perf_counter()
    veto_context = VetoContext(
        record_start=int(selected_records.start),
        record_stop=int(selected_records.stop),
        freq_start_mhz=float(np.nanmin(selected_freqs)),
        freq_stop_mhz=float(np.nanmax(selected_freqs)),
    )
    reviewed_candidates = review_candidates(
        final_candidates,
        context=veto_context,
        config=veto_config_from_scan_config(config),
    )
    write_reviewed_candidates_csv(run_dir / "candidates_reviewed.csv", reviewed_candidates)
    _timing_add(timing_totals, "veto_seconds", perf_counter() - veto_start)

    write_start = perf_counter()
    manifest_row = make_manifest_row(
        run_id=run_id,
        source_info=reader.info(),
        record_start=selected_records.start,
        record_stop=selected_records.stop,
        f_start_mhz=config.f_start,
        f_stop_mhz=config.f_stop,
        candidate_count=len(final_candidates),
        channel_quality=quality_summary,
    )
    write_manifest_csv(run_dir / "manifest.csv", [manifest_row])
    write_summary_json(
        run_dir / "summary.json",
        config,
        reader,
        run_id,
        final_candidates,
        reviewed_candidates,
        quality_summary,
    )
    for invalid_range in quality_summary["invalid_ranges"]:
        _emit(
            f"[CWT WARNING] run_id={run_id} invalid channels="
            f"{invalid_range['channel_start']}:{invalid_range['channel_stop']} "
            f"reason={invalid_range['reason']} action=excluded",
            progress=None,
        )
    _timing_add(timing_totals, "write_seconds", perf_counter() - write_start)
    if config.visualization_enabled:
        visualization_start = perf_counter()
        from ..reporting.visualization import (
            CWTVisualizationConfig,
            SearchVisualizationConfig,
            visualize_cwt_stages,
        )

        selected_block = reader.read_block(selected_records, reader.freq_slice(config.f_start, config.f_stop))
        visualize_cwt_stages(
            selected_block.data,
            selected_block.freqs_mhz,
            run_dir / "visualization",
            SearchVisualizationConfig(
                wavelet=config.wavelet,
                cwt_method=config.cwt_method,
                cwt_backend=config.cwt_backend,
                cuda_device=config.cuda_device,
                periods=periods,
                block_channels=config.block_channels,
                candidate_period_min_records=config.candidate_period_min_records,
                candidate_period_max_records=config.candidate_period_max_records,
                time_aggregation=config.time_aggregation,
                aggregation_percentile=config.aggregation_percentile,
                cpro_threshold_snr=config.cpro_threshold_snr,
                cpro_texture_quantile=config.cpro_texture_quantile,
                cpro_period_center_bins=config.cpro_period_center_bins,
                cpro_period_context_bins=config.cpro_period_context_bins,
                cpro_min_period_contrast=config.cpro_min_period_contrast,
                cpro_period_support_bins=config.cpro_period_support_bins,
                cpro_shape_power_softness=config.cpro_shape_power_softness,
                cpro_shape_contrast_softness=config.cpro_shape_contrast_softness,
                cpro_continuity_decay=config.cpro_continuity_decay,
                cpro_continuity_power=config.cpro_continuity_power,
                cpro_min_continuity_mean=config.cpro_min_continuity_mean,
                cpro_min_ridge_lock=config.cpro_min_ridge_lock,
                cprf_threshold_snr=config.cprf_threshold_snr,
                cprf_texture_quantile=config.cprf_texture_quantile,
                cprf_smooth_bins=config.cprf_smooth_bins,
                cprf_peak_band_fraction=config.cprf_peak_band_fraction,
                cprf_min_width_bins=config.cprf_min_width_bins,
                cprf_min_peak_strength=config.cprf_min_peak_strength,
                cprf_min_integrated_strength=config.cprf_min_integrated_strength,
                cprf_min_band_persistence=config.cprf_min_band_persistence,
                cprf_min_band_concentration=config.cprf_min_band_concentration,
                cprf_min_local_contrast=config.cprf_min_local_contrast,
                cprf_harmonic_weight=config.cprf_harmonic_weight,
                cprf_harmonic_min_relative=config.cprf_harmonic_min_relative,
                cprf_harmonic_window_scale=config.cprf_harmonic_window_scale,
                cprf_max_peak_hypotheses=config.cprf_max_peak_hypotheses,
            ),
            raw_candidates=final_candidates,
            reviewed_candidates=reviewed_candidates,
            time_windows=all_windows,
            run_id=run_id,
            source_name=str(config.input),
            record_offset=int(selected_block.record_range[0]),
            config=CWTVisualizationConfig(
                enabled=True,
                max_blocks=config.visualization_max_blocks,
                max_channels=config.visualization_max_channels,
                top_candidates=config.visualization_top_candidates,
                dpi=config.visualization_dpi,
            ),
        )
        _timing_add(timing_totals, "visualization_seconds", perf_counter() - visualization_start)
    timing_totals["total_seconds"] = perf_counter() - run_start
    if timing_enabled:
        _emit(_timing_summary_message(run_id, timing_totals, block_count), progress=None)
    return run_dir
