from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import CWTSearchConfig, cwt_config_to_nested_dict
from .cwt import cwt_power_cube, period_grid_records
from .detection import add_candidate_ids, detect_block_periods
from .io import SpectrumReader, open_spectrum_reader
from .models import (
    MANIFEST_FIELDNAMES,
    RAW_CANDIDATE_FIELDNAMES,
    REVIEWED_CANDIDATE_FIELDNAMES,
    TIME_WINDOW_FIELDNAMES,
    make_manifest_row,
    normalize_candidate_row,
)
from .runtime import runtime_info
from .veto import VetoContext, review_candidates, veto_config_from_scan_config


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
) -> None:
    vetoed_count = sum(1 for row in reviewed_candidates if row.get("candidate_status") == "vetoed")
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "config": cwt_config_to_nested_dict(config),
        "runtime": runtime_info(),
        "source": reader.info(),
        "candidate_count": len(candidates),
        "reviewed_candidate_count": len(reviewed_candidates),
        "vetoed_candidate_count": vetoed_count,
        "visualization": {
            "enabled": bool(config.visualization_enabled),
            "dir": "visualization" if config.visualization_enabled else "",
        },
        "top_candidates": candidates[:20],
        "notes": [
            "CWT single-channel low-floor PELT/profile detections are candidates, not signal claims.",
            "Frequency channels are processed independently; no cross-frequency public operation is used.",
            "Candidate period-domain filtering is applied before activity and profile scoring.",
            "Candidate periods require validation in the original time series.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


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


def _timing_add(totals: dict[str, float], key: str, seconds: float) -> None:
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
        f"floor_excess={_timing_value(detection_timing, 'floor_excess_seconds'):.3f}s "
        f"structure={_timing_value(detection_timing, 'structure_seconds'):.3f}s "
        f"activity={_timing_value(detection_timing, 'activity_seconds'):.3f}s "
        f"pelt={_timing_value(detection_timing, 'pelt_seconds'):.3f}s "
        f"profile={_timing_value(detection_timing, 'profile_seconds'):.3f}s"
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
        from .cwt_cuda import cuda_available
    except ImportError:
        return False
    return cuda_available(device=int(cuda_device))


def run_cwt_search(config: CWTSearchConfig) -> Path:
    run_start = perf_counter()
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
    timing_enabled = bool(config.timing_enabled)
    timing_totals: dict[str, float] = {}
    block_count = 0
    use_cuda_block_backend = _use_cuda_block_backend(config.cwt_backend, config.cwt_method, config.cuda_device)
    if use_cuda_block_backend:
        from .cwt_cuda import cwt_power_cube_cuda_gpu
        from .detection_cuda import detect_block_periods_cuda_power
    try:
        block_channels = max(1, int(config.block_channels))
        for block_index, block_start in enumerate(range(int(selected_channels.start), int(selected_channels.stop), block_channels), start=1):
            read_start = perf_counter()
            block_stop = min(block_start + block_channels, int(selected_channels.stop))
            block = reader.read_block(selected_records, slice(block_start, block_stop))
            read_seconds = perf_counter() - read_start
            _timing_add(timing_totals, "read_seconds", read_seconds)
            block_count += 1
            block_id = f"block_{block_index:04d}"
            cwt_start = perf_counter()
            if use_cuda_block_backend:
                power = cwt_power_cube_cuda_gpu(
                    block.data,
                    periods,
                    wavelet=config.wavelet,
                    method=config.cwt_method,
                    device=config.cuda_device,
                    normalize_channels=True,
                )
                if timing_enabled:
                    from .cwt_cuda import _cupy

                    _cupy().cuda.Stream.null.synchronize()
            else:
                power = cwt_power_cube(
                    block.data,
                    periods,
                    wavelet=config.wavelet,
                    method=config.cwt_method,
                    backend=config.cwt_backend,
                    cuda_device=config.cuda_device,
                    normalize_channels=True,
                )
            cwt_seconds = perf_counter() - cwt_start
            _timing_add(timing_totals, "cwt_seconds", cwt_seconds)
            detection_timing: dict[str, float] | None = {} if timing_enabled else None
            detect_start = perf_counter()
            detector = detect_block_periods_cuda_power if use_cuda_block_backend else detect_block_periods
            candidates, windows = detector(
                power_cube=power,
                periods=periods,
                freqs_mhz=block.freqs_mhz,
                record_start=block.record_range[0],
                candidate_period_min_records=config.candidate_period_min_records,
                candidate_period_max_records=config.candidate_period_max_records,
                noise_floor_fraction=config.noise_floor_fraction,
                excess_eps_fraction=config.excess_eps_fraction,
                structure_baseline_quantile=config.structure_baseline_quantile,
                structure_scale_quantile=config.structure_scale_quantile,
                structure_z_threshold=config.structure_z_threshold,
                structure_time_support_records=config.structure_time_support_records,
                structure_period_support_bins=config.structure_period_support_bins,
                structure_min_support_fraction=config.structure_min_support_fraction,
                activity_trim_low=config.activity_trim_low,
                activity_trim_high=config.activity_trim_high,
                activity_smooth_records=config.activity_smooth_records,
                pelt_penalty=config.pelt_penalty,
                pelt_min_size_records=config.pelt_min_size_records,
                window_min_duration_records=config.window_min_duration_records,
                window_min_activity_mean=config.window_min_activity_mean,
                window_min_activity_raw_mean=config.window_min_activity_raw_mean,
                window_merge_gap_records=config.window_merge_gap_records,
                profile_min_prominence=config.profile_min_prominence,
                profile_max_peaks_per_window=config.profile_max_peaks_per_window,
                max_candidates_per_channel=config.max_candidates_per_channel,
                max_candidates=config.max_candidates_per_block,
                timing=detection_timing,
            )
            detect_seconds = perf_counter() - detect_start
            del power
            _timing_add(timing_totals, "detect_seconds", detect_seconds)
            if timing_enabled and detection_timing is not None:
                _emit(
                    _timing_block_message(
                        run_id,
                        block_id,
                        block.channel_range,
                        int(block.data.shape[0]),
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
                row["schema_version"] = 1
                row["run_id"] = run_id
                row["source_file"] = str(config.input)
                row["block_id"] = block_id
                row["block_channel_start"] = block.channel_range[0]
                row["block_channel_stop"] = block.channel_range[1]
                all_windows.append(row)
            for row in candidates:
                row["cwt_wavelet"] = config.wavelet
                row["time_aggregation"] = config.time_aggregation
                row["block_channel_start"] = block.channel_range[0]
                row["block_channel_stop"] = block.channel_range[1]
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
                progress.update(int(block.data.shape[1]))
    finally:
        if progress is not None:
            progress.close()

    write_start = perf_counter()
    final_candidates = add_candidate_ids(all_candidates)
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
    )
    write_manifest_csv(run_dir / "manifest.csv", [manifest_row])
    write_summary_json(
        run_dir / "summary.json",
        config,
        reader,
        run_id,
        final_candidates,
        reviewed_candidates,
    )
    _timing_add(timing_totals, "write_seconds", perf_counter() - write_start)
    if config.visualization_enabled:
        visualization_start = perf_counter()
        from .visualization import CWTVisualizationConfig, SearchVisualizationConfig, visualize_cwt_stages

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
                noise_floor_fraction=config.noise_floor_fraction,
                excess_eps_fraction=config.excess_eps_fraction,
                structure_baseline_quantile=config.structure_baseline_quantile,
                structure_scale_quantile=config.structure_scale_quantile,
                structure_z_threshold=config.structure_z_threshold,
                structure_time_support_records=config.structure_time_support_records,
                structure_period_support_bins=config.structure_period_support_bins,
                structure_min_support_fraction=config.structure_min_support_fraction,
                activity_trim_low=config.activity_trim_low,
                activity_trim_high=config.activity_trim_high,
                activity_smooth_records=config.activity_smooth_records,
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
