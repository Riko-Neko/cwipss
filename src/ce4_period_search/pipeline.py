from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .config import CWTSearchConfig, cwt_config_to_nested_dict
from .cwt import cwt_power_cube, period_grid_records
from .detection import add_candidate_ids, summarize_scalogram_regions
from .io import CE4Reader
from .models import (
    MANIFEST_FIELDNAMES,
    RAW_CANDIDATE_FIELDNAMES,
    REVIEWED_CANDIDATE_FIELDNAMES,
    make_manifest_row,
    normalize_candidate_row,
)
from .runtime import runtime_info
from .veto import VetoContext, review_candidates, veto_config_from_scan_config


def _token(value: object) -> str:
    text = "full" if value is None else str(value)
    return text.replace(".", "p").replace("/", "_").replace(" ", "_")


def build_run_id(config: CWTSearchConfig, reader: CE4Reader) -> str:
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


def build_run_dir(config: CWTSearchConfig, reader: CE4Reader) -> Path:
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


def write_manifest_csv(path: Path, rows: list[dict]) -> None:
    write_rows_csv(path, rows, MANIFEST_FIELDNAMES)


def write_summary_json(
    path: Path,
    config: CWTSearchConfig,
    reader: CE4Reader,
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
            "CWT per-channel scalogram regions are candidates, not signal claims.",
            "Time aggregation creates only the period-channel overview map, not the primary detector input.",
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


def run_cwt_search(config: CWTSearchConfig) -> Path:
    if not config.input:
        raise ValueError("config.input is required")
    reader = CE4Reader(config.input)
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
    try:
        for block_index, block in enumerate(
            reader.iter_frequency_blocks(
                f_start=config.f_start,
                f_stop=config.f_stop,
                t_start=config.t_start,
                t_stop=config.t_stop,
                block_channels=config.block_channels,
            ),
            start=1,
        ):
            block_id = f"block_{block_index:04d}"
            power = cwt_power_cube(
                block.data,
                periods,
                wavelet=config.wavelet,
                method=config.cwt_method,
                normalize_channels=True,
            )
            candidates, _score_cube = summarize_scalogram_regions(
                power_cube=power,
                periods=periods,
                freqs_mhz=block.freqs_mhz,
                record_start=block.record_range[0],
                threshold=config.threshold,
                sigma_period_peak=config.dog_sigma_peak,
                sigma_period_background=config.dog_sigma_background,
                sigma_time=config.time_smooth_sigma,
                min_duration_records=config.min_duration_records,
                min_width_bins=config.min_width_bins,
                max_width_bins=config.max_width_bins,
                max_candidates_per_channel=config.max_candidates_per_channel,
                max_candidates=config.max_candidates_per_block,
            )
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

    final_candidates = add_candidate_ids(all_candidates)
    write_candidates_csv(run_dir / "candidates_raw.csv", final_candidates)
    if config.save_legacy_candidates_csv:
        write_candidates_csv(run_dir / "candidates.csv", final_candidates)

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
    if config.visualization_enabled:
        from .visualization import CWTVisualizationConfig, SearchVisualizationConfig, visualize_cwt_stages

        selected_block = reader.read_block(selected_records, reader.freq_slice(config.f_start, config.f_stop))
        visualize_cwt_stages(
            selected_block.data,
            selected_block.freqs_mhz,
            run_dir / "visualization",
            SearchVisualizationConfig(
                wavelet=config.wavelet,
                cwt_method=config.cwt_method,
                periods=periods,
                block_channels=config.block_channels,
                threshold=config.threshold,
                dog_sigma_peak=config.dog_sigma_peak,
                dog_sigma_background=config.dog_sigma_background,
                time_smooth_sigma=config.time_smooth_sigma,
                time_aggregation=config.time_aggregation,
                aggregation_percentile=config.aggregation_percentile,
            ),
            raw_candidates=final_candidates,
            reviewed_candidates=reviewed_candidates,
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
    return run_dir
