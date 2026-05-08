from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .config import SWTScanConfig, swt_config_to_nested_dict
from .detection import add_candidate_ids, robust_score_2d, summarize_components
from .io import CE4Reader
from .models import (
    MANIFEST_FIELDNAMES,
    RAW_CANDIDATE_FIELDNAMES,
    REVIEWED_CANDIDATE_FIELDNAMES,
    make_manifest_row,
    normalize_candidate_row,
)
from .runtime import runtime_info
from .swt import approximate_scale_records, swt_detail_power_matrix
from .veto import VetoContext, review_candidates, veto_config_from_scan_config


def _token(value: object) -> str:
    text = "full" if value is None else str(value)
    return text.replace(".", "p").replace("/", "_").replace(" ", "_")


def build_run_id(config: SWTScanConfig, reader: CE4Reader) -> str:
    if config.run_id:
        return _token(config.run_id)
    source = Path(config.input).stem
    return "__".join(
        [
            source,
            f"f_{_token(config.f_start)}_{_token(config.f_stop)}",
            f"t_{_token(config.t_start)}_{_token(config.t_stop)}",
            f"swt_{_token(config.wavelet)}_L{config.levels}",
        ]
    )


def build_run_dir(config: SWTScanConfig, reader: CE4Reader) -> Path:
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
    config: SWTScanConfig,
    reader: CE4Reader,
    run_id: str,
    candidates: list[dict],
    reviewed_candidates: list[dict],
) -> None:
    vetoed_count = sum(1 for row in reviewed_candidates if row.get("candidate_status") == "vetoed")
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "config": swt_config_to_nested_dict(config),
        "runtime": runtime_info(),
        "source": reader.info(),
        "candidate_count": len(candidates),
        "reviewed_candidate_count": len(reviewed_candidates),
        "vetoed_candidate_count": vetoed_count,
        "top_candidates": candidates[:20],
        "notes": [
            "SWT bright components are candidates, not signal claims.",
            "SWT level scale is approximate; refine candidate periods in validation.",
            "Frequency-block boundary components may be split in this prototype.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def run_swt_scan(config: SWTScanConfig) -> Path:
    if not config.input:
        raise ValueError("config.input is required")
    reader = CE4Reader(config.input)
    run_id = build_run_id(config, reader)
    run_dir = build_run_dir(config, reader)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.resolved.json").write_text(
        json.dumps(swt_config_to_nested_dict(config), indent=2, ensure_ascii=True)
    )

    all_candidates: list[dict] = []
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
        powers, level_numbers = swt_detail_power_matrix(
            block.data,
            wavelet=config.wavelet,
            levels=config.levels,
            normalize_channels=True,
        )
        for level_idx, level_number in enumerate(level_numbers):
            log_power = np.log10(powers[level_idx] + 1e-12)
            score = robust_score_2d(
                log_power,
                local_time=config.local_time,
                local_freq=min(config.local_freq, max(3, block.freqs_mhz.size | 1)),
            )
            candidates = summarize_components(
                score=score,
                freqs_mhz=block.freqs_mhz,
                record_start=block.record_range[0],
                level_number=int(level_number),
                threshold=config.threshold,
                min_pixels=config.min_pixels,
                max_components=config.max_candidates_per_block,
            )
            for row in candidates:
                row["approx_scale_records"] = approximate_scale_records(int(level_number))
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

    final_candidates = add_candidate_ids(all_candidates)
    write_candidates_csv(run_dir / "candidates_raw.csv", final_candidates)
    if config.save_legacy_candidates_csv:
        write_candidates_csv(run_dir / "candidates.csv", final_candidates)

    selected_records = reader.record_slice(config.t_start, config.t_stop)
    selected_freqs = reader.freqs_mhz[reader.freq_slice(config.f_start, config.f_stop)]
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
    return run_dir
