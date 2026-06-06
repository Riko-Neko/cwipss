"""Per-candidate raw and CWT image gallery generation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..config import CWTSearchConfig, cwt_config_from_mapping
from ..signal.cwt import cwt_power_cube, period_grid_records
from ..data.readers import SpectrumReader, open_spectrum_reader
from .report import read_csv_rows, read_json
from .plotting import ImageIndex, cwt_view, number, raw_view


@dataclass(frozen=True)
class CandidateGalleryConfig:
    top_n: int = 100
    sort_by: str = "auto"
    include_vetoed: bool = False
    context_periods: float = 16
    min_window_records: int = 256
    max_window_records: int = 4096
    freq_context_channels: int = 8
    period_radius: float = 2
    dpi: int = 140
    cwt_backend: str | None = None
    cuda_device: int | None = None


def _batch(path: Path) -> bool:
    return (path / "batch_config.resolved.json").exists()


def _rows(path: Path) -> list[dict[str, Any]]:
    suffix = ".all" if _batch(path) else ""
    candidate = path / f"candidates_reviewed{suffix}.csv"
    if not candidate.exists():
        candidate = path / f"candidates_raw{suffix}.csv"
    rows = [dict(row) for row in read_csv_rows(candidate)]
    stats = {
        (row.get("run_id"), row.get("candidate_id")): row
        for row in read_csv_rows(path / f"validation_reviewed{suffix}.csv")
    }
    fields = {
        "validation_status", "refined_period_records", "refined_period_seconds",
        "p_value", "q_value", "global_q_value", "evidence_rank", "fold_profile_snr",
    }
    for row in rows:
        match = stats.get((row.get("run_id"), row.get("candidate_id")), {})
        row.update({key: match[key] for key in fields & match.keys()})
    return rows


def select_candidate_rows(rows, *, top_n=100, sort_by="auto", include_vetoed=False):
    rows = [row for row in rows if include_vetoed or row.get("candidate_status") != "vetoed"]
    mode = sort_by
    if mode == "auto":
        mode = "evidence_rank" if any(math.isfinite(number(row, "evidence_rank")) for row in rows) else "integrated_score"
    if mode == "integrated_score":
        key, reverse = lambda row: (number(row, mode, -math.inf), number(row, "peak_score", -math.inf)), True
    elif mode in {"evidence_rank", "global_q_value"}:
        key, reverse = lambda row: (
            number(row, mode, math.inf), number(row, "q_value", math.inf),
            number(row, "p_value", math.inf), -number(row, "integrated_score", -math.inf),
        ), False
    else:
        raise ValueError(f"Unsupported candidate gallery sort mode: {sort_by}")
    rows.sort(key=key, reverse=reverse)
    return rows if top_n <= 0 else rows[:top_n]


def _source(row, root: Path | None, project: Path) -> Path:
    source = Path(str(row.get("source_file", "")))
    options = [source, project / source]
    if root:
        options += [root / source.name, root / source]
    try:
        return next(path for path in options if path.is_file())
    except StopIteration:
        raise FileNotFoundError("Source data not found: " + ", ".join(map(str, options)))


def _scan(path: Path, run_id: str) -> CWTSearchConfig:
    single = path / "config.resolved.json"
    per_run = path / "files" / run_id / "config.resolved.json"
    if single.exists() or per_run.exists():
        return cwt_config_from_mapping(read_json(per_run if per_run.exists() else single))
    payload = read_json(path / "batch_config.resolved.json")
    config = cwt_config_from_mapping(payload.get("scan_config", {}))
    job = next((job for job in payload.get("jobs", []) if str(job.get("run_id")) == run_id), {})
    overrides = {
        key: job[key] for key in ("input", "f_start", "f_stop", "t_start", "t_stop")
        if job.get(key) is not None
    }
    return replace(config, run_id=run_id, **overrides)


def _slices(row, reader: SpectrumReader, cfg: CandidateGalleryConfig):
    period = max(1, number(row, "peak_period_records", 1))
    duration = max(1, int(number(row, "duration_records", 1)))
    size = min(
        max(cfg.min_window_records, math.ceil(period * cfg.context_periods), min(duration, cfg.max_window_records)),
        cfg.max_window_records, reader.n_records,
    )
    center = int(number(row, "peak_record", number(row, "record_start", 0)))
    start = min(max(0, center - size // 2), reader.n_records - size)
    freq = number(row, "peak_freq_mhz")
    channel = (
        int(np.nanargmin(abs(reader.freqs_mhz - freq))) if math.isfinite(freq)
        else int(number(row, "block_channel_start")) + int(number(row, "channel_index"))
    )
    channel = min(max(channel, 0), reader.n_channels - 1)
    radius = max(0, cfg.freq_context_channels)
    return slice(start, start + size), slice(max(0, channel - radius), min(reader.n_channels, channel + radius + 1)), channel


def _render(row, reader, scan, cfg, output, rank):
    records, channels, channel = _slices(row, reader, cfg)
    block = reader.read_block(records, channels)
    periods = period_grid_records(scan.period_min_records, scan.period_max_records, scan.period_count, scan.period_spacing)
    backend = cfg.cwt_backend or scan.cwt_backend
    power = cwt_power_cube(
        block.data[:, channel - channels.start : channel - channels.start + 1],
        periods, wavelet=scan.wavelet, method=scan.cwt_method, backend=backend,
        cuda_device=scan.cuda_device if cfg.cuda_device is None else cfg.cuda_device,
    )[:, :, 0]
    title = f"rank {rank}, candidate {row.get('candidate_id', '-')}"
    filename = (
        f"{rank:04d}_{row.get('run_id', 'run')}_candidate_{row.get('candidate_id', '-')}.png"
        .replace("/", "_")
    )
    raw = raw_view(
        output / "raw" / filename, block.data, block.freqs_mhz,
        offset=records.start, title=f"Stage 01 input matrix: {title}", candidates=[row], dpi=cfg.dpi,
    )
    seed, radius = number(row, "peak_period_records"), max(1, cfg.period_radius)
    cwt = cwt_view(
        output / "cwt" / filename, power, periods,
        offset=records.start, title=f"Stage 02 CWT scalogram: {title}, channel {channel}",
        candidates=[row], refined=number(row, "refined_period_records"),
        ylim=(max(periods.min(), seed / radius), min(periods.max(), seed * radius)), dpi=cfg.dpi,
    )
    return {"raw_image": raw, "cwt_image": cwt, "backend": backend}


def visualize_candidate_gallery(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    source_root: str | Path | None = None,
    project_dir: str | Path | None = None,
    config: CandidateGalleryConfig | None = None,
    reader_factory: Callable[[str | Path], SpectrumReader] = open_spectrum_reader,
) -> Path:
    run_dir, cfg = Path(run_dir), config or CandidateGalleryConfig()
    output = Path(output_dir) if output_dir else run_dir / "candidate_gallery"
    output.mkdir(parents=True, exist_ok=True)
    selected = select_candidate_rows(
        _rows(run_dir), top_n=cfg.top_n, sort_by=cfg.sort_by, include_vetoed=cfg.include_vetoed
    )
    index = ImageIndex(
        output,
        f"Candidate Gallery: {run_dir.name}",
        {"gallery_config": cfg.__dict__, "candidate_count": len(selected), "rendered_count": 0, "error_count": 0},
    )
    index.write()
    readers, results = {}, []
    root = Path(source_root) if source_root else None
    project = Path(project_dir) if project_dir else Path.cwd()
    for rank, row in enumerate(selected, 1):
        result = {"display_rank": rank, **row, "status": "error", "error": ""}
        try:
            path = _source(row, root, project)
            if path not in readers:
                readers[path] = reader_factory(path)
            reader = readers[path]
            rendered = _render(row, reader, _scan(run_dir, str(row.get("run_id", ""))), cfg, output, rank)
            result.update({key: str(Path(value).relative_to(output)) if key.endswith("_image") else value for key, value in rendered.items()})
            result["status"] = "complete"
            label = f"Rank {rank}: {row.get('run_id', '')} candidate {row.get('candidate_id', '')}"
            index.add(label + " - Stage 01 Input Matrix", output / result["raw_image"], "Raw candidate-centered matrix.")
            index.add(label + " - Stage 02 CWT Scalogram", output / result["cwt_image"], "Candidate-channel CWT.")
        except Exception as exc:
            result["error"] = str(exc)
        results.append(result)
        index.metadata.update(
            rendered_count=sum(item["status"] == "complete" for item in results),
            error_count=sum(item["status"] != "complete" for item in results),
        )
        index.write()
    return index.write()
