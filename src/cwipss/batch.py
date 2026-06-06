from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import CWTSearchConfig, cwt_config_to_nested_dict
from .models import (
    BATCH_MANIFEST_FIELDNAMES,
    RAW_CANDIDATE_FIELDNAMES,
    REVIEWED_CANDIDATE_FIELDNAMES,
    TIME_WINDOW_FIELDNAMES,
    VALIDATION_FIELDNAMES,
    VALIDATION_REVIEWED_FIELDNAMES,
)
from .pipeline import run_cwt_search
from .stats import run_stats
from .validation import (
    read_csv_rows,
    validate_candidate_rows,
    validation_config_from_scan_config,
    write_validation_outputs,
)


@dataclass(frozen=True)
class BatchJob:
    input: str
    run_id: str | None = None
    f_start: float | None = None
    f_stop: float | None = None
    t_start: int | None = None
    t_stop: int | None = None


@dataclass(frozen=True)
class BatchConfig:
    batch_id: str
    output_dir: str = "runs"
    validate: bool = True
    stats: bool = True


ANSI_RESET = "\033[0m"
ANSI_BOLD_CYAN = "\033[1;36m"
ANSI_BOLD_GREEN = "\033[1;32m"
ANSI_BOLD_RED = "\033[1;31m"


def default_batch_id() -> str:
    return "batch_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _token(value: object) -> str:
    text = str(value)
    return text.replace(".", "p").replace("/", "_").replace(" ", "_")


def _color(text: str, color: str) -> str:
    return f"{color}{text}{ANSI_RESET}"


def _batch_progress(total: int, enabled: bool, leave: bool):
    if not enabled:
        return None
    from tqdm.auto import tqdm

    return tqdm(
        total=max(0, int(total)),
        desc="Batch files",
        unit="file",
        leave=bool(leave),
        dynamic_ncols=True,
    )


def _emit(message: str, progress=None) -> None:
    if progress is not None:
        progress.write(message)
    else:
        print(message, flush=True)


def run_id_for_input(path: str | Path, index: int) -> str:
    return f"{index:04d}_{_token(Path(path).stem)}"


def _coerce_float(value: object) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _coerce_int(value: object) -> int | None:
    if value in ("", None):
        return None
    return int(value)


def _relpath_if_possible(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path)


def discover_input_files(input_dir: str | Path, pattern: str = "*.2C", project_dir: str | Path | None = None) -> list[str]:
    root = Path(input_dir)
    files = sorted(path for path in root.glob(pattern) if path.is_file())
    if project_dir is None:
        return [str(path) for path in files]
    base = Path(project_dir)
    return [_relpath_if_possible(path, base) for path in files]


def read_batch_manifest(path: str | Path) -> list[BatchJob]:
    rows: list[BatchJob] = []
    with Path(path).open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            input_path = row.get("input") or row.get("source_file")
            if not input_path:
                raise ValueError("Batch manifest rows require input or source_file")
            rows.append(
                BatchJob(
                    input=input_path,
                    run_id=row.get("run_id") or None,
                    f_start=_coerce_float(row.get("f_start") or row.get("f_start_mhz")),
                    f_stop=_coerce_float(row.get("f_stop") or row.get("f_stop_mhz")),
                    t_start=_coerce_int(row.get("t_start")),
                    t_stop=_coerce_int(row.get("t_stop")),
                )
            )
    return rows


def make_batch_jobs(inputs: list[str], manifest: str | Path | None = None) -> list[BatchJob]:
    jobs = read_batch_manifest(manifest) if manifest is not None else [BatchJob(input=value) for value in inputs]
    return ensure_run_ids(jobs)


def ensure_run_ids(jobs: list[BatchJob]) -> list[BatchJob]:
    return [replace(job, run_id=job.run_id or run_id_for_input(job.input, idx)) for idx, job in enumerate(jobs, start=1)]


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _read_rows_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_csv_rows(path)


def _merge_csvs(run_dirs: list[Path], filename: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_dir in run_dirs:
        rows.extend(_read_rows_if_exists(run_dir / filename))
    return rows


def _summary_counts(run_dir: Path) -> dict[str, int]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {"candidate_count": 0, "vetoed_candidate_count": 0}
    payload = json.loads(summary_path.read_text())
    return {
        "candidate_count": int(payload.get("candidate_count", 0)),
        "vetoed_candidate_count": int(payload.get("vetoed_candidate_count", 0)),
    }


def _job_scan_config(base_config: CWTSearchConfig, job: BatchJob, batch_files_dir: Path) -> CWTSearchConfig:
    return replace(
        base_config,
        input=job.input,
        run_id=job.run_id,
        output_dir=str(batch_files_dir),
        f_start=job.f_start if job.f_start is not None else base_config.f_start,
        f_stop=job.f_stop if job.f_stop is not None else base_config.f_stop,
        t_start=job.t_start if job.t_start is not None else base_config.t_start,
        t_stop=job.t_stop if job.t_stop is not None else base_config.t_stop,
    )


def _run_validation_for_dir(run_dir: Path, scan_config: CWTSearchConfig, project_dir: Path) -> list[dict[str, Any]]:
    candidate_path = run_dir / "candidates_reviewed.csv"
    rows = read_csv_rows(candidate_path)
    validation_config = validation_config_from_scan_config(scan_config)
    validation_rows = validate_candidate_rows(rows, validation_config, project_dir=project_dir)
    write_validation_outputs(
        run_dir / "validation_summary.csv",
        run_dir / "validation",
        validation_rows,
        validation_config,
    )
    return validation_rows


def _write_batch_config(batch_dir: Path, batch_config: BatchConfig, base_config: CWTSearchConfig, jobs: list[BatchJob]) -> None:
    payload = {
        "batch": asdict(batch_config),
        "scan_config": cwt_config_to_nested_dict(base_config),
        "jobs": [asdict(job) for job in jobs],
    }
    (batch_dir / "batch_config.resolved.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def run_batch(
    jobs: list[BatchJob],
    base_config: CWTSearchConfig,
    batch_config: BatchConfig,
    project_dir: str | Path,
) -> Path:
    project_dir = Path(project_dir)
    batch_dir = Path(batch_config.output_dir) / batch_config.batch_id
    files_dir = batch_dir / "files"
    batch_dir.mkdir(parents=True, exist_ok=True)
    jobs = ensure_run_ids(jobs)
    _write_batch_config(batch_dir, batch_config, base_config, jobs)

    manifest_rows: list[dict[str, Any]] = []
    successful_run_dirs: list[Path] = []
    progress = _batch_progress(
        total=len(jobs),
        enabled=bool(base_config.progress_enabled),
        leave=bool(base_config.progress_leave),
    )
    try:
        iterable = enumerate(jobs, start=1)
        for job_index, job in iterable:
            _emit(
                _color(
                    f"[CWT BATCH] START {job_index}/{len(jobs)} run_id={job.run_id} file={Path(job.input).name}",
                    ANSI_BOLD_CYAN,
                ),
                progress=progress,
            )
            start_time = time.perf_counter()
            run_dir = files_dir / str(job.run_id)
            status = "complete"
            error = ""
            validation_count = 0
            stats_count = 0
            try:
                scan_config = _job_scan_config(base_config, job, files_dir)
                run_dir = run_cwt_search(scan_config)
                successful_run_dirs.append(run_dir)
                if batch_config.validate:
                    validation_rows = _run_validation_for_dir(run_dir, scan_config, project_dir=project_dir)
                    validation_count = len(validation_rows)
                if batch_config.stats and (run_dir / "validation_summary.csv").exists():
                    stats_rows = run_stats(run_dir / "validation_summary.csv", run_dir / "validation_reviewed.csv")
                    stats_count = len(stats_rows)
                counts = _summary_counts(run_dir)
            except Exception as exc:
                status = "error"
                error = str(exc)
                counts = {"candidate_count": 0, "vetoed_candidate_count": 0}
            duration = time.perf_counter() - start_time
            manifest_rows.append(
                {
                    "batch_id": batch_config.batch_id,
                    "run_id": job.run_id,
                    "source_file": job.input,
                    "run_dir": str(run_dir),
                    "status": status,
                    "error": error,
                    "duration_seconds": f"{duration:.3f}",
                    "candidate_count": counts["candidate_count"],
                    "vetoed_candidate_count": counts["vetoed_candidate_count"],
                    "validation_count": validation_count,
                    "stats_count": stats_count,
                }
            )
            _write_rows_csv(batch_dir / "manifest.csv", manifest_rows, BATCH_MANIFEST_FIELDNAMES)
            if status == "complete":
                _emit(
                    _color(
                        f"[CWT BATCH] DONE  {job_index}/{len(jobs)} run_id={job.run_id} "
                        f"candidates={counts['candidate_count']} validation={validation_count} "
                        f"duration={duration:.1f}s results={run_dir}",
                        ANSI_BOLD_GREEN,
                    ),
                    progress=progress,
                )
            else:
                _emit(
                    _color(
                        f"[CWT BATCH] ERROR {job_index}/{len(jobs)} run_id={job.run_id} file={Path(job.input).name}: {error}",
                        ANSI_BOLD_RED,
                    ),
                    progress=progress,
                )
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    _write_rows_csv(batch_dir / "manifest.csv", manifest_rows, BATCH_MANIFEST_FIELDNAMES)
    _write_rows_csv(
        batch_dir / "candidates_raw.all.csv",
        _merge_csvs(successful_run_dirs, "candidates_raw.csv"),
        RAW_CANDIDATE_FIELDNAMES,
    )
    _write_rows_csv(
        batch_dir / "candidates_reviewed.all.csv",
        _merge_csvs(successful_run_dirs, "candidates_reviewed.csv"),
        REVIEWED_CANDIDATE_FIELDNAMES,
    )
    _write_rows_csv(
        batch_dir / "time_windows.all.csv",
        _merge_csvs(successful_run_dirs, "time_windows.csv"),
        TIME_WINDOW_FIELDNAMES,
    )

    validation_rows = _merge_csvs(successful_run_dirs, "validation_summary.csv")
    _write_rows_csv(batch_dir / "validation_summary.all.csv", validation_rows, VALIDATION_FIELDNAMES)
    if batch_config.stats:
        run_stats(batch_dir / "validation_summary.all.csv", batch_dir / "validation_reviewed.all.csv")
    else:
        _write_rows_csv(batch_dir / "validation_reviewed.all.csv", [], VALIDATION_REVIEWED_FIELDNAMES)
    return batch_dir
