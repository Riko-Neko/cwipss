from __future__ import annotations

import csv
import json
from pathlib import Path

from ce4_period_search import batch
from ce4_period_search.batch import BatchConfig, BatchJob, discover_input_files, ensure_run_ids, read_batch_manifest
from ce4_period_search.config import CWTSearchConfig
from ce4_period_search.models import (
    RAW_CANDIDATE_FIELDNAMES,
    REVIEWED_CANDIDATE_FIELDNAMES,
    VALIDATION_FIELDNAMES,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def test_discover_input_files_returns_project_relative_paths(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    first = data_dir / "a.2C"
    second = data_dir / "b.2C"
    first.write_text("")
    second.write_text("")
    (data_dir / "ignore.txt").write_text("")

    found = discover_input_files(data_dir, project_dir=tmp_path)

    assert found == ["data/a.2C", "data/b.2C"]


def test_read_manifest_and_run_id_assignment_preserves_overrides(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_csv(
        manifest,
        ["source_file", "f_start_mhz", "f_stop_mhz", "t_start", "t_stop"],
        [{"source_file": "data/a.2C", "f_start_mhz": "38.0", "f_stop_mhz": "39.0", "t_start": "10", "t_stop": "20"}],
    )

    jobs = ensure_run_ids(read_batch_manifest(manifest))

    assert jobs[0].run_id == "0001_a"
    assert jobs[0].f_start == 38.0
    assert jobs[0].f_stop == 39.0
    assert jobs[0].t_start == 10
    assert jobs[0].t_stop == 20


def test_run_batch_merges_outputs_and_recomputes_global_stats(tmp_path: Path, monkeypatch) -> None:
    def fake_run_cwt_search(config: CWTSearchConfig) -> Path:
        run_dir = Path(config.output_dir) / str(config.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(
            run_dir / "candidates_raw.csv",
            RAW_CANDIDATE_FIELDNAMES,
            [{"schema_version": 1, "run_id": config.run_id, "source_file": config.input, "candidate_id": 1, "peak_score": 5.0}],
        )
        _write_csv(
            run_dir / "candidates_reviewed.csv",
            REVIEWED_CANDIDATE_FIELDNAMES,
            [
                {
                    "schema_version": 1,
                    "run_id": config.run_id,
                    "source_file": config.input,
                    "candidate_id": 1,
                    "peak_score": 5.0,
                    "candidate_status": "needs_validation",
                }
            ],
        )
        (run_dir / "summary.json").write_text(
            json.dumps({"candidate_count": 1, "vetoed_candidate_count": 0}, ensure_ascii=True)
        )
        return run_dir

    def fake_validation(run_dir: Path, scan_config: CWTSearchConfig, project_dir: Path):
        pvalue = "0.01" if scan_config.run_id == "run_a" else "0.20"
        rows = [
            {
                "schema_version": 1,
                "run_id": scan_config.run_id,
                "source_file": scan_config.input,
                "candidate_id": 1,
                "candidate_status": "needs_validation",
                "validation_status": "evaluated",
                "shuffle_pvalue": pvalue,
                "observed_metric": "4.0",
                "fold_profile_snr": "4.0",
            }
        ]
        _write_csv(run_dir / "validation_summary.csv", VALIDATION_FIELDNAMES, rows)
        return rows

    monkeypatch.setattr(batch, "run_cwt_search", fake_run_cwt_search)
    monkeypatch.setattr(batch, "_run_validation_for_dir", fake_validation)

    batch_dir = batch.run_batch(
        jobs=[BatchJob(input="a.2C", run_id="run_a"), BatchJob(input="b.2C", run_id="run_b")],
        base_config=CWTSearchConfig(),
        batch_config=BatchConfig(batch_id="batch_test", output_dir=str(tmp_path), validate=True, stats=True),
        project_dir=tmp_path,
    )

    manifest_rows = list(csv.DictReader((batch_dir / "manifest.csv").open()))
    reviewed_rows = list(csv.DictReader((batch_dir / "validation_reviewed.all.csv").open()))

    assert [row["status"] for row in manifest_rows] == ["complete", "complete"]
    assert len(list(csv.DictReader((batch_dir / "candidates_reviewed.all.csv").open()))) == 2
    assert len(reviewed_rows) == 2
    assert reviewed_rows[0]["global_q_value"] == "0.02"
    assert reviewed_rows[1]["global_q_value"] == "0.2"
