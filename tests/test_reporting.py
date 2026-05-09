from __future__ import annotations

import csv
import json
from pathlib import Path

from ce4_period_search.reporting import generate_report_markdown, veto_distribution, write_report


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def test_veto_distribution_counts_pipe_delimited_flags() -> None:
    counts = veto_distribution(
        [
            {"veto_flags": "freq_edge|burst_train"},
            {"veto_flags": "freq_edge"},
            {"veto_flags": ""},
        ]
    )

    assert counts["freq_edge"] == 2
    assert counts["burst_train"] == 1
    assert counts["none"] == 1


def test_generate_single_run_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "run_a",
                "candidate_count": 1,
                "vetoed_candidate_count": 0,
                "reviewed_candidate_count": 1,
                "source": {"filename": "data/example.2C"},
                "runtime": {"python": "3.12", "numpy": "2", "pywavelets": "1", "scipy": "1"},
            }
        )
    )
    _write_csv(
        run_dir / "candidates_reviewed.csv",
        ["candidate_id", "source_file", "candidate_status", "peak_score", "peak_period_records", "peak_record", "peak_freq_mhz", "veto_flags"],
        [
            {
                "candidate_id": "1",
                "source_file": "data/example.2C",
                "candidate_status": "needs_validation",
                "peak_score": "9.0",
                "peak_period_records": "8",
                "peak_record": "100",
                "peak_freq_mhz": "38.1",
                "veto_flags": "",
            }
        ],
    )
    _write_csv(
        run_dir / "validation_reviewed.csv",
        ["evidence_rank", "candidate_id", "source_file", "validation_status", "refined_period_records", "p_value", "q_value", "global_q_value", "observed_metric"],
        [
            {
                "evidence_rank": "1",
                "candidate_id": "1",
                "source_file": "data/example.2C",
                "validation_status": "evaluated",
                "refined_period_records": "8",
                "p_value": "0.01",
                "q_value": "0.01",
                "global_q_value": "0.01",
                "observed_metric": "5",
            }
        ],
    )

    markdown = generate_report_markdown(run_dir, top_n=5)

    assert "# CWT Period Search Report: run_a" in markdown
    assert "## Veto Distribution" in markdown
    assert "example.2C" in markdown
    assert "does not claim a confirmed periodic signal" in markdown


def test_generate_batch_report_and_write_file(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch_a"
    batch_dir.mkdir()
    (batch_dir / "batch_config.resolved.json").write_text(json.dumps({"batch": {"batch_id": "batch_a"}}))
    _write_csv(
        batch_dir / "manifest.csv",
        ["run_id", "source_file", "status", "candidate_count", "vetoed_candidate_count", "validation_count", "stats_count"],
        [{"run_id": "run_a", "source_file": "data/a.2C", "status": "complete", "candidate_count": "0"}],
    )
    _write_csv(batch_dir / "candidates_reviewed.all.csv", ["veto_flags"], [])
    _write_csv(batch_dir / "validation_reviewed.all.csv", ["evidence_rank"], [])

    report_path = write_report(batch_dir, top_n=3)

    assert report_path == batch_dir / "report.md"
    text = report_path.read_text()
    assert "# CWT Period Search Batch Report: batch_a" in text
    assert "## File Summary" in text
    assert "run_a" in text


def test_generate_report_includes_injection_benchmark_section(tmp_path: Path) -> None:
    run_dir = tmp_path / "injection_run"
    run_dir.mkdir()
    (run_dir / "injection_summary.json").write_text(
        json.dumps(
            {
                "run_id": "inj_run",
                "source": "synthetic",
                "injection_count": 1,
                "candidate_count": 1,
                "reviewed_candidate_count": 1,
                "detected_raw_rate": 1.0,
                "detected_after_veto_rate": 1.0,
                "validated_rate": 0.0,
                "failure_stage_counts": {"period_mismatch": 1},
            }
        )
    )
    _write_csv(
        run_dir / "injection_performance.csv",
        [
            "signal_model",
            "period_records",
            "amplitude",
            "injection_count",
            "detected_raw_rate",
            "detected_after_veto_rate",
            "validated_rate",
            "failure_stage_counts_json",
        ],
        [
            {
                "signal_model": "pulsed_periodic",
                "period_records": "8",
                "amplitude": "5",
                "injection_count": "1",
                "detected_raw_rate": "1",
                "detected_after_veto_rate": "1",
                "validated_rate": "0",
                "failure_stage_counts_json": '{"period_mismatch": 1}',
            }
        ],
    )
    _write_csv(
        run_dir / "injection_results.csv",
        [
            "injection_id",
            "signal_model",
            "period_records",
            "amplitude",
            "failure_stage",
            "matched_candidate_id",
            "period_error_fraction",
            "peak_score",
        ],
        [
            {
                "injection_id": "inj_0001",
                "signal_model": "pulsed_periodic",
                "period_records": "8",
                "amplitude": "5",
                "failure_stage": "period_mismatch",
            }
        ],
    )

    markdown = generate_report_markdown(run_dir, top_n=5)

    assert "## Injection Benchmark" in markdown
    assert "validated_rate" in markdown
    assert "inj_0001" in markdown
