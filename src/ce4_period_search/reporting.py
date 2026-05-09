from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _float(value: object, default: float = math.nan) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: object, digits: int = 4) -> str:
    number = _float(value)
    if math.isfinite(number):
        return f"{number:.{digits}g}"
    text = "" if value is None else str(value)
    return text if text else "-"


def _short_path(value: object) -> str:
    text = str(value or "")
    return Path(text).name if text else "-"


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) if str(item) else "-" for item in row) + " |")
    return "\n".join(lines)


def veto_distribution(rows: list[dict[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        flags = str(row.get("veto_flags", "")).strip()
        if not flags:
            counts["none"] += 1
            continue
        for flag in flags.split("|"):
            counts[flag or "none"] += 1
    return counts


def top_candidates(rows: list[dict[str, str]], limit: int = 10) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: _float(row.get("peak_score"), -math.inf), reverse=True)[:limit]


def top_validation_rows(rows: list[dict[str, str]], limit: int = 10) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            _float(row.get("evidence_rank"), math.inf),
            _float(row.get("global_q_value"), math.inf),
            _float(row.get("q_value"), math.inf),
            _float(row.get("p_value"), math.inf),
        ),
    )[:limit]


def _candidate_table(rows: list[dict[str, str]], limit: int) -> str:
    table_rows = []
    for row in top_candidates(rows, limit=limit):
        table_rows.append(
            [
                row.get("candidate_id", "-"),
                _short_path(row.get("source_file")),
                row.get("candidate_status", "-"),
                _fmt(row.get("peak_score")),
                _fmt(row.get("peak_period_records")),
                _fmt(row.get("peak_record")),
                _fmt(row.get("peak_freq_mhz")),
                row.get("veto_flags", "") or "-",
            ]
        )
    return _markdown_table(
        ["candidate_id", "source", "status", "peak_score", "period_rec", "peak_record", "peak_mhz", "veto_flags"],
        table_rows,
    )


def _validation_table(rows: list[dict[str, str]], limit: int) -> str:
    table_rows = []
    for row in top_validation_rows(rows, limit=limit):
        table_rows.append(
            [
                row.get("evidence_rank", "-"),
                row.get("candidate_id", "-"),
                _short_path(row.get("source_file")),
                row.get("validation_status", "-"),
                _fmt(row.get("refined_period_records")),
                _fmt(row.get("p_value")),
                _fmt(row.get("q_value")),
                _fmt(row.get("global_q_value")),
                _fmt(row.get("observed_metric")),
            ]
        )
    return _markdown_table(
        ["rank", "candidate_id", "source", "validation", "period_rec", "p", "q", "global_q", "metric"],
        table_rows,
    )


def _injection_performance_table(rows: list[dict[str, str]], limit: int) -> str:
    table_rows = []
    for row in rows[:limit]:
        table_rows.append(
            [
                row.get("signal_model", "-"),
                _fmt(row.get("period_records")),
                _fmt(row.get("amplitude")),
                row.get("injection_count", "0"),
                _fmt(row.get("detected_raw_rate")),
                _fmt(row.get("detected_after_veto_rate")),
                _fmt(row.get("validated_rate")),
                row.get("failure_stage_counts_json", "-"),
            ]
        )
    return _markdown_table(
        ["model", "period_rec", "amplitude", "n", "raw_rate", "after_veto_rate", "validated_rate", "failures"],
        table_rows,
    )


def _injection_results_table(rows: list[dict[str, str]], limit: int) -> str:
    table_rows = []
    for row in rows[:limit]:
        table_rows.append(
            [
                row.get("injection_id", "-"),
                row.get("signal_model", "-"),
                _fmt(row.get("period_records")),
                _fmt(row.get("amplitude")),
                row.get("failure_stage", "-"),
                row.get("matched_candidate_id", "-"),
                _fmt(row.get("period_error_fraction")),
                _fmt(row.get("peak_score")),
            ]
        )
    return _markdown_table(
        ["injection_id", "model", "period_rec", "amplitude", "stage", "candidate_id", "period_error", "peak_score"],
        table_rows,
    )


def _veto_table(rows: list[dict[str, str]]) -> str:
    counts = veto_distribution(rows)
    table_rows = [[flag, count] for flag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    return _markdown_table(["veto_flag", "count"], table_rows)


def _batch_summary_table(manifest_rows: list[dict[str, str]]) -> str:
    table_rows = []
    for row in manifest_rows:
        table_rows.append(
            [
                row.get("run_id", "-"),
                _short_path(row.get("source_file")),
                row.get("status", "-"),
                row.get("candidate_count", "0"),
                row.get("vetoed_candidate_count", "0"),
                row.get("validation_count", "0"),
                row.get("stats_count", "0"),
            ]
        )
    return _markdown_table(
        ["run_id", "source", "status", "candidates", "vetoed", "validated", "stats"],
        table_rows,
    )


def detect_report_kind(run_dir: Path) -> str:
    return "batch" if (run_dir / "batch_config.resolved.json").exists() else "single"


def generate_report_markdown(run_dir: str | Path, top_n: int = 10) -> str:
    run_dir = Path(run_dir)
    kind = detect_report_kind(run_dir)
    if kind == "batch":
        return generate_batch_report(run_dir, top_n=top_n)
    return generate_single_run_report(run_dir, top_n=top_n)


def generate_single_run_report(run_dir: str | Path, top_n: int = 10) -> str:
    run_dir = Path(run_dir)
    summary = read_json(run_dir / "summary.json")
    injection_summary = read_json(run_dir / "injection_summary.json")
    run_summary = summary or injection_summary
    candidates = read_csv_rows(run_dir / "candidates_reviewed.csv")
    validation = read_csv_rows(run_dir / "validation_reviewed.csv")
    injection_performance = read_csv_rows(run_dir / "injection_performance.csv")
    injection_results = read_csv_rows(run_dir / "injection_results.csv")
    runtime = summary.get("runtime", injection_summary.get("runtime", {}))
    visualization_index = run_dir / "visualization" / "index.md"
    lines = [
        f"# CWT Period Search Report: {run_dir.name}",
        "",
        "This report summarizes candidate evidence for review. It does not claim a confirmed periodic signal.",
        "",
        "## Run Summary",
        "",
        _markdown_table(
            ["metric", "value"],
            [
                ["run_id", run_summary.get("run_id", run_dir.name)],
                ["source", _short_path(summary.get("source", {}).get("filename", run_summary.get("source", "")))],
                ["candidate_count", run_summary.get("candidate_count", len(candidates))],
                ["vetoed_candidate_count", run_summary.get("vetoed_candidate_count", "-")],
                ["reviewed_candidate_count", run_summary.get("reviewed_candidate_count", len(candidates))],
                ["python", runtime.get("python", "-")],
                ["numpy", runtime.get("numpy", "-")],
                ["pywavelets", runtime.get("pywavelets", "-")],
                ["scipy", runtime.get("scipy", "-")],
            ],
        ),
        "",
        "## Veto Distribution",
        "",
        _veto_table(candidates),
        "",
    ]
    if visualization_index.exists():
        lines.extend(
            [
                "## Stage Visualization",
                "",
                "Staged diagnostic figures: [visualization/index.md](visualization/index.md)",
                "",
            ]
        )
    lines.extend(
        [
            f"## Top {top_n} Candidates By CWT Score",
            "",
            _candidate_table(candidates, limit=top_n),
            "",
            f"## Top {top_n} Validation Evidence Rows",
            "",
            _validation_table(validation, limit=top_n),
            "",
        ]
    )
    if injection_summary or injection_performance or injection_results:
        lines.extend(
            [
                "## Injection Benchmark",
                "",
                _markdown_table(
                    ["metric", "value"],
                    [
                        ["injection_count", injection_summary.get("injection_count", len(injection_results))],
                        ["detected_raw_rate", _fmt(injection_summary.get("detected_raw_rate"))],
                        ["detected_after_veto_rate", _fmt(injection_summary.get("detected_after_veto_rate"))],
                        ["validated_rate", _fmt(injection_summary.get("validated_rate"))],
                        [
                            "failure_stage_counts",
                            json.dumps(
                                injection_summary.get("failure_stage_counts", {}),
                                sort_keys=True,
                                ensure_ascii=True,
                            ),
                        ],
                    ],
                ),
                "",
                f"## Top {top_n} Injection Performance Rows",
                "",
                _injection_performance_table(injection_performance, limit=top_n),
                "",
                f"## Top {top_n} Injection Results",
                "",
                _injection_results_table(injection_results, limit=top_n),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation Note",
            "",
            "Low p-values, q-values, or high folding metrics are review evidence only. Candidate periods require domain review and additional controls before any signal claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_batch_report(run_dir: str | Path, top_n: int = 10) -> str:
    run_dir = Path(run_dir)
    batch_config = read_json(run_dir / "batch_config.resolved.json")
    manifest = read_csv_rows(run_dir / "manifest.csv")
    candidates = read_csv_rows(run_dir / "candidates_reviewed.all.csv")
    validation = read_csv_rows(run_dir / "validation_reviewed.all.csv")
    complete_count = sum(1 for row in manifest if row.get("status") == "complete")
    error_count = sum(1 for row in manifest if row.get("status") == "error")
    lines = [
        f"# CWT Period Search Batch Report: {run_dir.name}",
        "",
        "This report summarizes candidate evidence for review. It does not claim a confirmed periodic signal.",
        "",
        "## Batch Summary",
        "",
        _markdown_table(
            ["metric", "value"],
            [
                ["batch_id", batch_config.get("batch", {}).get("batch_id", run_dir.name)],
                ["job_count", len(manifest)],
                ["complete_count", complete_count],
                ["error_count", error_count],
                ["candidate_count", len(candidates)],
                ["validation_rows", len(validation)],
            ],
        ),
        "",
        "## File Summary",
        "",
        _batch_summary_table(manifest),
        "",
        "## Veto Distribution",
        "",
        _veto_table(candidates),
        "",
        f"## Top {top_n} Candidates By CWT Score",
        "",
        _candidate_table(candidates, limit=top_n),
        "",
        f"## Top {top_n} Validation Evidence Rows",
        "",
        _validation_table(validation, limit=top_n),
        "",
        "## Interpretation Note",
        "",
        "Batch-level global q-values are corrected over the merged validation table. They remain review evidence, not a signal claim.",
    ]
    return "\n".join(lines) + "\n"


def write_report(run_dir: str | Path, output_path: str | Path | None = None, top_n: int = 10) -> Path:
    run_dir = Path(run_dir)
    path = Path(output_path) if output_path is not None else run_dir / "report.md"
    path.write_text(generate_report_markdown(run_dir, top_n=top_n))
    return path
