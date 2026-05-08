from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


RAW_CANDIDATE_SCHEMA_VERSION = 1


RAW_CANDIDATE_FIELDNAMES = [
    "schema_version",
    "run_id",
    "source_file",
    "candidate_id",
    "block_id",
    "swt_level",
    "approx_scale_records",
    "approx_scale_seconds",
    "component_id",
    "area_pixels",
    "record_start",
    "record_stop",
    "duration_records",
    "duration_seconds",
    "freq_start_mhz",
    "freq_stop_mhz",
    "bandwidth_mhz",
    "peak_record",
    "peak_time_seconds",
    "peak_freq_mhz",
    "peak_score",
    "mean_score",
    "block_channel_start",
    "block_channel_stop",
]


REVIEWED_CANDIDATE_FIELDNAMES = RAW_CANDIDATE_FIELDNAMES + [
    "candidate_status",
    "veto_flags",
    "veto_reason",
    "veto_rule_count",
    "veto_details_json",
]


VALIDATION_FIELDNAMES = [
    "schema_version",
    "run_id",
    "source_file",
    "candidate_id",
    "candidate_status",
    "validation_status",
    "validation_notes",
    "validation_record_start",
    "validation_record_stop",
    "validation_duration_records",
    "validation_freq_start_mhz",
    "validation_freq_stop_mhz",
    "validation_channel_count",
    "approx_period_records",
    "period_min_records",
    "period_max_records",
    "refined_period_records",
    "refined_period_seconds",
    "acf_best_lag_records",
    "acf_peak",
    "acf_prominence",
    "periodogram_best_period_records",
    "periodogram_peak_power",
    "folding_best_period_records",
    "fold_profile_snr",
    "fold_bin_count",
    "observed_metric",
    "null_max_metric",
    "shuffle_trials",
    "shuffle_pvalue",
]


MANIFEST_FIELDNAMES = [
    "run_id",
    "source_file",
    "label_file",
    "records",
    "channels",
    "tsamp_seconds",
    "freq_min_mhz",
    "freq_max_mhz",
    "record_start",
    "record_stop",
    "f_start_mhz",
    "f_stop_mhz",
    "candidate_count",
    "status",
    "error",
]


def normalize_candidate_row(
    row: Mapping[str, Any],
    *,
    run_id: str,
    source_file: str | Path,
    block_id: str,
    tsamp_seconds: float,
) -> dict[str, Any]:
    """Attach stable metadata and derived physical units to one raw candidate."""
    candidate = dict(row)
    scale_records = int(candidate.get("approx_scale_records") or 0)
    duration_records = int(candidate.get("duration_records") or 0)
    peak_record = int(candidate.get("peak_record") or 0)

    candidate["schema_version"] = RAW_CANDIDATE_SCHEMA_VERSION
    candidate["run_id"] = run_id
    candidate["source_file"] = str(source_file)
    candidate["block_id"] = block_id
    candidate["approx_scale_seconds"] = float(scale_records * tsamp_seconds)
    candidate["duration_seconds"] = float(duration_records * tsamp_seconds)
    candidate["peak_time_seconds"] = float(peak_record * tsamp_seconds)
    return candidate


def make_manifest_row(
    *,
    run_id: str,
    source_info: Mapping[str, Any],
    record_start: int | None,
    record_stop: int | None,
    f_start_mhz: float | None,
    f_stop_mhz: float | None,
    candidate_count: int,
    status: str = "complete",
    error: str = "",
) -> dict[str, Any]:
    records = int(source_info["records"])
    return {
        "run_id": run_id,
        "source_file": source_info["filename"],
        "label_file": source_info.get("label") or "",
        "records": records,
        "channels": int(source_info["channels"]),
        "tsamp_seconds": float(source_info["tsamp_seconds"]),
        "freq_min_mhz": float(source_info["freq_min_mhz"]),
        "freq_max_mhz": float(source_info["freq_max_mhz"]),
        "record_start": 0 if record_start is None else int(record_start),
        "record_stop": records if record_stop is None else int(record_stop),
        "f_start_mhz": "" if f_start_mhz is None else float(f_start_mhz),
        "f_stop_mhz": "" if f_stop_mhz is None else float(f_stop_mhz),
        "candidate_count": int(candidate_count),
        "status": status,
        "error": error,
    }
