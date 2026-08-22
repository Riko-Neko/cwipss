"""CSV field schemas and row normalization."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


RAW_CANDIDATE_SCHEMA_VERSION = 6


RAW_CANDIDATE_FIELDNAMES = [
    "schema_version",
    "run_id",
    "source_file",
    "candidate_id",
    "block_id",
    "window_id",
    "method",
    "wavelet",
    "time_agg",
    "channel",
    "freq_mhz",
    "t0_rec",
    "t1_rec",
    "dur_rec",
    "dur_s",
    "t_peak_rec",
    "t_peak_s",
    "period_rec",
    "period_s",
    "p0_rec",
    "p1_rec",
    "p_span_rec",
    "p_bins",
    "noise_sigma",
    "cpro_thr",
    "shape_mean",
    "shape_max",
    "pelt_z_mean",
    "pelt_z_max",
    "pelt_pen",
    "cprf_thr",
    "ridge_peak",
    "ridge_int",
    "band_conc",
    "band_persist",
    "local_contrast",
    "h2",
    "h3",
    "harm_n",
    "core_score",
    "score",
    "block_ch0",
    "block_ch1",
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


VALIDATION_REVIEWED_FIELDNAMES = VALIDATION_FIELDNAMES + [
    "p_value",
    "q_value",
    "global_q_value",
    "evidence_rank",
    "stats_status",
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
    "selected_channel_count",
    "valid_channel_count",
    "invalid_channel_count",
    "quality_status",
    "invalid_reason_counts_json",
    "invalid_ranges_json",
    "status",
    "error",
]


TIME_WINDOW_FIELDNAMES = [
    "schema_version",
    "run_id",
    "source_file",
    "block_id",
    "window_id",
    "method",
    "channel",
    "freq_mhz",
    "t0_rec",
    "t1_rec",
    "dur_rec",
    "noise_sigma",
    "cpro_thr",
    "shape_mean",
    "shape_max",
    "pelt_z_mean",
    "pelt_z_max",
    "pelt_pen",
    "accepted",
    "cprf_thr",
    "period_rec",
    "p0_rec",
    "p1_rec",
    "p_bins",
    "ridge_peak",
    "ridge_int",
    "band_conc",
    "band_persist",
    "local_contrast",
    "h2",
    "h3",
    "harm_n",
    "core_score",
    "score",
    "block_ch0",
    "block_ch1",
]


BATCH_MANIFEST_FIELDNAMES = [
    "batch_id",
    "run_id",
    "source_file",
    "run_dir",
    "status",
    "error",
    "duration_seconds",
    "candidate_count",
    "vetoed_candidate_count",
    "selected_channel_count",
    "valid_channel_count",
    "invalid_channel_count",
    "quality_status",
    "invalid_reason_counts_json",
    "invalid_ranges_json",
    "validation_count",
    "stats_count",
]


INJECTION_TRUTH_FIELDNAMES = [
    "injection_id",
    "signal_model",
    "period_records",
    "amplitude",
    "record_start",
    "record_stop",
    "duration_records",
    "channel_start",
    "channel_stop",
    "channel_center",
    "bandwidth_channels",
    "freq_start_mhz",
    "freq_stop_mhz",
    "freq_center_mhz",
    "duty_cycle",
    "phase",
    "drift_channels",
]


INJECTION_RESULT_FIELDNAMES = [
    "injection_id",
    "signal_model",
    "period_records",
    "amplitude",
    "detected_raw",
    "detected_after_veto",
    "validated",
    "matched_candidate_id",
    "failure_stage",
    "time_overlap_fraction",
    "freq_overlap_fraction",
    "period_error_fraction",
    "score",
    "candidate_status",
    "veto_flags",
    "p_value",
    "q_value",
    "global_q_value",
    "evidence_rank",
    "refined_period_records",
]


INJECTION_PERFORMANCE_FIELDNAMES = [
    "signal_model",
    "period_records",
    "amplitude",
    "injection_count",
    "detected_raw_count",
    "detected_after_veto_count",
    "validated_count",
    "detected_raw_rate",
    "detected_after_veto_rate",
    "validated_rate",
    "failure_stage_counts_json",
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
    period_rec = float(candidate.get("period_rec") or 0.0)
    dur_rec = int(candidate.get("dur_rec") or 0)
    t_peak_rec = int(candidate.get("t_peak_rec") or 0)

    candidate["schema_version"] = RAW_CANDIDATE_SCHEMA_VERSION
    candidate["run_id"] = run_id
    candidate["source_file"] = str(source_file)
    candidate["block_id"] = block_id
    candidate["period_s"] = float(period_rec * tsamp_seconds)
    candidate["dur_s"] = float(dur_rec * tsamp_seconds)
    candidate["t_peak_s"] = float(t_peak_rec * tsamp_seconds)
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
    channel_quality: Mapping[str, Any] | None = None,
    status: str = "complete",
    error: str = "",
) -> dict[str, Any]:
    records = int(source_info["records"])
    quality = channel_quality or {}
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
        "selected_channel_count": int(quality.get("selected_channel_count", 0)),
        "valid_channel_count": int(quality.get("valid_channel_count", 0)),
        "invalid_channel_count": int(quality.get("invalid_channel_count", 0)),
        "quality_status": str(quality.get("quality_status", "")),
        "invalid_reason_counts_json": json.dumps(
            quality.get("invalid_reason_counts", {}), separators=(",", ":")
        ),
        "invalid_ranges_json": json.dumps(
            quality.get("invalid_ranges", []), separators=(",", ":")
        ),
        "status": status,
        "error": error,
    }
