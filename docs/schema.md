# SWT Period Search Output Schema

Schema version 1 covers the current single-file SWT period-candidate generator.
These tables describe candidates and run provenance only; they do not claim
validated periodic signals. CE-4 `.2C/.2CL` paths may appear in current runs
because CE-4 is the first application adapter.

## Run Products

Each run directory contains:

- `config.resolved.json`: structured config after CLI overrides.
- `manifest.csv`: one row per processed source file.
- `candidates_raw.csv`: raw SWT connected-component candidates.
- `candidates_reviewed.csv`: raw candidates plus veto labels and evidence.
- `candidates.csv`: compatibility alias for `candidates_raw.csv`.
- `validation_summary.csv`: optional original time-series validation evidence.
- `validation_reviewed.csv`: optional validation evidence with BH/FDR statistics.
- `validation/candidate_*.json`: optional per-candidate validation details.
- `summary.json`: compact run summary and top candidates.
- `report.md`: optional Markdown review report.

Batch runs write a separate batch directory containing:

- `batch_config.resolved.json`: batch settings, base scan config, and jobs.
- `manifest.csv`: one row per attempted source file.
- `files/<run_id>/`: isolated single-file run products.
- `candidates_raw.all.csv`: merged raw candidates.
- `candidates_reviewed.all.csv`: merged veto-reviewed candidates.
- `validation_summary.all.csv`: merged validation evidence.
- `validation_reviewed.all.csv`: merged validation statistics with global correction.
- `report.md`: optional Markdown batch review report.

## `manifest.csv`

Single-file run manifests use:

| Field | Meaning |
| --- | --- |
| `run_id` | Stable run directory id. |
| `source_file` | Input data file path for the active application adapter. |
| `label_file` | Auxiliary metadata/label file path when the adapter provides one. |
| `records` | Total records in the input file. |
| `channels` | Total frequency channels. |
| `tsamp_seconds` | Inferred sample spacing per record. |
| `freq_min_mhz`, `freq_max_mhz` | Full inferred channel coordinate range in MHz for the current CE-4 adapter. |
| `record_start`, `record_stop` | Actual scanned record range. |
| `f_start_mhz`, `f_stop_mhz` | Requested frequency bounds, blank when full range. |
| `candidate_count` | Number of retained raw candidates. |
| `status` | Processing status for this source file. |
| `error` | Error message for failed rows. |

Batch-level manifests use:

| Field | Meaning |
| --- | --- |
| `batch_id` | Batch output id. |
| `run_id` | Per-file run id. |
| `source_file` | Input data file path. |
| `run_dir` | Isolated per-file run directory. |
| `status` | `complete` or `error`. |
| `error` | Error message for failed files. |
| `duration_seconds` | Wall-clock runtime for the file. |
| `candidate_count` | Raw candidates from the per-file summary. |
| `vetoed_candidate_count` | Vetoed candidates from the per-file summary. |
| `validation_count` | Validation rows produced for the file. |
| `stats_count` | Statistics rows produced for the file. |

## `candidates_raw.csv`

| Field | Meaning |
| --- | --- |
| `schema_version` | Candidate table schema version. |
| `run_id` | Run that produced the row. |
| `source_file` | Input data file path for the active application adapter. |
| `candidate_id` | Rank after sorting by `peak_score` descending. |
| `block_id` | Frequency block id in the scan. |
| `swt_level` | SWT level number reported by PyWavelets. |
| `approx_scale_records` | Approximate SWT scale in records. |
| `approx_scale_seconds` | Approximate SWT scale in seconds. |
| `component_id` | Connected-component label inside the score map. |
| `area_pixels` | Component area in time-frequency pixels. |
| `record_start`, `record_stop` | Half-open record span. |
| `duration_records`, `duration_seconds` | Component duration. |
| `freq_start_mhz`, `freq_stop_mhz` | Component channel coordinate span in MHz for the current CE-4 adapter. |
| `bandwidth_mhz` | Component channel-coordinate span. |
| `peak_record`, `peak_time_seconds` | Peak score location in time. |
| `peak_freq_mhz` | Peak score frequency. |
| `peak_score`, `mean_score` | Local robust S/N statistics. |
| `block_channel_start`, `block_channel_stop` | Half-open source channel span for the scanned block. |

Future validation stages should write separate tables with refined periods,
null-test p-values, and multiple-testing correction fields.

## `candidates_reviewed.csv`

This table starts with all `candidates_raw.csv` fields and appends:

| Field | Meaning |
| --- | --- |
| `candidate_status` | `vetoed` or `needs_validation`. |
| `veto_flags` | Pipe-delimited rule names, blank when no veto fired. |
| `veto_reason` | Human-readable reasons for fired rules. |
| `veto_rule_count` | Number of fired veto rules. |
| `veto_details_json` | Per-rule metrics and thresholds as JSON. |

`needs_validation` means the candidate survived this first rule-based veto
layer. It does not mean the candidate is a validated periodic signal.

## `validation_summary.csv`

This table is written by `scripts/run_validation.py`.

| Field | Meaning |
| --- | --- |
| `schema_version` | Validation table schema version. |
| `run_id`, `source_file`, `candidate_id` | Candidate provenance. |
| `candidate_status` | Status from `candidates_reviewed.csv`. |
| `validation_status` | `evaluated`, `insufficient_data`, or `error`. |
| `validation_notes` | Short processing note or error text. |
| `validation_record_start`, `validation_record_stop` | Half-open record window used for validation. |
| `validation_duration_records` | Validation window size. |
| `validation_freq_start_mhz`, `validation_freq_stop_mhz` | Channel coordinate span extracted from the original source data. |
| `validation_channel_count` | Number of channels aggregated. |
| `approx_period_records` | SWT-scale period seed. |
| `period_min_records`, `period_max_records` | Integer-record search range. |
| `refined_period_records`, `refined_period_seconds` | Current best folding period. |
| `acf_best_lag_records`, `acf_peak`, `acf_prominence` | Autocorrelation evidence near the search range. |
| `periodogram_best_period_records`, `periodogram_peak_power` | FFT periodogram evidence near the search range. |
| `folding_best_period_records`, `fold_profile_snr`, `fold_bin_count` | Folding evidence. |
| `observed_metric`, `null_max_metric`, `shuffle_trials`, `shuffle_pvalue` | Shuffle/null-test evidence for the folding metric. |

Validation metrics are evidence for review, not a final signal claim. A later
statistics stage should apply multiple-testing correction across all evaluated
candidates.

## `validation_reviewed.csv`

This table is written by `scripts/run_stats.py`. It starts with all
`validation_summary.csv` fields and appends:

| Field | Meaning |
| --- | --- |
| `p_value` | Candidate-local p-value copied from `shuffle_pvalue`. |
| `q_value` | Benjamini-Hochberg adjusted p-value within each `run_id`. |
| `global_q_value` | Benjamini-Hochberg adjusted p-value across all rows in the input file. |
| `evidence_rank` | Deterministic review order among rows with valid p-values. |
| `stats_status` | `evaluated` for rows with valid p-values, otherwise `missing_pvalue`. |

For a single-run input, `global_q_value` is usually the same as the run-level
correction unless the input file contains multiple `run_id` groups. In batch
mode, global correction should be applied to the merged validation table.
