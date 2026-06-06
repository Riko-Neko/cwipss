# Cwipss Output Schema

Schema version 1 covers the current single-channel low-floor,
structure-gated PELT/profile candidate generator.

## Candidate Tables

`candidates_raw.csv` and `candidates_reviewed.csv` contain one row per period
peak detected from a single channel's PELT-windowed CWT period profile.
By default this is one period-family row per PELT window. Extra profile peaks
inside the same window are considered response substructure unless
`profile_max_peaks_per_window` is explicitly increased for high-recall
exploration.

Important fields:

- `candidate_id`: run-local candidate id sorted by descending `integrated_score`.
- `block_id`: frequency-block id.
- `cwt_wavelet`: PyWavelets CWT wavelet.
- `time_aggregation`: method used only for overview maps.
- `detection_method`: current detector, usually `single_channel_lowfloor_pelt_profile`.
- `window_id`: source PELT time-window id.
- `channel_index`: frequency-channel index within the current block.
- `record_start`, `record_stop`, `duration_records`: detected PELT window span.
- `period_start_records`, `period_stop_records`: local period-profile peak
  support, not the full Sa-like response envelope.
- `peak_period_records`: strongest period-profile peak.
- `peak_period_seconds`: `peak_period_records * tsamp_seconds`.
- `freq_start_mhz`, `freq_stop_mhz`: candidate channel/frequency coordinate.
- `peak_freq_mhz`: strongest channel/frequency coordinate.
- `peak_record`: strongest activity location inside the detected window.
- `peak_score`: windowed period-profile score.
- `mean_score`: mean standardized activity in the source window.
- `integrated_score`: same ranking score as `peak_score`.
- `activity_raw_mean`, `activity_raw_max`: source-window structured activity
  before robust standardization.
- `noise_floor`: single-channel low-20% CWT power floor used before
  structure gating.
- `period_peak_prominence`: period-profile peak prominence.
- `candidate_status`, `veto_flags`, `veto_reason`: present in reviewed tables.

## Time Windows

`time_windows.csv` records the PELT windows that feed period-profile candidate
generation. Important fields:

- `window_id`: channel-local window id.
- `detection_method`: usually `single_channel_lowfloor_pelt`.
- `channel_index`, `freq_mhz`: source channel.
- `record_start`, `record_stop`, `duration_records`: window span.
- `activity_mean`, `activity_max`: standardized activity statistics after
  period-axis compression of the structure-gated CWT map.
- `activity_raw_mean`, `activity_raw_max`: same window measured before robust
  standardization; the default raw mean floor is `25.0`.
- `noise_floor`: low-fraction CWT power floor used for this channel.
- `pelt_penalty`, `pelt_cost`: segmentation diagnostics.

## Validation Tables

`validation_summary.csv` records original-time-series checks around each
candidate period. `validation_reviewed.csv` adds:

- `p_value`: shuffle/null p-value.
- `q_value`: Benjamini-Hochberg q-value within a run.
- `global_q_value`: Benjamini-Hochberg q-value over the merged table.
- `evidence_rank`: deterministic review order.

## Injection Tables

`injection_truth.csv` records controlled injected signals.
`injection_results.csv` records whether each injection was detected, survived
veto, and validated.
`injection_performance.csv` aggregates recovery rates by signal model, period,
and amplitude.

## Batch Tables

Batch runs write merged tables under the batch directory:

- `manifest.csv`: one row per source file run.
- `candidates_raw.all.csv` and `candidates_reviewed.all.csv`: merged candidate
  tables.
- `time_windows.all.csv`: merged PELT time-window table.
- `validation_summary.all.csv` and `validation_reviewed.all.csv`: merged
  validation/statistics tables.
