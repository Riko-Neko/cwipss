# CWT Period Search Output Schema

Schema version 1 covers the current per-channel CWT scalogram region candidate generator.

## Candidate Tables

`candidates_raw.csv` and `candidates_reviewed.csv` contain one row per detected
time-bounded region in one channel's `period x time` scalogram.

Important fields:

- `candidate_id`: run-local candidate id sorted by descending `integrated_score`.
- `block_id`: frequency-block id.
- `cwt_wavelet`: PyWavelets CWT wavelet.
- `time_aggregation`: method used only for overview maps.
- `detection_method`: current detector, usually `per_channel_scalogram_region`.
- `channel_index`: frequency-channel index within the current block.
- `record_start`, `record_stop`, `duration_records`: detected time span.
- `period_start_records`, `period_stop_records`: detected period-band span.
- `peak_period_records`: strongest period bin in the region.
- `peak_period_seconds`: `peak_period_records * tsamp_seconds`.
- `freq_start_mhz`, `freq_stop_mhz`: candidate channel/frequency coordinate.
- `peak_freq_mhz`: strongest channel/frequency coordinate.
- `peak_record`: strongest time location inside the detected region.
- `peak_score`: maximum robust scalogram-region score.
- `mean_score`: mean score over the detected region bounds.
- `integrated_score`: time-integrated region score normalized by duration.
- `candidate_status`, `veto_flags`, `veto_reason`: present in reviewed tables.

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
