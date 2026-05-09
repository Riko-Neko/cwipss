# CWT Period Search Output Schema

Schema version 1 covers the current CWT period-channel candidate generator.

## Candidate Tables

`candidates_raw.csv` and `candidates_reviewed.csv` contain one row per detected
channel-wise period-profile peak.

Important fields:

- `candidate_id`: run-local candidate id sorted by descending `peak_score`.
- `block_id`: frequency-block id.
- `cwt_wavelet`: PyWavelets CWT wavelet.
- `time_aggregation`: method used to collapse time before detection.
- `detection_method`: current detector, usually `channel_period_peak_dog`.
- `channel_index`: frequency-channel index within the current block.
- `period_start_records`, `period_stop_records`: 1D peak width span.
- `peak_period_records`: strongest period bin in the profile peak.
- `peak_period_seconds`: `peak_period_records * tsamp_seconds`.
- `freq_start_mhz`, `freq_stop_mhz`: candidate channel/frequency coordinate.
- `peak_freq_mhz`: strongest channel/frequency coordinate.
- `peak_record`: strongest time location from the unaggregated CWT power cube.
- `peak_score`: robust score after channel-wise period-profile DoG filtering.
- `peak_prominence`: 1D prominence of the period-profile peak.
- `peak_width_bins`: peak width in period-grid bins.
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
