# CWT Period Search Output Schema

Schema version 1 covers the current CWT period-channel candidate generator.

## Candidate Tables

`candidates_raw.csv` and `candidates_reviewed.csv` contain one row per detected
period-channel component.

Important fields:

- `candidate_id`: run-local candidate id sorted by descending `peak_score`.
- `block_id`: frequency-block id.
- `cwt_wavelet`: PyWavelets CWT wavelet.
- `time_aggregation`: method used to collapse time before detection.
- `period_start_records`, `period_stop_records`: component period span.
- `peak_period_records`: strongest period bin in the component.
- `peak_period_seconds`: `peak_period_records * tsamp_seconds`.
- `freq_start_mhz`, `freq_stop_mhz`: component channel/frequency span.
- `peak_freq_mhz`: strongest channel/frequency coordinate.
- `peak_record`: strongest time location from the unaggregated CWT power cube.
- `peak_score`: local robust S/N on the period-channel map.
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
