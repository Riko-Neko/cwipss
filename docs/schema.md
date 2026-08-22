# Cwipss Output Schema

Schema version 6 defines single-output CPRO -> PELT -> CPRF candidate and
time-window fields.

Canonical field lists and row normalization are defined in
`cwipss.data.schemas`.

## Candidate Tables

`candidates_raw.csv` and `candidates_reviewed.csv` contain one row per
CPRF-accepted period family from a single channel's PELT window. CPRF is the
only production period-family method and emits at most one row per window.

Important fields:

- `candidate_id`: run-local id sorted by descending `score`.
- `method`, `wavelet`, `time_agg`: method and transform provenance.
- `window_id`, `block_id`, `channel`, `freq_mhz`: source PELT window, absolute source-channel index, and frequency.
- `t0_rec`, `t1_rec`, `dur_rec`: half-open PELT time span `[t0_rec, t1_rec)`.
- `t_peak_rec`: maximum continuous CPRO shape activity location within that span.
- `period_rec`, `p0_rec`, `p1_rec`, `p_span_rec`, `p_bins`: selected period,
  ridge-band bounds, period-coordinate span, and grid-bin width.
- `period_s`, `dur_s`, `t_peak_s`: physical-time conversions using `tsamp_seconds`.
- `noise_sigma`: robust raw-series noise estimate from first differences.
- `cpro_thr`: absolute CWT-power threshold used by CPRO.
- `shape_mean`, `shape_max`: continuous CPRO shape evidence used by PELT inside
  the window; these are not absolute signal-strength statistics.
- `pelt_z_mean`, `pelt_z_max`: robust-standardized CPRO shape evidence inside the segment.
- `pelt_pen`: configured native PELT penalty.
- `cprf_thr`: independent absolute-power normalization threshold used by CPRF.
- `ridge_peak`: CPRF profile peak excess above its local period background.
- `ridge_int`: width-normalized integrated profile excess.
- `band_conc`: fraction of total CPRF profile mass in the selected ridge band.
- `band_persist`: strength-weighted time occupancy of that ridge band.
- `local_contrast`: ridge-band excess relative to its local period background.
- `h2`, `h3`, `harm_n`: diagnostic second/third harmonic response and supported-harmonic count.
- `core_score`: multiplicative ridge score before harmonic weighting.
- `score`: final CPRF ordering statistic. It is not a probability, p-value, or S/N.
- `block_ch0`, `block_ch1`: half-open source block channel range.
- `candidate_status`, `veto_flags`, `veto_reason`: present in reviewed tables.

The CPRF diagnostics preserve their mathematical meaning:

```text
ridge_int = sum(max(profile_band - local_background, 0)) / sqrt(p_bins)
band_conc = sum(profile_band) / sum(profile)
band_persist = sum(time_occupancy * profile_band) / sum(profile_band)
local_contrast = (mean(profile_band) - local_background) / max(abs(local_background), 1)
core_score = ridge_peak^0.40 * ridge_int^0.30 * band_conc^0.15 * band_persist^0.15
score = core_score * (1 + 0.5 * harmonic_weight * (h2 + h3))
```

`ridge_peak`, `ridge_int`, and the profile are dimensionless CPRF-threshold
units. `band_conc`, `band_persist`, `h2`, and `h3` are bounded ratios. Neither
`core_score` nor `score` is calibrated as a probability or significance.

## Time Windows

`time_windows.csv` records PELT windows and their CPRF decision. CPRF-rejected
windows remain present with `accepted=0`; only accepted windows feed
candidate generation. Important fields:

- `window_id`: channel-local window id.
- `method`, `channel`, `freq_mhz`: source method and channel.
- `t0_rec`, `t1_rec`, `dur_rec`: PELT window span.
- `cpro_thr`, `shape_*`, `pelt_*`: the same stage-specific evidence as candidate rows.
- `accepted`: CPRF gate decision.
- `period_rec`, `p0_rec`, `p1_rec`, `p_bins`: best CPRF hypothesis even when rejected.
- `ridge_*`, `band_*`, `local_contrast`, `h2`, `h3`, `harm_n`, `core_score`,
  `score`: complete CPRF diagnostics for threshold analysis.

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
