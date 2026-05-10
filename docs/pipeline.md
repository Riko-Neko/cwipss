# CWT Period Search Pipeline

This project generates period candidates from `time x channel` data using a
per-channel continuous wavelet transform.

## Core Flow

```text
time x channel data
  -> per-channel CWT over explicit period grid
  -> period x time x channel power cube
  -> candidate period-domain filter
  -> single-channel low-fraction CWT noise floor
  -> signed relative excess power
  -> trimmed period-axis activity curve
  -> PELT time windows per channel
  -> windowed period-profile peaks
  -> veto, validation, statistics, report
```

The overview projection is directly interpretable:

- x-axis: observation channel or MHz coordinate;
- y-axis: period in records or seconds;
- value: CWT overview or recorded single-channel candidates.

## Candidate Meaning

A peak in a PELT-windowed period profile is only a candidate. It is not a
confirmed periodic signal. Final review still requires original time-series
validation, null tests, RFI veto, and multiple-testing correction.

## Candidate Sensitivity

The default detector is set for low sensitivity and higher review purity:

- candidate period domain `10..200` records;
- low-floor noise fraction `0.20`;
- PELT minimum segment/window size `256` records;
- retained candidates capped at `max_candidates_per_block=50`;
- validation capped at `validation.max_candidates=25`.

Lower PELT/profile thresholds are debug/high-recall settings. They are useful
for inspecting weak period responses, but they can create many visually
plausible candidates and should not be used as the default review configuration.

## Feasible Period Domain

The default candidate domain rejects periods below 10 records and above 200
records. This is a detection-domain filter, not a CWT-grid limit.

For CE-4 files with labels, one record is about one second. In the current
4096-record low-frequency review windows, the assumed minimum real signal span
is at least half the window:

```text
N_window = 4096 records
L_min = 0.5 * N_window = 2048 records
N_cycles_min = 10
P_max = floor(L_min / N_cycles_min) ~= 204 records -> 200 records
```

The lower cutoff is not the mathematical Nyquist limit. Nyquist only gives
`P > 2` records. For this detector, periods below roughly 10 records have too
few samples per cycle for stable CWT/folding evidence and are where persistent
instrument-like stripes dominate. The practical domain is therefore:

```text
P_min = 10 records
P_max = floor(min_expected_signal_span_records / 10)
```

For other scan lengths or signal-duration assumptions, recompute `P_max` with
that formula and override `candidate_period_min_records` /
`candidate_period_max_records`.

## Progress Reporting

CLI scans enable a CWT channel-progress tqdm by default. The pipeline still
computes CWT in frequency blocks for performance; the progress unit is the
number of selected frequency channels completed. Use `--no-progress` to silence
the bar or `--progress-leave` to keep it in terminal logs.

## Time Aggregation

CWT first produces `period x time x channel`. Detection uses this first-hand
cube directly. The default aggregation is `p95` and is now used only for overview
maps, not candidate generation.
Supported aggregation methods are `max`, `mean`, `median`, `pNN`, and
`percentile`.

Before aggregation, visualization can write representative-channel
`period x time` scalograms, trusted-period relative-excess maps, PELT activity
curves, and windowed period profiles. These are the required middleware views
for inspecting whether a period response is persistent, burst-like, or
contaminated.

## Outputs

Each run writes:

- `candidates_raw.csv`
- `candidates_reviewed.csv`
- `time_windows.csv`
- `manifest.csv`
- `summary.json`
- optional `visualization/index.md`

Candidate rows include `peak_period_records`, `period_start_records`,
`period_stop_records`, `peak_freq_mhz`, and `peak_record`.
