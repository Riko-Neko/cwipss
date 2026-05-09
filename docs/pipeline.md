# CWT Period Search Pipeline

This project generates period candidates from `time x channel` data using a
per-channel continuous wavelet transform.

## Core Flow

```text
time x channel data
  -> per-channel CWT over explicit period grid
  -> period x time x channel power cube
  -> representative period x time scalograms for visual review
  -> time aggregation
  -> period x channel response map
  -> local robust S/N
  -> connected-component period candidates
  -> veto, validation, statistics, report
```

The candidate map is directly interpretable:

- x-axis: observation channel or MHz coordinate;
- y-axis: period in records or seconds;
- value: aggregated CWT power or local robust S/N.

## Candidate Meaning

A connected component in the period-channel map is only a candidate. It is not a
confirmed periodic signal. Final review still requires original time-series
validation, null tests, RFI veto, and multiple-testing correction.

## Candidate Sensitivity

The default detector is set for low sensitivity and higher review purity:

- local robust S/N `threshold=6.0`;
- connected-component area `min_pixels=6`;
- retained components capped at `max_candidates_per_block=50`;
- validation capped at `validation.max_candidates=25`.

Lower thresholds such as `1.4` and `min_pixels=1` are debug/high-recall
settings. They are useful for inspecting the response map, but they can create
many visually plausible candidates and should not be used as the default review
configuration.

## Progress Reporting

CLI scans enable a CWT channel-progress tqdm by default. The pipeline still
computes CWT in frequency blocks for performance; the progress unit is the
number of selected frequency channels completed. Use `--no-progress` to silence
the bar or `--progress-leave` to keep it in terminal logs.

## Time Aggregation

CWT first produces `period x time x channel`. The default aggregation is `p95`,
which is more stable than max while still preserving localized strong responses.
Supported aggregation methods are `max`, `mean`, `median`, `pNN`, and
`percentile`.

Before aggregation, visualization can write representative-channel
`period x time` scalograms. This is the required middleware view for inspecting
whether a period response is persistent, burst-like, or contaminated.

## Outputs

Each run writes:

- `candidates_raw.csv`
- `candidates_reviewed.csv`
- `manifest.csv`
- `summary.json`
- optional `visualization/index.md`

Candidate rows include `peak_period_records`, `period_start_records`,
`period_stop_records`, `peak_freq_mhz`, and `peak_record`.
