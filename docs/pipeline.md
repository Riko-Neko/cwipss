# Cwipss Pipeline

Cwipss generates period candidates from `time x channel` data using a
per-channel continuous wavelet transform. CE4 `.2C/.2CL` is the currently
supported input format; FilterBank support belongs in the same input adapter
layer.

## Core Flow

```text
time x channel data
  -> per-channel CWT over explicit period grid
  -> period x time x channel power cube
  -> candidate period-domain filter
  -> single-channel low-fraction CWT noise floor
  -> signed relative excess power
  -> per-period low-quantile standardization
  -> local 2D time-period support gate
  -> trimmed period-axis activity curve from structured CWT map
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
- per-period structure background quantile `0.10`, scale quantile `0.20`,
  structure z threshold `1.0`, time support `64` records, period support `3`
  bins, and minimum local support fraction `0.10`;
- PELT penalty `16`, activity smoothing `16` records, minimum segment/window
  size `384` records, and exact PELT endpoint search
  (`pelt_jump_records=1`); CUDA runs can parallelize native CPU PELT across
  channels with `pelt_threads > 1`;
- CUDA structure preprocessing uses the stable per-channel path by default;
  `cuda_structure_batch=true` batches structure/activity preprocessing in
  channel chunks, with `cuda_structure_batch_channels=16` as the default
  memory/performance tradeoff;
- after PELT, a raw structured-activity mean floor of `25.0` is applied before
  period-profile candidates are emitted;
- nearby PELT windows are merged across gaps up to `256` records;
- each PELT time window emits one default period-family candidate
  (`profile_max_peaks_per_window=1`), so Sa-like side lobes are not split into
  separate default candidates;
- retained candidates capped per channel: `max_candidates_per_channel=auto`
  uses `max_candidates_per_record=3/4096`, while an integer value is a hard
  per-channel cap;
- validation capped at `validation.max_candidates=25`.

Lower PELT/profile thresholds or `profile_max_peaks_per_window > 1` are
debug/high-recall settings. They are useful for inspecting weak period
responses, but they can split one Sa-like response envelope into many visually
plausible candidates and should not be used as the default review
configuration.

The raw structured-activity floor is deliberately separate from
`window_min_activity_mean`: PELT runs on robust-standardized activity to find
time boundaries, but candidate emission also needs absolute structured CWT
energy. Without this second gate, a noise-only channel can be standardized to
unit variance and segmented into visually plausible but weak false windows.

## Feasible Period Domain

The default candidate domain rejects periods below 10 records and above 200
records. This is a detection-domain filter, not a CWT-grid limit.

For currently supported CE4 files with `.2CL` labels, one record is about one
second. In the current 4096-record low-frequency review windows, the assumed
minimum real signal span is at least half the window:

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

Use `--timing` to print per-block timing diagnostics. This is disabled by
default. Each timing line reports block read time, CWT time, total detection
time, and detection substage totals for floor/excess, structure gating,
activity compression, PELT, and period-profile scoring. The final summary line
reports file-level totals.

## CWT Backend

The default backend is `cpu`, which uses the original PyWavelets implementation.
This is the compatibility path and remains the default for all existing configs
and commands.

`--cwt-backend cuda` enables the optional CuPy FFT backend for CWT power
generation. It currently supports `--cwt-method fft`. In the main scan and
injection benchmark paths, CUDA keeps the CWT power and array-heavy structure /
activity / profile compression on the GPU; PELT, window handling, profile peak
selection, and candidate row construction reuse the same CPU logic as the
default backend. `--cwt-backend auto` uses CUDA only when CuPy and the selected
CUDA device are available; otherwise it falls back to the CPU backend. Explicit
`cuda` mode fails fast if CUDA cannot be initialized.

## Time Aggregation

CWT first produces `period x time x channel`. Detection uses this first-hand
cube directly. The default aggregation is `p95` and is now used only for overview
maps, not candidate generation.
Supported aggregation methods are `max`, `mean`, `median`, `pNN`, and
`percentile`.

Before aggregation, visualization can write representative-channel
`period x time` scalograms, trusted-period structure-gated maps, PELT activity
curves, and windowed period profiles. These are the required middleware views
for inspecting whether a period response is persistent, burst-like, coherent
over neighboring period-time pixels, or contaminated by isolated texture.

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
