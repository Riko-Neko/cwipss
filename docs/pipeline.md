# Cwipss Pipeline

Cwipss generates period candidates from `time x channel` data using a
per-channel continuous wavelet transform. CE4 `.2C/.2CL` is the currently
supported input format; FilterBank support belongs in the same input adapter
layer.

The production method is CPRO -> native PELT -> CPRF. The former frequency-referenced detector is
isolated in `packages/frcr` and is not imported by this pipeline. CPRO supplies
the 1D activity axis to the required native C++ PELT time-window bridge. No
alternate window detector or Python PELT fallback exists.

Core numerical stages do not switch estimators when data are degenerate. Raw
first-difference noise calibration must be finite and positive; otherwise the
channel fails explicitly. CPRF candidate emission returns no row when its
frozen concentration, contrast, persistence, width, and strength gates fail.

Implementation is split between `cwipss.workflows.search` for orchestration,
`cwipss.signal` for CWT/detection algorithms, and `cwipss.data.readers` for
input adapters. See `architecture.md` for package boundaries.

## Core Flow

```text
time x channel data
  -> per-channel CWT over explicit period grid
  -> period x time x channel power cube
  -> candidate period-domain filter
  -> absolute noise and wavelet-gain calibration
  -> soft absolute-power and local period-ridge contrast response
  -> 3-bin period support and strongest-period reduction
  -> one edge-preserving CPRO observation axis per channel
  -> native C++ PELT mean-shift segments on that axis
  -> bidirectional single-ridge continuity evidence on CWT2D
  -> active-segment mean + continuity + ridge-lock gates
  -> merge adjacent accepted PELT segments
  -> accepted PELT time windows
  -> CPRF on unmasked, independently normalized absolute CWT
  -> one accepted period-family candidate per accepted CPRF window
  -> veto, validation, statistics, report
```

The overview projection is directly interpretable:

- x-axis: observation channel or MHz coordinate;
- y-axis: period in records or seconds;
- value: CWT overview or recorded single-channel candidates.

## Candidate Meaning

A CPRF-accepted peak family inside a PELT window is only a candidate. It is not a
confirmed periodic signal. Final review still requires original time-series
validation, null tests, RFI veto, and multiple-testing correction.

## Candidate Sensitivity

See [cpro.md](cpro.md) for equations and the CUDA transfer boundary. Defaults:

- candidate period domain `10..200` records;
- calibrated threshold `32` and texture quantile `0.9375`;
- period contrast `1.5` using center/context widths `3/15` bins;
- period support `3` bins, power softness `1.0`, and contrast softness `0.10`;
- fixed strongest-period reduction supplies the unsmoothed PELT boundary axis;
- bidirectional continuity uses decay `0.995` and power `2`; accepted PELT
  segments require normalized continuity mean `0.47` and single-ridge energy
  lock `0.94`;
- native PELT uses penalty `16`, minimum segment size `64`, endpoint stride
  `8`, and eight C++ workers in the main configuration;
- active PELT segments require standardized activity mean `0.05`; adjacent
  accepted segments are merged with gap `0`. There is no independent hard
  duration gate after PELT; `pelt_min_size_records=64` is the sole time-size
  constraint and belongs to the segmentation model itself;
- CPRF uses `min_band_persistence=0.20`, `min_band_concentration=0.30`,
  `min_local_contrast=0.65`, and `min_integrated_strength=0.0`; it evaluates up
  to eight local peak hypotheses
  but emits only the highest-scoring accepted period family;
- retained candidates capped per channel: `max_candidates_per_channel=auto`
  uses `max_candidates_per_record=3/4096`, while an integer value is a hard
  per-channel cap;
- validation capped at `validation.max_candidates=25`.

CPRO parameters are scientific configuration, not performance knobs. Modified
values must be retained in `config.resolved.json`.

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
time, and detection substage totals for CPRO, native PELT, PELT wait, and
CPRF scoring. The final summary line
reports file-level totals.

## CWT Backend

The default backend is `cpu`, which uses the original PyWavelets implementation.
This is the compatibility path and remains the default for all existing configs
and commands.

`--cwt-backend cuda` enables the optional CuPy FFT backend for CWT power
generation. It currently supports `--cwt-method fft`. In the main scan and
injection benchmark paths, CUDA keeps the CWT power and array-heavy structure /
activity, and window-specific CPRF computation on the GPU. Only the 1D CPRO
shape axis reaches CPU for native PELT; returned window indices
resume CPRF on retained 2D CWT, and only final candidate scalars reach CPU.
The default pending depth is two blocks so native PELT overlaps the next GPU block.
`--cwt-backend auto` uses CUDA only when CuPy and the selected
CUDA device are available; otherwise it falls back to the CPU backend. Explicit
`cuda` mode fails fast if CUDA cannot be initialized.

## Time Aggregation

CWT first produces `period x time x channel`. Detection uses this first-hand
cube directly. The default aggregation is `p95` and is now used only for overview
maps, not candidate generation.
Supported aggregation methods are `max`, `mean`, `median`, `pNN`, and
`percentile`.

Before aggregation, visualization can write representative-channel
`period x time` scalograms, CPRO score maps, PELT activity windows, period profiles,
and period-channel overview maps. These are middleware views
for inspecting whether a period response is persistent, burst-like, coherent
over neighboring period-time pixels, or contaminated by isolated texture.

Runtime visualization is intentionally representative. A completed run or
batch can be post-processed with `scripts/run_candidate_gallery.py` to produce
the standard raw time-frequency and period-time CWT views for each selected
Top-N candidate without rerunning candidate detection.

## Outputs

Each run writes:

- `candidates_raw.csv`
- `candidates_reviewed.csv`
- `time_windows.csv`
- `manifest.csv`
- `summary.json`
- optional `visualization/index.md`

Candidate rows use schema v7 stage-specific evidence: `shape_*`, `pelt_*`,
`cont_mean`, `ridge_lock`, `ridge_*`, `band_conc`, `band_persist`,
`local_contrast`, and `score`.
Coordinates use `t0_rec`, `t1_rec`, `freq_mhz`, and `period_rec`.

Post-processing may additionally create `candidate_gallery/index.md` and
candidate-identified PNGs grouped under `candidate_gallery/raw/` and
`candidate_gallery/cwt/`. These are derived review artifacts, not primary
detector outputs.
