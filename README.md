# CWIPSS: Continuous Wavelet Investigation for Periodic Spectral Signals

Cwipss (CWIPSS) is a reproducible CWT period-candidate search pipeline for
time-channel spectral data.

The core pipeline treats input as a dynamic spectrum or equivalent
`time x channel` matrix. Mission- or instrument-specific file formats belong to
the adapter layer. The current supported data format is CE4 LFRS `.2C/.2CL`;
FilterBank support is planned for the same adapter interface. CWT detection,
veto, validation, injection benchmarking, and reporting are defined
independently of any one input format.

The candidate generator is:

1. read time-channel data through an adapter;
2. run per-channel CWT over an explicit period grid;
3. crop to the configured trusted period domain;
4. estimate a single-channel low-fraction CWT noise floor from the lowest 20%
   of valid CWT power;
5. compute signed relative excess power;
6. build a structure-gated CWT map using per-period low-quantile background
   removal and local 2D time-period support;
7. compress the structured map into a trimmed period-axis activity curve;
8. use PELT to propose time windows per channel;
9. integrate each window into a period profile and rank period peaks.

Detected period-profile peaks are candidates only. Validation in the original
time series is still required before any signal interpretation.

Single-channel period candidates are valid targets. The legacy-style
`fixed_channel` and time-edge vetoes are disabled by default because they are
not meaningful for this single-channel candidate engine.

## Layout

```text
.
  configs/                 JSON configs for reproducible scans
  docs/                    design notes and scientific assumptions
  scripts/                 command-line entrypoints
  src/cwipss/              Cwipss implementation package and input adapters
  tests/                   synthetic tests
```

Generated products go under `runs/` and are ignored by git.

## Quick Start

Activate a Python environment with the project dependencies installed, then run:

```bash
python scripts/run_cwt_candidates.py \
  --input data/CE4/example.2C \
  --f-start 38.0 \
  --f-stop 38.3 \
  --t-start 0 \
  --t-stop 2048 \
  --period-min-records 2 \
  --period-max-records 512 \
  --period-count 96 \
  --block-channels 32
```

Config files can use the structured layout in `configs/cwt_default.json`. CLI
arguments override matching config values.

Default candidate generation is intentionally conservative:

- `candidate_period_min_records=10` and `candidate_period_max_records=200`:
  reject low-period instrument-like stripes and long-period trend-like domains.
- `noise_floor_fraction=0.20`: estimate the channel floor from the lowest 20%
  of trusted CWT power.
- `structure_baseline_quantile=0.10`, `structure_scale_quantile=0.20`,
  `structure_z_threshold=1.0`, `structure_time_support_records=64`,
  `structure_period_support_bins=3`, and
  `structure_min_support_fraction=0.10`: suppress isolated noise texture before
  the period-axis activity compression.
- `activity_smooth_records=16`, `pelt_penalty=16`, `pelt_min_size_records=384`,
  and `window_min_duration_records=384`: reject
  short unstable activity windows.
- `window_min_activity_raw_mean=25.0`: after PELT proposes a window on the
  standardized activity curve, require enough raw structured CWT activity
  before the window can emit period candidates. This prevents per-channel
  robust standardization from turning weak residual noise texture into false
  windows.
- `window_merge_gap_records=256`: merge nearby PELT segments so one persistent
  signal is less likely to be split into multiple windows.
- `profile_max_peaks_per_window=1`: treat one PELT time window as one
  period-family candidate; Sa-like side lobes or harmonics are not emitted as
  separate default candidates.
- `max_candidates_per_block=50`: cap retained candidates per frequency block.
- `validation.max_candidates=25`: cap rows passed to validation by default.

For diagnostic/high-recall sweeps, lower `--pelt-penalty`,
`--window-min-activity-mean`, or `--profile-min-prominence` explicitly. Do not
use high-recall settings as the default review mode.

The default candidate period domain assumes local scans of roughly 4096 records
and target responses lasting at least half that window: with at least 10 cycles
required for stable CWT/folding evidence, `0.5 * 4096 / 10 ~= 200` records.
The lower cutoff of 10 records is a practical samples-per-cycle floor above the
Nyquist limit and suppresses persistent short-period artifacts.

CLI scans show a CWT channel-progress bar by default. Use `--no-progress` to
disable it, or `--progress-leave` to keep the finished bar in terminal logs.
Use `--timing` to print per-block timing diagnostics for read, CWT, detection,
and detection substages. It is disabled by default.

The CWT backend defaults to `cpu`, preserving the original PyWavelets path.
Systems with CuPy/CUDA can opt in with `--cwt-backend cuda --cuda-device 0`.
`--cwt-backend auto` uses CUDA when available and otherwise falls back to CPU.
CUDA scans keep CWT power plus structure/activity/profile compression on the GPU
before returning to the shared CPU PELT/candidate logic.

Each run writes:

- `config.resolved.json`
- `manifest.csv`
- `time_windows.csv`
- `candidates_raw.csv`
- `candidates_reviewed.csv`
- `summary.json`

`summary.json` records Python, NumPy, PyWavelets, SciPy, and local-filter
runtime information.

## Validation

```bash
python scripts/run_validation.py \
  --run-dir runs/<run_id> \
  --max-candidates 20 \
  --shuffle-trials 100
```

This writes `validation_summary.csv` and per-candidate JSON files under
`validation/`.

```bash
python scripts/run_stats.py \
  --run-dir runs/<run_id>
```

This writes `validation_reviewed.csv` with p-values, run-level q-values,
global q-values, and deterministic evidence ranks.

## Batch

```bash
python scripts/run_batch.py \
  --input-dir data/CE4 \
  --pattern "*.2C" \
  --batch-id smoke_batch \
  --f-start 38.0 \
  --f-stop 38.3 \
  --t-start 0 \
  --t-stop 2048
```

Each source file gets an isolated run under `runs/<batch_id>/files/`. Batch
runs also print a colored start/done/error line per input file, show a file-level
progress bar plus the per-file CWT channel progress, and copy per-file CSVs into
`runs/<batch_id>/per_file_results/` as each file completes. The batch directory
also receives merged candidate, validation, and statistics tables.

## Visualization

```bash
python scripts/run_cwt_candidates.py \
  --input data/CE4/example.2C \
  --f-start 38.0 \
  --f-stop 38.3 \
  --t-start 0 \
  --t-stop 2048 \
  --visualize
```

This writes `visualization/index.md` plus PNG diagnostics for:

- raw time-channel matrix;
- representative-channel full `period x time` CWT scalograms;
- trusted-period structure-gated maps after floor normalization, per-period
  low-quantile standardization, and local 2D support gating;
- single-channel signed activity curves with recorded PELT windows;
- windowed period profiles with candidate period spans;
- aggregated `period x channel` overview maps for review only;
- veto review and optional validation/injection summaries.

## Injection Benchmark

```bash
python scripts/run_injection_benchmark.py \
  --background synthetic \
  --records 1024 \
  --channels 64 \
  --injection-config configs/injection_synthetic_smoke.json \
  --visualize \
  --run-id injection_smoke
```

The smoke config uses single-channel, time-modulated periodic signals.
Cross-channel injection models remain available only as explicit stress tests
inside injection configs.
Injection benchmark settings are always loaded from JSON; use
`configs/injection_lowfreq_random_weak.json` for a weak randomized suite with
sampled periods, modulation, time spans, frequencies, and same-signal frequency
copies.

## Report

```bash
python scripts/run_report.py \
  --run-dir runs/<run_id-or-batch_id>
```

This writes `report.md` with candidate counts, veto distribution, top CWT
candidates, validation evidence, and links to stage visualizations when present.

## License

This project is licensed under the MIT License. See `LICENSE`.
