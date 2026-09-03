# CWIPSS: Continuous Wavelet Investigation for Periodic Spectral Signals

Cwipss (CWIPSS) is a reproducible CWT period-candidate search pipeline for
time-channel spectral data.

The core pipeline treats input as a dynamic spectrum or equivalent
`time x channel` matrix. Mission- or instrument-specific file formats belong to
the adapter layer. The current supported data format is CE4 LFRS `.2C/.2CL`;
FilterBank support is planned for the same adapter interface. CWT detection,
veto, validation, injection benchmarking, and reporting are defined
independently of any one input format.

The production chain is Calibrated Period-Ridge Observation (CPRO), native
C++ PELT, then the Concentrated Periodic Ridge Filter (CPRF). Every stage uses
one physical frequency channel's own absolute `period x time` CWT map; no
neighboring physical channel contributes to the result.

The frequency-referenced detector retained from earlier experiments now lives
as the independent [`packages/frcr`](packages/frcr) extension. It is not
imported by `src/cwipss` and is documented only for its single-channel or
narrowband frequency-contrast use case.

## Layout

```text
.
  configs/                 JSON configs for reproducible scans
  docs/                    design notes and scientific assumptions
  scripts/                 command-line entrypoints
  src/cwipss/
    signal/                CWT, CPRO, CPRF, native PELT, CPU/CUDA
    data/                  spectrum readers and CSV schemas
    workflows/             single-file search and multi-file batch orchestration
    analysis/              veto, validation, statistics, injection benchmarks
    reporting/             reports, plots, stage views, candidate galleries
    config.py              resolved project configuration
    runtime.py             runtime and dependency metadata
  tests/                   synthetic tests
  packages/frcr/           standalone frequency-referenced detector core
  datasets/cpsr_1993/      CPSR-1993 manual-review dataset documentation
```

See `docs/architecture.md` for package boundaries and dependency direction.

Generated products go under `runs/` and are ignored by git.

## Installation

Install the project in the active environment before running scripts:

```bash
python -m pip install -e .
```

Candidate generation requires the native C++ PELT extension. Building it
requires a C++17 compiler and CMake; there is intentionally no Python or
alternate window-detector fallback.

## Search Entrypoint

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

The standalone frequency-referenced extension is documented in
[`packages/frcr/README.md`](packages/frcr/README.md).
CPRO and the PELT-to-CPRF boundary are documented in [`docs/cpro.md`](docs/cpro.md).
The 1,993-case manual-review benchmark is documented in
[`datasets/cpsr_1993/README.md`](datasets/cpsr_1993/README.md). Its 57 MB
single-channel data archive is distributed as a GitHub Release asset rather
than committed to the repository.

Default candidate generation is intentionally conservative:

- `candidate_period_min_records=10` and `candidate_period_max_records=200`:
  reject low-period instrument-like stripes and long-period trend-like domains.
- `cpro_threshold_snr=32` and `cpro_texture_quantile=0.9375`: require absolute
  calibrated power above both the fixed and map-texture thresholds.
- `cpro_min_period_contrast=1.5`, center width `3`, and context width `15`:
  require a narrow period ridge rather than broad CWT texture.
- `cpro_period_support_bins=3`, `cpro_shape_power_softness=1.0`, and
  `cpro_shape_contrast_softness=0.10`: form an edge-preserving CPRO map and take
  the strongest period response at each record for PELT boundaries.
- `pelt_penalty=16`, `pelt_min_size_records=64`, and `pelt_jump_records=8`:
  run the required native mean-shift segmentation on that 1D activity.
- `cpro_continuity_decay=0.995`, `cpro_continuity_power=2`,
  `cpro_min_continuity_mean=0.47`, and `cpro_min_ridge_lock=0.94`: retain PELT
  segments whose energy persists bidirectionally in time and remains locked to
  one CWT period ridge. No independent `96`/`640`-record duration cutoff exists.
- standardized PELT activity gate `0.05` and merge gap `0`: select and join
  adjacent active segments without moving PELT boundaries.
- `cprf_min_band_persistence=0.20`, `cprf_min_band_concentration=0.30`,
  `cprf_min_local_contrast=0.65`, and `cprf_min_integrated_strength=0`: apply
  the current broad-recall CPRF working point to unmasked absolute CWT inside
  each PELT window.
- `max_candidates_per_channel=auto` and `max_candidates_per_record=3/4096`:
  derive a per-channel cap from the current record length. Set
  `max_candidates_per_channel` to an integer to use that hard per-channel cap.
- `validation.max_candidates=25`: cap rows passed to validation by default.

Changing CPRO or CPRF parameters changes the scientific method and must be
recorded in resolved configuration; there is no alternate scientific fallback.

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
CUDA scans keep CWT power, CPRO maps, and CPRF on the GPU. Only CPRO's 1D shape
axis crosses to CPU for native C++ PELT; returned window
indices resume CPRF on the retained 2D CWT, and only final scalars return.

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
progress bar plus the per-file CWT channel progress. Each isolated run contains
that file's CSV results, while the batch directory contains merged candidate,
validation, and statistics tables.

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
- aggregated `period x channel` overview maps for review only;
- veto review and optional validation/injection summaries.

Runtime visualization intentionally renders representative blocks and
channels. To render one raw image and one CWT scalogram for each globally
ranked candidate in an existing run or batch:

```bash
python scripts/run_candidate_gallery.py \
  --run-dir runs/<run_id-or-batch_id> \
  --top 100
```

The default ordering uses validation `evidence_rank` when available and CPRF
`score` otherwise. Raw views are written as
`candidate_gallery/raw/<rank>_<run>_candidate_<id>.png`, and CWT views use the
same filename under `candidate_gallery/cwt/`. Use `--source-root` when result
CSVs contain source paths from another machine.

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
`configs/injection_fullband_random_100.json` for a weak randomized
full-duration suite with 100 base signals, full-band frequency placement, and
same-signal frequency copies.

## Report

```bash
python scripts/run_report.py \
  --run-dir runs/<run_id-or-batch_id>
```

This writes `report.md` with candidate counts, veto distribution, top CWT
candidates, validation evidence, and links to runtime stage visualizations when
present. Candidate-gallery results remain available through their own
`candidate_gallery/index.md`.

## License

This project is licensed under the MIT License. See `LICENSE`.
