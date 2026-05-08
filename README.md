# SWT Period Search Pipeline

Reproducible SWT period-candidate search pipeline for time-channel data.

The core pipeline treats the input as a dynamic spectrum or equivalent
time-channel matrix. Mission- or instrument-specific file formats belong to the
application adapter layer. The current application adapter reads CE-4 LFRS
`.2C/.2CL` files, but the SWT detection, veto, validation, and reporting
contracts are defined independently of that file format.

The current development target is a whole-file, time-axis SWT post-processing
pipeline:

1. read time-channel data through an application adapter;
2. run block-wise stationary wavelet transform along the time axis;
3. estimate local robust S/N on wavelet power maps;
4. detect bright bands / bright points as candidates;
5. write a candidate table for later RFI veto and period validation.

## Layout

```text
.
  configs/                 JSON configs for reproducible scans
  docs/                    design notes and scientific assumptions
  scripts/                 command-line entrypoints
  src/ce4_period_search/   current implementation package and CE-4 adapter
  tests/                   small synthetic tests
```

Generated products go under `runs/` and are ignored by git.

Core runtime dependencies are declared in `pyproject.toml`. The detection stage
requires SciPy and always uses `scipy.ndimage.median_filter` for local robust
S/N estimation and `scipy.ndimage.label` for connected components.

## Quick Start

Use the `pytorch` conda environment:

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_swt_candidates.py \
  --input data/CE4/CE4_GRAS_LFRS-TR_SCI_P_20190830160000_20190831040000_0056_B.2C \
  --f-start 38.0 \
  --f-stop 40.0 \
  --levels 5 \
  --block-channels 128
```

For a small smoke run:

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_swt_candidates.py \
  --input data/CE4/CE4_GRAS_LFRS-TR_SCI_P_20190830160000_20190831040000_0056_B.2C \
  --f-start 38.0 \
  --f-stop 38.3 \
  --t-start 0 \
  --t-stop 2048 \
  --levels 3 \
  --block-channels 32
```

Config files can use the structured layout in `configs/swt_default.json`. CLI
arguments override matching config values.

Each run writes:

- `config.resolved.json`
- `manifest.csv`
- `candidates_raw.csv`
- `candidates_reviewed.csv`
- `candidates.csv` as a temporary compatibility alias
- `summary.json`

`summary.json` records the Python, NumPy, PyWavelets, and SciPy versions used
for the run, plus the local-filter implementation.

Candidate validation is a separate step:

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_validation.py \
  --run-dir runs/<run_id> \
  --max-candidates 20 \
  --shuffle-trials 100
```

This writes `validation_summary.csv` and per-candidate JSON files under
`validation/`.

Validation statistics are also a separate step:

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_stats.py \
  --run-dir runs/<run_id>
```

This writes `validation_reviewed.csv` with p-values, run-level q-values,
global q-values, and deterministic evidence ranks.

Batch processing is available for multiple inputs:

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_batch.py \
  --input-dir data/CE4 \
  --pattern "*.2C" \
  --batch-id smoke_batch \
  --f-start 38.0 \
  --f-stop 38.3 \
  --t-start 0 \
  --t-stop 2048
```

Each source file gets an isolated run under `runs/<batch_id>/files/`, and the
batch directory receives merged candidate, validation, and statistics tables.

Generate a Markdown review report for either a single run or a batch:

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_report.py \
  --run-dir runs/<run_id-or-batch_id>
```

This writes `report.md` with candidate counts, veto distribution, top SWT
candidates, and top validation evidence rows.

Stage visualization can be enabled on scans, batches, and injection benchmarks:

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_swt_candidates.py \
  --input data/CE4/example.2C \
  --f-start 38.0 \
  --f-stop 38.3 \
  --t-start 0 \
  --t-stop 2048 \
  --visualize
```

This writes `visualization/index.md` plus PNG diagnostics for the raw matrix,
SWT power, local S/N, candidate overlays, veto review, and optional
validation/injection stages. See `docs/visualization.md`.

Run an injection benchmark to test recovery of controlled synthetic periodic
signals:

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_injection_benchmark.py \
  --background synthetic \
  --records 1024 \
  --channels 64 \
  --period-records 8 16 32 \
  --amplitudes 4 6 8 \
  --grid \
  --run-id injection_smoke
```

This writes truth, candidate, validation, reviewed statistics, and per-injection
recovery/performance tables. See `docs/injection_benchmark.md`.

## Current Application Adapter

The bundled reader in `src/ce4_period_search/io.py` supports CE-4 LFRS
`.2C/.2CL` dynamic spectra. This is the first application target, not the
definition of the core SWT period search method. Future readers should expose
the same time-channel block interface used by the pipeline.

## Interpretation

Detected components are not claimed signals. They are SWT-localized candidates.
Each candidate still needs validation in the original time series, including
folding, autocorrelation / periodogram checks, shuffle controls, and RFI veto.

## License

This project is licensed under the MIT License. See `LICENSE`.
