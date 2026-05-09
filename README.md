# CScout: A CWT Period Search Pipeline

Reproducible CWT period-candidate search for time-channel data.

The core pipeline treats input as a dynamic spectrum or equivalent
`time x channel` matrix. Mission- or instrument-specific file formats belong to
the application adapter layer. The bundled adapter reads CE-4 LFRS `.2C/.2CL`
files, but CWT detection, veto, validation, injection benchmarking, and
reporting are defined independently of that format.

The candidate generator is:

1. read time-channel data through an adapter;
2. run per-channel CWT over an explicit period grid;
3. score each full `period x time` scalogram with a period-axis
   Difference-of-Gaussians filter;
4. extract time-bounded period-response regions per frequency channel inside
   the configured candidate period domain;
5. rank regions by time-integrated score;
6. project candidates onto a `period x channel` overview map for review.

Detected scalogram regions are candidates only. Validation in the original time series
is still required before any signal interpretation.

Single-channel period candidates are valid targets. The legacy-style
`fixed_channel` and time-edge vetoes are disabled by default because they are
not meaningful for the time-aggregated CWT overview projection.

## Layout

```text
.
  configs/                 JSON configs for reproducible scans
  docs/                    design notes and scientific assumptions
  scripts/                 command-line entrypoints
  src/ce4_period_search/   implementation package and CE-4 adapter
  tests/                   synthetic tests
```

Generated products go under `runs/` and are ignored by git.

## Quick Start

Use the `pytorch` conda environment:

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_cwt_candidates.py \
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

- `threshold=2.5`: minimum per-channel scalogram region score.
- `candidate_period_min_records=10` and `candidate_period_max_records=200`:
  reject low-period instrument-like stripes and long-period trend-like domains.
- `min_duration_records=8`: minimum candidate time span.
- `max_width_bins=10`: reject broad period bands.
- `max_candidates_per_block=50`: cap retained regions per frequency block.
- `validation.max_candidates=25`: cap rows passed to validation by default.

For diagnostic/high-recall sweeps, lower `--threshold` or
`--min-duration-records` explicitly. Do not use low thresholds as the default
review mode.

The default candidate period domain assumes local scans of roughly 4096 records
and target responses lasting at least half that window: with at least 10 cycles
required for stable CWT/folding evidence, `0.5 * 4096 / 10 ~= 200` records.
The lower cutoff of 10 records is a practical samples-per-cycle floor above the
Nyquist limit and suppresses persistent short-period artifacts.

CLI scans show a CWT channel-progress bar by default. Use `--no-progress` to
disable it, or `--progress-leave` to keep the finished bar in terminal logs.

Each run writes:

- `config.resolved.json`
- `manifest.csv`
- `candidates_raw.csv`
- `candidates_reviewed.csv`
- `summary.json`

`summary.json` records Python, NumPy, PyWavelets, SciPy, and local-filter
runtime information.

## Validation

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_validation.py \
  --run-dir runs/<run_id> \
  --max-candidates 20 \
  --shuffle-trials 100
```

This writes `validation_summary.csv` and per-candidate JSON files under
`validation/`.

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_stats.py \
  --run-dir runs/<run_id>
```

This writes `validation_reviewed.csv` with p-values, run-level q-values,
global q-values, and deterministic evidence ranks.

## Batch

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

## Visualization

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_cwt_candidates.py \
  --input data/CE4/example.2C \
  --f-start 38.0 \
  --f-stop 38.3 \
  --t-start 0 \
  --t-stop 2048 \
  --visualize
```

This writes `visualization/index.md` plus PNG diagnostics for:

- raw time-channel matrix;
- representative-channel `period x time` CWT scalograms before time aggregation;
- aggregated `period x channel` response maps;
- projected per-channel scalogram score maps and candidate overlays;
- veto review and optional validation/injection summaries.

## Injection Benchmark

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_injection_benchmark.py \
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
/opt/miniconda3/envs/pytorch/bin/python scripts/run_report.py \
  --run-dir runs/<run_id-or-batch_id>
```

This writes `report.md` with candidate counts, veto distribution, top CWT
candidates, validation evidence, and links to stage visualizations when present.

## License

This project is licensed under the MIT License. See `LICENSE`.
