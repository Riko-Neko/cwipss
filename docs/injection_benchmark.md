# Injection Benchmark

The injection benchmark tests whether the SWT period-search pipeline can
recover controlled synthetic periodic signals. It is an engineering performance
test, not evidence that any real candidate is astrophysical or physical.

## Purpose

The normal SWT pipeline detects bright local structures on SWT power maps. That
only proves that the detector produced a candidate. Injection benchmarking adds
a known truth table so we can measure where a controlled signal is lost:

- `missed_detection`: no SWT component overlaps the injected time-frequency
  span.
- `vetoed`: a component overlaps the injection but rule-based veto rejected it.
- `not_validated`: the matched candidate did not enter validation.
- `period_mismatch`: validation ran but the refined period was too far from
  truth.
- `validated`: detection, veto, and validation all passed the configured match
  rules.

## Command

Synthetic background:

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_injection_benchmark.py \
  --background synthetic \
  --records 1024 \
  --channels 64 \
  --period-records 8 16 32 \
  --amplitudes 4 6 8 \
  --grid \
  --repeats 3 \
  --threshold 4.0 \
  --visualize \
  --min-pixels 6 \
  --validation-shuffle-trials 100 \
  --run-id injection_smoke
```

CE-4 background with synthetic injections:

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_injection_benchmark.py \
  --background ce4 \
  --input data/CE4/example.2C \
  --f-start 38.0 \
  --f-stop 38.3 \
  --t-start 0 \
  --t-stop 4096 \
  --period-records 16 32 \
  --amplitudes 5 8 \
  --run-id ce4_injection_smoke
```

## Outputs

The benchmark output directory contains:

- `injection_truth.csv`: injected signal truth table.
- `candidates_raw.csv`: SWT connected-component candidates before veto.
- `candidates_reviewed.csv`: candidate table after rule-based veto.
- `validation_summary.csv`: original-time-series validation metrics.
- `validation_reviewed.csv`: validation p/q values and evidence ranks.
- `injection_results.csv`: one row per injection with recovery status.
- `injection_performance.csv`: grouped recovery rates by signal model, period,
  and amplitude.
- `injection_summary.json`: counts and benchmark configuration.
- `visualization/index.md`: staged diagnostic figures when `--visualize` is
  enabled.

## Current Models

The first implementation includes five simple morphology models:

- `pulsed_periodic`: narrow pulses repeated with a fixed period.
- `intermittent_periodic`: currently uses the same pulse kernel as
  `pulsed_periodic`; later versions should add on/off windows.
- `sinusoidal_narrowband`: sinusoid in time with a narrow channel envelope.
- `band_limited_periodic`: sinusoidal time modulation over a wider band.
- `drifting_ridge`: sinusoidal time modulation with a linearly drifting channel
  center.

## Metrics To Add Next

The current result table is per-injection and stage-oriented. The next layer
should aggregate detection efficiency by period, amplitude, bandwidth, duty
cycle, background type, and veto rule. False-positive rate should be measured
with no-injection synthetic/null runs and with shuffled backgrounds.
