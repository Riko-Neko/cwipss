# Injection Benchmark

The injection benchmark tests whether the Cwipss pipeline recovers
controlled synthetic periodic signals.

Simulation, injection, validation, and benchmark orchestration live under
`cwipss.analysis`.

The standard injected signal model is `single_channel_periodic`: a
time-modulated periodic signal added to one channel only. Cross-channel models
are explicit stress tests, not the default scientific assumption.

## Command

```bash
python scripts/run_injection_benchmark.py \
  --background synthetic \
  --records 1024 \
  --channels 64 \
  --injection-config configs/injection_synthetic_smoke.json \
  --visualize
```

All injection suites are declared in JSON and passed with `--injection-config`:

```bash
python scripts/run_injection_benchmark.py \
  --background ce4 \
  --input data/CE4/example.2C \
  --f-start 0.1 \
  --f-stop 40.0 \
  --t-start 0 \
  --t-stop 4096 \
  --injection-config configs/injection_fullband_random_100.json \
  --visualize
```

`--background ce4` uses the currently supported CE4 `.2C/.2CL` data-format
adapter. The benchmark logic itself remains format-independent.

When a config is supplied, it is copied to `injection_config.json` inside the
run directory. The command line controls background selection, search,
validation, and visualization only; injection parameters live in the simulation
config.

The benchmark uses the same CPRO implementation and parameters as the main
pipeline. It does not import the standalone frequency-referenced extension and
uses the same required native C++ PELT bridge. No alternate window detector or
Python PELT fallback is available.

## Injection Config

The config is a reproducible simulation plan. A top-level `seed` controls all
sampling. Each entry in `sets` defines a group of base signals with `count` and
sampled parameters:

- `signal_model`: usually `single_channel_periodic` for channel-local tests.
- `period_records`: fixed value, list, or sampled range.
- `amplitude`: fixed value or sampled range, including `log_uniform`.
- `frequency_mhz` or `channel_center`: sampled injection channel.
- `time.duration_records` or `time.duration_fraction`: injection time span; the
  bundled full-band weak suite fixes this to the entire record range.
- `modulation.phase` and `modulation.duty_cycle`: random temporal modulation
  shape.
- `replication`: probability and maximum number of same-signal copies at other
  channels or frequencies.

Sampler fields accept a raw value, a list, `{ "values": [...] }`,
`{ "value": ... }`, or `{ "distribution": "uniform|log_uniform|integer_uniform",
"min": ..., "max": ... }`.

## Outputs

- `injection_truth.csv`
- `time_windows.csv`
- `candidates_raw.csv`
- `candidates_reviewed.csv`
- `validation_summary.csv`
- `validation_reviewed.csv`
- `injection_results.csv`
- `injection_performance.csv`
- `injection_summary.json`
- optional `visualization/index.md`

The optional visualization uses the same function-based raw/CWT rendering core
as normal scans. Injection-specific Stage 10 and Stage 11 summaries are added
only when injection result rows are present.

## Failure Stages

- `missed_detection`: no single-channel CPRO/profile candidate overlaps the injection.
- `vetoed`: the best matching raw candidate did not survive veto.
- `not_validated`: the matched candidate was not in the validation table.
- `period_mismatch`: validation refined period is too far from truth.
- `validated`: detection, veto, and validation all passed configured criteria.
