# Injection Benchmark

The injection benchmark tests whether the CWT period-search pipeline recovers
controlled synthetic periodic signals.

The standard injected signal model is `single_channel_periodic`: a
time-modulated periodic signal added to one channel only. Cross-channel models
are explicit stress tests, not the default scientific assumption.

## Command

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_injection_benchmark.py \
  --background synthetic \
  --records 1024 \
  --channels 64 \
  --injection-config configs/injection_synthetic_smoke.json \
  --visualize
```

All injection suites are declared in JSON and passed with `--injection-config`:

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_injection_benchmark.py \
  --background ce4 \
  --input data/CE4/example.2C \
  --f-start 0.1 \
  --f-stop 2.0 \
  --t-start 0 \
  --t-stop 4096 \
  --injection-config configs/injection_lowfreq_random_weak.json \
  --visualize
```

When a config is supplied, it is copied to `injection_config.json` inside the
run directory. The command line controls background selection, search,
validation, and visualization only; injection parameters live in the simulation
config.

The benchmark uses the same single-channel low-floor PELT/profile detector as
the main pipeline: candidate period domain `10..200` records, low-floor fraction
`0.20`, PELT windowing, and `max_candidates_per_block=50`. Lower
`pelt_penalty`, `window_min_activity_mean`, or `profile_min_prominence` only for
high-recall diagnostic sweeps.

## Injection Config

The config is a reproducible simulation plan. A top-level `seed` controls all
sampling. Each entry in `sets` defines a group of base signals with `count` and
sampled parameters:

- `signal_model`: usually `single_channel_periodic` for channel-local tests.
- `period_records`: fixed value, list, or sampled range.
- `amplitude`: fixed value or sampled range, including `log_uniform`.
- `frequency_mhz` or `channel_center`: sampled injection channel.
- `time.duration_records` or `time.duration_fraction`: random time span; the
  start is random unless `time.record_start` is specified.
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

## Failure Stages

- `missed_detection`: no single-channel PELT/profile candidate overlaps the injection.
- `vetoed`: the best matching raw candidate did not survive veto.
- `not_validated`: the matched candidate was not in the validation table.
- `period_mismatch`: validation refined period is too far from truth.
- `validated`: detection, veto, and validation all passed configured criteria.
