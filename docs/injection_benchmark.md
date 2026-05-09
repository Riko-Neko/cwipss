# Injection Benchmark

The injection benchmark tests whether the CWT period-search pipeline recovers
controlled synthetic periodic signals.

The default injected signal is `single_channel_periodic`: a time-modulated
periodic signal added to one channel only. Cross-channel models are explicit
stress tests, not the default scientific assumption.

## Command

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_injection_benchmark.py \
  --background synthetic \
  --records 1024 \
  --channels 64 \
  --period-records 8 16 32 \
  --amplitudes 8 16 \
  --grid \
  --visualize
```

The benchmark uses the same conservative candidate defaults as the main
pipeline: `threshold=6.0`, `min_pixels=6`, and
`max_candidates_per_block=50`. Lower these only for high-recall diagnostic
sweeps.

## Outputs

- `injection_truth.csv`
- `candidates_raw.csv`
- `candidates_reviewed.csv`
- `validation_summary.csv`
- `validation_reviewed.csv`
- `injection_results.csv`
- `injection_performance.csv`
- `injection_summary.json`
- optional `visualization/index.md`

## Failure Stages

- `missed_detection`: no CWT period-channel component overlaps the injection.
- `vetoed`: the best matching raw candidate did not survive veto.
- `not_validated`: the matched candidate was not in the validation table.
- `period_mismatch`: validation refined period is too far from truth.
- `validated`: detection, veto, and validation all passed configured criteria.
