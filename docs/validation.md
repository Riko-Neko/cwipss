# SWT Period Validation MVP

The validation stage re-examines SWT candidates in the original time-channel
source data through the active application adapter. It produces candidate-level
evidence only; it does not promote a candidate to a confirmed periodic signal.

## Command

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_validation.py \
  --run-dir runs/<run_id> \
  --max-candidates 20 \
  --shuffle-trials 100
```

By default, vetoed candidates are skipped. Use `--include-vetoed` for debugging
or threshold-tuning runs.

## Method

For each selected candidate:

- extract a record window around `peak_record`;
- extract the candidate channel span from the original source data;
- robust-normalize each selected frequency channel;
- average the normalized channels into one candidate time series;
- search integer-record periods around `approx_scale_records`;
- compute autocorrelation, FFT periodogram, and folding evidence;
- shuffle the time series to estimate an empirical p-value for the folding
  metric.

## Outputs

- `validation_summary.csv`: one row per evaluated candidate.
- `validation/validation_config.json`: resolved validation settings.
- `validation/candidate_000001.json`: per-candidate evidence records.

## Current Limits

- The search grid is integer records only.
- The period seed is still the approximate SWT scale, not a physical period.
- The shuffle/null test is candidate-local; project-level multiple-testing
  control is still a later statistics stage.
- No claim should be based on one validation metric alone.
