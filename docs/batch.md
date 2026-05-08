# Batch Processing

Batch processing runs the single-file SWT candidate pipeline over multiple
inputs, keeps each file's artifacts isolated, and writes merged tables at the
batch root.

## Inputs

`scripts/run_batch.py` supports three input modes:

- `--input-dir DIR --pattern "*.2C"` discovers files in a directory.
- `--inputs file1 file2 ...` runs explicit paths.
- `--manifest manifest.csv` reads per-file jobs from a CSV.

Manifest rows require `input` or `source_file`. Optional per-file columns are:

- `run_id`
- `f_start` or `f_start_mhz`
- `f_stop` or `f_stop_mhz`
- `t_start`
- `t_stop`

## Output Layout

```text
runs/<batch_id>/
  batch_config.resolved.json
  manifest.csv
  candidates_raw.all.csv
  candidates_reviewed.all.csv
  validation_summary.all.csv
  validation_reviewed.all.csv
  files/
    <run_id>/
      config.resolved.json
      manifest.csv
      candidates_raw.csv
      candidates_reviewed.csv
      validation_summary.csv
      validation_reviewed.csv
```

The batch-level `validation_reviewed.all.csv` recomputes `global_q_value` across
the merged validation rows. Per-file `validation_reviewed.csv` files keep their
own run-local statistics.

## Failure Behavior

Files are processed serially. A failed file is recorded in the batch manifest
with `status=error` and does not stop the remaining files. The manifest is
rewritten after each file, so partial progress remains visible if a run is
interrupted.
