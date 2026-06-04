# Batch Processing

Batch processing runs the single-file CWT candidate pipeline over multiple input
files.

Inputs can come from:

- `--input-dir` and `--pattern`;
- `--inputs`;
- a manifest CSV with `input` or `source_file`.

Each file gets an isolated run under:

```text
runs/<batch_id>/files/<run_id>/
```

As each file completes, the batch runner also copies flat per-file CSVs to:

```text
runs/<batch_id>/per_file_results/<run_id>.candidates_reviewed.csv
runs/<batch_id>/per_file_results/<run_id>.validation_reviewed.csv
```

Additional per-file CSVs include raw candidates, time windows, and validation
summary when those files exist. This directory is written incrementally, so a
server run can be inspected before the full batch finishes.

The console prints a colored start/done/error line for every input file. Batch
runs also show a file-level progress bar; each file still shows its CWT
channel-progress bar unless `--no-progress` is used.

The batch directory receives merged candidate, validation, and statistics
tables:

- `manifest.csv`
- `candidates_raw.all.csv`
- `candidates_reviewed.all.csv`
- `time_windows.all.csv`
- `validation_summary.all.csv`
- `validation_reviewed.all.csv`
