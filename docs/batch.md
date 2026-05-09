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

The batch directory receives merged candidate, validation, and statistics
tables:

- `manifest.csv`
- `candidates_raw.all.csv`
- `candidates_reviewed.all.csv`
- `validation_summary.all.csv`
- `validation_reviewed.all.csv`
