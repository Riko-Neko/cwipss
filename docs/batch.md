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

Each isolated run is the single source of truth for that file and contains its
raw/reviewed candidates, time windows, validation tables, configuration, and
summary. These files are written when that source file completes, so a server
run can be inspected before the full batch finishes.

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

## Candidate Visualization

Batch runtime visualization, when enabled, is stored inside each completed
`files/<run_id>/visualization/` directory and uses representative blocks and
channels.

To generate globally ranked per-candidate raw and CWT images from the merged
batch tables:

```bash
python scripts/run_candidate_gallery.py \
  --run-dir runs/<batch_id> \
  --top 100
```

If `source_file` values point to a different machine, provide a local directory
containing files with the same basenames:

```bash
python scripts/run_candidate_gallery.py \
  --run-dir runs/<batch_id> \
  --source-root /local/path/to/CE4_LFRS_2C \
  --top 100
```

The gallery reads merged candidate and validation tables, so it does not rerun
detection or validation.
