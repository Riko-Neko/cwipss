# Markdown Reports

`scripts/run_report.py` generates a lightweight Markdown review report for a
single run or a batch directory.

## Command

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_report.py \
  --run-dir runs/<run_id-or-batch_id> \
  --top-n 10
```

## Contents

Single-run reports include:

- run summary and runtime versions;
- veto distribution;
- top candidates by SWT score;
- top validation evidence rows;
- an interpretation note that no signal claim is made.

Batch reports include:

- batch summary;
- per-file status table;
- merged veto distribution;
- top candidates by SWT score;
- top validation evidence rows using batch-level global q-values.

Reports are intended for triage and review. The source CSV/JSON files remain
the authoritative machine-readable outputs.
