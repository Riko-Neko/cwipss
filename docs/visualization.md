# Stage Visualization

Stage visualization is optional and writes PNG diagnostics plus
`visualization/index.md`.

## Enable

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_cwt_candidates.py \
  --input data/example.2C \
  --f-start 38.0 \
  --f-stop 38.3 \
  --t-start 0 \
  --t-stop 2048 \
  --visualize
```

## Stages

- Stage 01: raw `time x channel` matrix.
- Stage 02: representative-channel `period x time` CWT scalograms before time
  aggregation.
- Stage 03: aggregated `period x channel` response map.
- Stage 04: local robust S/N and candidate overlays.
- Stage 05: candidate review overview colored by veto status.
- Stage 06: validation/statistics overview when available.
- Stage 07: injection recovery rates when available.
- Stage 08: injected period versus refined period when available.

## Controls

- `--threshold`: local robust S/N cutoff. Default `6.0` is conservative.
- `--min-pixels`: minimum connected-component area. Default `6`.
- `--viz-max-blocks`: maximum blocks to visualize; `0` renders all.
- `--viz-max-channels`: representative channels per block for period-time CWT
  scalograms.
- `--viz-top-candidates`: maximum candidates drawn in overview plots.
- `--viz-dpi`: PNG resolution.

Use small time/channel windows first. Full-file visualization can be expensive.
Avoid debug settings such as `--threshold 1.4 --min-pixels 1` unless the goal is
high-recall visual exploration rather than candidate review.
