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
- Stage 02: representative-channel full `period x time` CWT scalograms.
- Stage 03: representative-channel trusted-period relative excess after
  single-channel low-20% floor normalization.
- Stage 04: signed period-axis activity curve with recorded PELT time windows.
- Stage 05: windowed period profiles used to choose candidate periods.
- Stage 06: aggregated `period x channel` overview response map for review
  only; detection does not use this projection.
- Stage 07: candidate-domain `period x channel` overview with recorded
  candidate overlays.
- Stage 08: candidate review overview colored by veto status.
- Stage 09: validation/statistics overview when available.
- Stage 10: injection recovery rates when available.
- Stage 11: injected period versus refined period when available.

## Controls

- `--candidate-period-min-records`: reject candidates below this period.
  Default `10`.
- `--candidate-period-max-records`: reject candidates above this period.
  Default `200`.
- `--noise-floor-fraction`: lowest fraction of trusted CWT power used as
  channel noise floor. Default `0.20`.
- `--pelt-penalty`: PELT mean-shift penalty. Default `8`.
- `--window-min-duration-records`: minimum PELT window duration. Default `256`.
- `--profile-min-prominence`: minimum period-profile peak prominence.
  Default `0.5`.
- `--viz-max-blocks`: maximum blocks to visualize; `0` renders all.
- `--viz-max-channels`: representative channels per block for period-time CWT
  scalograms.
- `--viz-top-candidates`: maximum candidates drawn in overview plots.
- `--viz-dpi`: PNG resolution.

Use small time/channel windows first. Full-file visualization can be expensive.
Avoid low-score debug settings unless the goal is high-recall visual exploration
rather than candidate review.
