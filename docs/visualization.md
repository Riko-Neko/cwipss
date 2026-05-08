# Stage Visualization

Stage visualization is an optional review layer. It writes PNG diagnostics and
an `index.md` file under each run directory. The figures are meant to audit the
pipeline stages; they are not evidence for a confirmed signal.

## Enable It

Single-file scan:

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_swt_candidates.py \
  --input data/example.2C \
  --f-start 38.0 \
  --f-stop 38.3 \
  --t-start 0 \
  --t-stop 2048 \
  --visualize
```

Injection benchmark:

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

Batch scan:

```bash
/opt/miniconda3/envs/pytorch/bin/python scripts/run_batch.py \
  --input-dir data/CE4 \
  --pattern "*.2C" \
  --visualize \
  --viz-max-blocks 1 \
  --viz-max-levels 2
```

## Stages

- Stage 01 input matrix: raw time-channel data, with injection truth boxes when
  available.
- Stage 02 SWT log-power: SWT detail-power map before local robust S/N.
- Stage 03 local robust S/N: median/MAD-normalized score map and threshold
  contour.
- Stage 04 candidate overlay: connected-component boxes on the S/N map.
- Stage 05 candidate review overview: candidate positions colored by veto
  status.
- Stage 06 validation/statistics overview: q-values and refined periods when
  validation rows exist.
- Stage 07 injection recovery: raw, after-veto, and validated recovery rates
  when injection rows exist.
- Stage 08 injection period recovery: injected period versus refined period.

## Controls

- `--viz-max-blocks`: maximum frequency blocks to render. Use `0` to render all
  blocks.
- `--viz-max-levels`: maximum SWT levels to render per block. Use `0` to render
  all levels.
- `--viz-top-candidates`: maximum candidates drawn in overview plots.
- `--viz-dpi`: PNG resolution.

Visualization can be expensive for full files. Use small time/frequency windows
first, then increase block and level limits once the configuration is stable.
