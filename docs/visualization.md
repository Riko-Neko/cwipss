# Stage Visualization

Stage visualization is optional and writes PNG diagnostics plus
`visualization/index.md`.

Runtime stages and post-validation candidate galleries share the same small
function-based plotting core:

- `heatmap()` renders matrix data and optional boxes, lines, or shaded ranges;
- `raw_view()` applies the standard raw time-frequency format;
- `cwt_view()` applies the standard period-time CWT format;
- `save_figure()` handles non-matrix summary plots.

The runtime and gallery code only select data and pass plotting parameters.
They do not maintain separate raw/CWT plotting implementations.

The modules are `cwipss.reporting.visualization`,
`cwipss.reporting.gallery`, and `cwipss.reporting.plotting`.

## Enable

```bash
python scripts/run_cwt_candidates.py \
  --input data/example.2C \
  --f-start 38.0 \
  --f-stop 38.3 \
  --t-start 0 \
  --t-stop 2048 \
  --visualize
```

## Stages

- Stage 01, `stage_01_input_matrix.png`: raw `time x channel` matrix.
- Stage 02, `stage_02_<block>_channel_<channel>_scalogram.png`:
  representative-channel full `period x time` CWT scalograms.
- Stage 03: representative-channel trusted-period structure-gated CWT map
  after single-channel low-20% floor normalization, per-period low-quantile
  standardization, and local 2D time-period support gating.
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
- `--structure-baseline-quantile`: low time-quantile background used before
  structure gating. Default `0.10`.
- `--structure-scale-quantile`: low time-quantile subset used for per-period
  scale estimation. Default `0.20`.
- `--structure-z-threshold`: robust z threshold for local 2D support.
  Default `1.0`.
- `--structure-time-support-records`: time-neighborhood support width.
  Default `64`.
- `--structure-period-support-bins`: period-neighborhood support width.
  Default `3`.
- `--structure-min-support-fraction`: minimum local support fraction retained.
  Default `0.10`.
- `--pelt-penalty`: PELT mean-shift penalty. Default `16`.
- `--window-min-duration-records`: minimum PELT window duration. Default `384`.
- `--window-min-activity-raw-mean`: minimum raw structured activity mean needed
  for a PELT window to emit period candidates. Default `25.0`.
- `--window-merge-gap-records`: merge nearby PELT windows. Default `256`.
- `--profile-min-prominence`: minimum period-profile peak prominence.
  Default `0.5`.
- `--viz-max-blocks`: maximum blocks to visualize; `0` renders all.
- `--viz-max-channels`: representative channels per block for period-time CWT
  scalograms; `0` renders all channels in each visualized block.
- `--viz-top-candidates`: maximum candidates drawn in overview plots.
- `--viz-dpi`: PNG resolution.

Use small time/channel windows first. Full-file visualization can be expensive.
Avoid low-score debug settings unless the goal is high-recall visual exploration
rather than candidate review.

## Per-Candidate Gallery

Runtime staged visualization selects representative blocks and channels; it
does not guarantee one raw/CWT pair for every Top-N candidate. Generate that
view after detection or validation from an existing single run or batch:

```bash
python scripts/run_candidate_gallery.py \
  --run-dir runs/<run_or_batch_id> \
  --top 100
```

The default `auto` ordering uses `evidence_rank` when reviewed validation
statistics exist, otherwise it uses `integrated_score`. Override this with
`--sort-by integrated_score` or `--sort-by global_q_value`. Use `--top 0` to
render every candidate and `--include-vetoed` to include rejected rows.

Images are grouped by type and reuse the former candidate-directory identifier
as their filename:

- `candidate_gallery/raw/0001_<run>_candidate_<id>.png`: raw candidate window;
- `candidate_gallery/cwt/0001_<run>_candidate_<id>.png`: matching CWT scalogram.

Candidate boxes are drawn on both views. The validation-refined period is drawn
on the CWT view when available. No separate filename-mapping CSV is generated.

If the result CSV contains server paths that are unavailable on the current
machine, point to a directory containing the same source basenames:

```bash
python scripts/run_candidate_gallery.py \
  --run-dir runs/<batch_id> \
  --source-root /path/to/CE4_LFRS_2C \
  --top 100
```

Outputs are written to `candidate_gallery/index.md`, `candidate_gallery/raw/`,
and `candidate_gallery/cwt/`. Existing unrelated or stale files are not
automatically removed. The index is created before rendering starts and updated
after each candidate, so completed entries remain browsable if a long run is
interrupted.

Useful gallery controls:

- `--context-periods`: target time context measured in candidate periods;
- `--min-window-records`, `--max-window-records`: time-window bounds;
- `--freq-context-channels`: raw channels shown on each side of the candidate;
- `--period-radius`: CWT period-axis factor around the candidate seed;
- `--cwt-backend`, `--cuda-device`: override the saved compute backend;
- `--output-dir`: write the gallery outside the source run directory.
