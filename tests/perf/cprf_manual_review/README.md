# CPRF Manual Review Dataset

This retained performance-review environment contains all candidates selected by:

- `band_conc >= 0.30`
- `local_contrast >= 1.80`
- `ridge_int >= 0.0`
- non-vetoed candidates only

The selection is sorted by the current pipeline `score` and contains 1,993 candidates.

## Extract On The Server

Copy this directory to the server and run without arguments:

```bash
python extract_single_channel.py
```

The extractor reads only the selected frequency channel and candidate-centered time context. It writes one compressed archive plus metadata under `artifacts/`.

## Render Review Images

After copying `artifacts/single_channel_slices.npz` and `artifacts/metadata.json` back into this directory:

```bash
python render_review.py
```

Each candidate produces one combined raw-series and CWT image under `artifacts/review/`. To verify a small subset first:

```bash
python render_review.py --limit 10
```

Every reviewed row stores one or more extensible annotations in the JSON `intervals` column. Each
object contains `t0`, `t1`, `lc`, `rc`, `label`, and `conf`, so reliable, false, and uncertain
structures may coexist in one candidate without a separate row-level classification.

For faster review, start the local UI and open the printed address:

```bash
python label_review.py
```

The Chinese UI writes directly and atomically to the same `labels.csv`. Drag the blue left boundary
and red right boundary directly over the diagnostic image. Click the active boundary and use the
left/right arrow keys for one-record adjustments, or Shift plus an arrow for eight records. A new
`keep` case cannot be saved until every displayed interval has been confirmed. Use the add/delete
controls to mark multiple structures independently; clicking a line selects its interval for
classification, confidence, censor flags, and one-record keyboard adjustment. Each interval's
`conf` describes confidence in that classification and annotation, not signal strength.
Hover over the header info icon for keyboard controls: `N`/`M` add or remove an interval, `K`
confirms the current interval's existing boundaries, `A`/`D` toggle left or right censoring, and
`Q`/`E` classify the current interval as real or false
positive, and `F` classifies it as false positive and immediately saves and advances.
`1`/`2`/`3` assign low/medium/high confidence, and Space saves and advances. Shortcuts are disabled
while an input, select, or notes field has focus.

The collapsible review filter selects already annotated intervals by classification and confidence.
Selections within one group are OR conditions, while classification and confidence are combined
with AND. A case is included when any one of its intervals matches both groups; the matching
interval is selected automatically. Saving and moving forward rebuilds the filtered queue, so an
interval changed from medium to low confidence immediately leaves a medium-confidence review pass.

## Compare CPRO With Direct Reductions

The comparison is intentionally locked to these 1,993 extracted cases and refuses any archive
with a different case count:

```bash
python compare_activity.py
```

It compares the edge-preserving CPRO, direct period mean, and normalized period integral under the
same native PELT. CPRO calibrates absolute CWT power, applies soft absolute-power and local
period-contrast responses, averages three adjacent period bins, and takes the strongest period
response at each record. It performs no time smoothing: PELT alone determines persistence and
time boundaries. Outputs are written to `artifacts/activity_comparison/`. `summary.json` sets
`formal_metrics_ready` only for the complete 1,993-case labelled dataset; `--limit` is smoke-only.

For labelled `keep` cases the report includes any-overlap recall, IoU recall at 0.10/0.30/0.50,
union coverage of the manual truth window, median IoU, signed left/right boundary bias, left/right
MAE, and paired IoU wins/ties/losses against direct mean, normalized trapezoidal period integral,
and the production CPRO. `fp` cases report the fraction that still produce one or more PELT
windows. Mean and integral are independent baselines because the logarithmic period grid is
nonuniform.

Algorithm selection prioritizes intervals labelled `keep` with `high` confidence. Their metrics use
the `priority_target_` prefix and paired CPRO comparisons are reported under
`priority_paired_target_comparisons`. On the fixed set, CPRO reaches median IoU `0.901`, median
boundary absolute error `40` records, any-overlap recall `99.36%`, IoU>=0.5 recall `84.51%`, and
median truth coverage `94.27%`. Direct integral reaches `0.817`, `61`, `99.04%`, `76.99%`, and
`80.54%`, respectively. The original `target_` metrics retain every Real interval as a secondary
robustness check; confidence never changes whether an interval is Real or FP.

Render a separate boundary-comparison gallery without changing the original review images:

```bash
python render_review.py \
  --comparison-windows artifacts/activity_comparison/windows.csv \
  --output-dir artifacts/comparison_review
```

Add `--confidence high --limit 30` for a focused high-confidence Real audit. Solid green lines are
manual Real boundaries; blue dashed boundaries are CPRO, coral boundaries are direct mean, and
ochre boundaries are normalized direct integral. The gray region is the original candidate window.
For each manual Real interval, the gallery shows only each algorithm's highest-IoU PELT window;
`windows.csv` retains every produced window.

## Validate The Production Continuity Gate

The final cutoff-free CPRO gate must be replayed through the functions imported
from `src`, not inferred only from an exported feature table:

```bash
python compare_activity.py \
  --production-continuity \
  --assert-production-target \
  --output-dir artifacts/production_continuity_validation_v1
```

This requires all 1,993 labelled cases and fails if the selected frontier
regresses. The current production result is `304/313 = 97.12%` high-confidence
Real recall, `68/816 = 8.33%` FP-interval response, and `61/605 = 10.08%`
pure-FP retention. Even/odd review-rank recall is `96.05%/98.14%`, and median
IoU on fully observed high-confidence Real cases is `0.9115`. The production
branch uses PELT minimum size `64` and no independent duration gate; the
historical `96/640` values in the same report belong only to the comparison
baseline.
