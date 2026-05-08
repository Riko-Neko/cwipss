# SWT Period Search Pipeline

## Claim Discipline

A bright point or bright band in an SWT power map is a candidate, not a signal
claim. It becomes a credible periodicity candidate only after it survives:

- local robust S/N thresholding;
- component-level shape statistics;
- broadband / fixed-channel / edge veto;
- validation in the original time series;
- null tests for the scan's multiple-comparison burden.

## First Production Target

Run 1D SWT along the time axis for each channel in a time-channel matrix. This
preserves the channel interpretation and avoids the higher memory multiplier of
a full 2D SWT. The channel axis is used during post-processing to identify
localized components and reject broadband interference.

CE-4 LFRS `.2C/.2CL` support is an application adapter for the first science
target. It is not part of the core SWT period search definition.

## Data Flow

```text
dynamic spectrum / time-channel source
  -> application reader
  -> selected time/channel block
  -> robust per-channel normalization
  -> time-axis SWT detail coefficients
  -> log power maps per SWT level
  -> SciPy local median / MAD S/N
  -> SciPy connected components
  -> raw candidate table
  -> auditable veto labels
  -> original time-series validation evidence
  -> config.resolved.json + manifest.csv + candidates_raw.csv + candidates_reviewed.csv + summary.json
```

## Candidate Fields

Each connected component stores:

- SWT level and approximate scale in records;
- channel coordinate range and peak channel coordinate;
- record range and peak record;
- area, duration, channel span;
- peak and mean local S/N.

The machine-readable schema is documented in `docs/schema.md`. The raw table is
still a candidate-generation product only. Veto rules append review status in
`candidates_reviewed.csv` without overwriting raw candidate facts.

The validation stage is run separately with `scripts/run_validation.py`. It
extracts a candidate channel span from the original source data, searches
around the approximate SWT scale, and writes ACF, periodogram, folding, and
shuffle evidence to `validation_summary.csv`.

## Known Limitations Of The Current Prototype

- Components crossing frequency-block boundaries may be split.
- Local median/MAD uses a rectangular window; production scans should compare
  multiple window sizes.
- SciPy is a required dependency; there is no global-baseline fallback for
  production scans.
- SWT level is a scale index, not a precise physical period. Candidate periods
  must be refined later with folding, autocorrelation, or periodograms.
- Validation currently searches integer record periods near the SWT scale.
