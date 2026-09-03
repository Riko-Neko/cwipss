# CPSR-1993

CPSR-1993 is the CWIPSS CE4 Periodic-Structure Review Set. It contains 1,993
candidate-centered, single-frequency-channel excerpts from CE4 LFRS Level 2C
observations and manually reviewed time intervals.

The dataset is attached to the CWIPSS `v2.0.0` GitHub release as
`cpsr-1993-v1.0.0.zip`. The archive contains:

- `labels.csv`: manual interval annotations;
- `selection.csv`: candidate and pipeline metadata with source basenames;
- `metadata.json`: extraction geometry with source basenames;
- `single_channel_slices.npz`: 1,993 variable-length `float32` time series;
- `README.md`: schema, provenance, and usage notes;
- `LICENSE-ANNOTATIONS.txt`: annotation license;
- `SHA256SUMS`: checksums for every payload file.

Each `intervals` item in `labels.csv` contains `t0`, `t1`, `lc`, `rc`, `label`,
and `conf`. The labels have the following restricted meanings:

- `keep`: morphology consistent with the periodic structures targeted by the
  review; it is not confirmation of an astrophysical or extraterrestrial
  origin;
- `fp`: false candidate relative to that target morphology;
- `conf`: confidence in the manual classification and boundary placement, not
  signal strength;
- `lc` and `rc`: left- and right-censored interval boundaries.

The cases were selected from an earlier CWIPSS candidate run using
`band_conc >= 0.30`, `local_contrast >= 1.80`, `ridge_int >= 0`, and no veto.
They are therefore a candidate-conditioned benchmark, not a random CE4 sample,
and must not be used to estimate population prevalence.

Load a slice by its stable `raw_key`:

```python
import numpy as np

with np.load("single_channel_slices.npz", allow_pickle=False) as data:
    series = data["r0001"]
```

For leakage-safe model evaluation, group cases from the same source
observation together. Overlapping time intervals in nearby physical frequency
channels may represent one event and should also remain in the same split.

The manual annotations are released under CC BY 4.0. The included excerpts are
derived from CE4 LFRS scientific data provided by the China National Space
Administration (CNSA); original source-data rights and attribution requirements
remain with the provider.
