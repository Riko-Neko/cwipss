# Calibrated Persistent Ridge Occupancy

Calibrated Persistent Ridge Occupancy (CPRO) compresses one channel's absolute
`period x time` CWT power map into a 1D time activity axis for native PELT
segmentation. Physical frequency channels are independent.

For raw series `x(t)`, first-difference MAD estimates white-noise sigma:

```text
sigma = 1.4826 * MAD(diff(x)) / sqrt(2)
```

For period-dependent unit-noise CWT gain `g(p)` and absolute CWT power `P(p,t)`:

```text
C(p,t) = P(p,t) / (sigma^2 * g(p))
tau = max(32, quantile(C, 0.9375))
```

A period ridge must exceed `tau`, have center-to-sideband contrast at least
`1.5` using center/context widths `3/15` bins, occupy at least `0.65` of a
65-record support, and cover 3 contiguous period bins. A second 769-record
consensus requires occupancy `0.40`.

The score map retains the occupied mean in calibrated-power units. Its maximum
over period is the 1D activity axis. `cpro_activity` stops here: it does not fill
time gaps, delete short runs, or define windows. The required native C++
mean-shift PELT implementation alone converts the activity axis into time
segments. A boundary-only minimum-duration gate may reject a PELT window before
stage 3.

In the strict performance benchmark, each accepted PELT window sends only its
indices to stage 3. Stage 3 reads the original unmasked absolute CWT power; it
never consumes the CPRO score map, ridge mask, or a CPRO-generated window.
CPRF calibration has its own recorded threshold parameters. Its selected
production gates are concentration `0.55`, local contrast `3.60`, and
integrated strength `0.0`.

## Frozen evidence

The stage-1 rank screened 366 strict single-channel `O(P*T)` activity
candidates, then tested the finalists on 100 configured weak-signal groups and
two independent CE4 observations. The frozen CPRO activity configuration uses
`window_support_records=769`.

- 2021 observation: 89% group recall; 7 PELT windows over 23 sampled real
  no-injection channels.
- 2019 observation: 86% group recall; 6 PELT windows over the same 23-channel
  design.
- The former 385-record baseline produced 53 and 66 windows respectively.

The authoritative runs are:

```text
runs/cpro_activity_pelt_full100_neg23_20211205_v3_production_noise
runs/cpro_activity_pelt_full100_neg23_20190830_v3_production_noise
```

## CPU and CUDA

`cpro.py` and `cpro_cuda.py` implement the same scientific operations. Explicit
CUDA mode fails when CUDA/CuPy is unavailable; it never invokes the CPU CPRO
implementation as a fallback.

CUDA keeps these products on device:

- absolute CWT power;
- calibrated power, contrast, exceedance, and occupancy maps;
- persistent ridge and long-window masks;
- 2D score map and period-to-time compression.

Only 1D activity/occupancy arrays and calibration scalars transfer to CPU before
PELT. No 2D CWT/CPRO map crosses this boundary. Returned PELT window indices
select slices from the still-resident unmasked CWT, and `cprf_cuda.py` completes
normalization, compression, peak-family metrics, and acceptance on the GPU.
Only the final CPRF scalar pack crosses back to CPU.

CUDA orchestration is explicitly split into prepare, native-PELT future, and
finalize stages. `cuda_max_pending_blocks` bounds how many GPU blocks may remain
resident while PELT runs; its reproducible default is `2`, allowing one block's
native PELT to overlap the next block's GPU work.
