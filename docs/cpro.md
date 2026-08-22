# Calibrated Persistent Ridge Occupancy

Calibrated Persistent Ridge Occupancy (CPRO) compresses one physical
frequency channel's `period x time` CWT power map into one continuous 1D time
axis. Physical frequency channels remain independent.

For raw series `x(t)`, first-difference MAD estimates white-noise sigma:

```text
sigma = 1.4826 * MAD(diff(x)) / sqrt(2)
```

For period-dependent unit-noise CWT gain `g(p)` and CWT power `P(p,t)`:

```text
C(p,t) = P(p,t) / (sigma^2 * g(p))
tau = max(32, quantile(C, 0.9375))
```

CPRO measures continuous proposal evidence around the same calibrated-power,
period-contrast, short-occupancy, and long-occupancy reference levels:

```text
soft_power = tau * sP * softplus(log(C / tau) / sP)
soft_contrast = sigmoid(log(R / Rmin) / sR)
soft_occupancy = sigmoid((occupancy - occupancy_min) / sO)
shape_activity = Top3Mean(time/period-supported soft evidence)
```

Defaults are `Rmin=1.5`, short support `65`, short occupancy `0.65`, period
support `3` bins, long support `769`, long occupancy `0.40`, and softness
`sP/sR/sO=0.50/0.25/0.10`. The hard exceedance map is used only to estimate
occupancy inside this continuous formula; it is not emitted as another
detector output.

CPRO does not fill time gaps, delete short runs, or define windows. The required
native C++ mean-shift PELT implementation converts only `shape_activity` into
time-window indices. PELT applies its standardized activity and duration
conditions; there is no absolute-activity gate.

CPRF receives each PELT window's indices and evaluates the original, unmasked
CWT2D slice. Therefore absolute power, ridge strength, concentration,
persistence, local contrast, and harmonic diagnostics have one owner: CPRF.
The selected production gates are persistence `0.40`, concentration `0.50`,
local contrast `1.20`, and integrated strength `0.0`.

## CPU and CUDA

`cpro.py` and `cpro_cuda.py` implement the same operations. Explicit CUDA mode
fails when CUDA/CuPy is unavailable; it never invokes CPU CPRO as a scientific
fallback.

CUDA retains CWT2D and all CPRO maps on device. Only the 1D `shape_activity`
axis and calibration scalars cross to CPU for native PELT. Returned window
indices select slices from the still-resident CWT2D, and `cprf_cuda.py`
completes normalization and acceptance on GPU. Only final CPRF scalars cross
back to CPU.

CUDA orchestration is split into prepare, native-PELT future, and finalize
stages. `cuda_max_pending_blocks=2` permits native PELT for one block to overlap
the next block's GPU work while bounding resident memory.

## Tuned working point

The current parameters were selected on the same 100-group weak injection set
and independently checked against two CE4 observations. Each observation also
contributed 90 real no-injection channels.

| Observation | PELT group coverage | CPRF recall, period error <=10% | CPRF recall, period error <=50% | Retained real negative windows |
|---|---:|---:|---:|---:|
| 2021-12-05 | 84/100 | 59/100 | 60/100 | 0/1162 |
| 2019-08-30 | 82/100 | 51/100 | 52/100 | 0/1029 |

The selected CPRF gates are `min_band_persistence=0.40`,
`min_band_concentration=0.50`, `min_local_contrast=1.20`, and
`min_integrated_strength=0.0`. Lowering contrast recovers weak ridges while the
persistence and concentration gates preserve separation from the sampled real
negative windows.

Changing native PELT endpoint stride from `1` to `8` preserved the above PELT
coverage and reduced combined PELT time over the 180 negative channels from
about `77.3 s` to `2.33 s` in the CPU benchmark.

Authoritative reproducible outputs:

```text
runs/cpro_shape_pipeline_tuned_full100_neg90_2021_v1
runs/cpro_shape_pipeline_tuned_full100_neg90_2019_v1
```
