# FRCR

Frequency-Referenced Coherent Ridge (FRCR) is a small detector core for finding
a periodic ridge that is strong in one physical frequency channel but absent
from nearby reference channels. It is useful for single-channel or narrowband
signals whose main discriminant is frequency contrast.

FRCR is intentionally independent from the CWIPSS production pipeline. It does
not provide CWT generation, PELT windowing, candidate tables, plotting, or file
I/O. It is not a channel-independent compression method: changing neighboring
physical frequency channels can change its result.

## Install

```bash
python -m pip install -e packages/frcr
```

Install the CuPy build matching the local CUDA runtime to use `frcr.cuda`.

## Input and output

`power_cube` has shape `(period, time, physical_frequency_channel)`. `periods`
contains one period in records per first-axis row. The target channel is an
integer index into the final axis.

```python
from frcr import FRCRParameters, frcr_channel

result = frcr_channel(power_cube, periods, target_channel=12, params=FRCRParameters())
score_map = result.score_map  # period x time
activity = result.activity    # time
activity_z = result.activity_z
```

CUDA keeps the two-dimensional score map and one-dimensional activity arrays on
the device:

```python
from frcr.cuda import frcr_channel_cuda

result_gpu = frcr_channel_cuda(power_cube_gpu, periods, target_channel=12)
```

The default parameters use 8 nearest non-guard reference channels, require
positive temporal support across 8 candidate cycles, apply a native score floor
of 0.20, clip signed log contrast to 1.5, and average the strongest 3 period rows
into the time activity axis.
