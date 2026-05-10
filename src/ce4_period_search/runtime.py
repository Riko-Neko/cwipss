from __future__ import annotations

import platform

import numpy as np
import pywt
import scipy


def runtime_info() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pywavelets": pywt.__version__,
        "scipy": scipy.__version__,
        "local_filter": {
            "enabled": True,
            "noise_floor": "numpy.partition low-fraction mean",
            "activity_smoother": "scipy.ndimage.uniform_filter1d",
            "window_detector": "internal PELT mean-shift",
        },
    }
