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
            "scalogram_filter": "scipy.ndimage.gaussian_filter1d",
            "region_labeler": "scipy.ndimage.label",
        },
    }
