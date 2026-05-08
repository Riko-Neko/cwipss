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
            "median_filter": "scipy.ndimage.median_filter",
            "component_labeler": "scipy.ndimage.label",
        },
    }
