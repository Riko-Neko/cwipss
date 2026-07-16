from __future__ import annotations

import platform

import numpy as np
import pywt
import scipy

from .signal.windows import native_pelt_available


def runtime_info() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pywavelets": pywt.__version__,
        "scipy": scipy.__version__,
        "local_filter": {
            "enabled": True,
            "activity_compressor": "Calibrated Persistent Ridge Occupancy (CPRO)",
            "window_detector": "native C++ PELT mean-shift segmentation",
            "period_filter": "Concentrated Periodic Ridge Filter (CPRF)",
            "native_pelt_required": True,
            "native_pelt_available": native_pelt_available(),
        },
    }
