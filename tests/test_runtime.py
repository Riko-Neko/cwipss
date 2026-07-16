from __future__ import annotations

from cwipss.runtime import runtime_info


def test_runtime_info_records_required_filter_stack() -> None:
    info = runtime_info()

    assert info["numpy"]
    assert info["pywavelets"]
    assert info["scipy"]
    assert info["local_filter"]["enabled"] is True
    assert info["local_filter"]["activity_compressor"] == "Calibrated Persistent Ridge Occupancy (CPRO)"
    assert info["local_filter"]["period_filter"] == "Concentrated Periodic Ridge Filter (CPRF)"
    assert info["local_filter"]["window_detector"] == "native C++ PELT mean-shift segmentation"
    assert info["local_filter"]["native_pelt_required"] is True
