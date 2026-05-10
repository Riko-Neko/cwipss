from __future__ import annotations

from ce4_period_search.runtime import runtime_info


def test_runtime_info_records_required_filter_stack() -> None:
    info = runtime_info()

    assert info["numpy"]
    assert info["pywavelets"]
    assert info["scipy"]
    assert info["local_filter"]["enabled"] is True
    assert info["local_filter"]["noise_floor"] == "numpy.partition low-fraction mean"
    assert info["local_filter"]["window_detector"] == "internal PELT mean-shift"
