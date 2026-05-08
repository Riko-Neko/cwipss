from __future__ import annotations

import numpy as np

from ce4_period_search.detection import summarize_components
from ce4_period_search.swt import swt_detail_power_matrix


def test_swt_detail_power_shape() -> None:
    rng = np.random.default_rng(42)
    data = rng.normal(size=(257, 3)).astype(np.float32)
    power, levels = swt_detail_power_matrix(data, wavelet="db2", levels=3)
    assert power.shape == (3, 257, 3)
    assert levels.tolist() == [3, 2, 1]
    assert np.isfinite(power).all()


def test_component_summary_detects_patch() -> None:
    score = np.zeros((64, 8), dtype=np.float32)
    score[10:16, 2:4] = 7.0
    freqs = np.linspace(38.0, 39.0, score.shape[1])
    rows = summarize_components(
        score=score,
        freqs_mhz=freqs,
        record_start=100,
        level_number=2,
        threshold=5.0,
        min_pixels=4,
    )
    assert len(rows) == 1
    assert rows[0]["record_start"] == 110
    assert rows[0]["record_stop"] == 116
    assert rows[0]["area_pixels"] == 12
