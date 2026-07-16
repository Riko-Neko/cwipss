from __future__ import annotations

from pathlib import Path

import numpy as np

from cwipss.reporting.visualization import (
    CWTVisualizationConfig,
    SearchVisualizationConfig,
    visualize_cwt_stages,
)


def test_visualize_cwt_stages_writes_index_and_pngs(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    data = rng.normal(size=(64, 9)).astype(np.float32)
    freqs = np.arange(9, dtype=np.float64)
    candidates = [
        {
            "candidate_id": 1,
            "block_id": "block_0001",
            "channel": 2,
            "t0_rec": 10,
            "t1_rec": 18,
            "p0_rec": 8,
            "p1_rec": 8,
            "period_rec": 8,
            "t_peak_rec": 12,
            "freq_mhz": 2.0,
            "score": 6.0,
            "candidate_status": "needs_validation",
        }
    ]
    truths = [
        {
            "record_start": 8,
            "record_stop": 20,
            "period_records": 8,
            "freq_start_mhz": 2.0,
            "freq_stop_mhz": 4.0,
            "channel_start": 2,
            "channel_stop": 4,
        }
    ]

    index_path = visualize_cwt_stages(
        data,
        freqs,
        tmp_path / "viz",
        SearchVisualizationConfig(
            wavelet="cmor1.5-1.0",
            periods=np.geomspace(2, 16, 8),
            block_channels=9,
            candidate_period_min_records=2.0,
            candidate_period_max_records=16.0,
        ),
        raw_candidates=candidates,
        reviewed_candidates=candidates,
        truths=truths,
        injection_results=[{"detected_raw": True, "detected_after_veto": True, "validated": False, "period_records": 8, "refined_period_records": 4, "failure_stage": "period_mismatch"}],
        config=CWTVisualizationConfig(enabled=True, max_blocks=1, max_channels=1, top_candidates=5, dpi=80),
    )

    assert index_path.exists()
    assert (tmp_path / "viz" / "stage_01_input_matrix.png").exists()
    assert (tmp_path / "viz" / "stage_10_injection_recovery.png").exists()
    assert "Stage 04 Activity Windows" in index_path.read_text()
