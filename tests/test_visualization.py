from __future__ import annotations

from pathlib import Path

import numpy as np

from ce4_period_search.visualization import (
    SearchVisualizationConfig,
    VisualizationConfig,
    visualize_matrix_stages,
)


def test_visualize_matrix_stages_writes_index_and_pngs(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    data = rng.normal(size=(64, 8)).astype(np.float32)
    freqs = np.arange(8, dtype=np.float64)
    candidates = [
        {
            "candidate_id": 1,
            "block_id": "block_0001",
            "swt_level": 2,
            "record_start": 10,
            "record_stop": 18,
            "freq_start_mhz": 2.0,
            "freq_stop_mhz": 3.0,
            "peak_record": 12,
            "peak_freq_mhz": 2.0,
            "peak_score": 6.0,
            "candidate_status": "needs_validation",
        }
    ]
    truths = [
        {
            "record_start": 8,
            "record_stop": 20,
            "freq_start_mhz": 2.0,
            "freq_stop_mhz": 4.0,
        }
    ]

    index_path = visualize_matrix_stages(
        data,
        freqs,
        tmp_path / "viz",
        SearchVisualizationConfig(wavelet="haar", levels=2, block_channels=8, threshold=2.0, local_time=17, local_freq=5),
        raw_candidates=candidates,
        reviewed_candidates=candidates,
        truths=truths,
        injection_results=[{"detected_raw": True, "detected_after_veto": True, "validated": False, "period_records": 8, "refined_period_records": 4, "failure_stage": "period_mismatch"}],
        config=VisualizationConfig(enabled=True, max_blocks=1, max_levels=1, top_candidates=5, dpi=80),
    )

    assert index_path.exists()
    assert (tmp_path / "viz" / "stage_01_input_matrix.png").exists()
    assert (tmp_path / "viz" / "stage_07_injection_recovery.png").exists()
    assert "Stage 04 Candidate Overlay" in index_path.read_text()
