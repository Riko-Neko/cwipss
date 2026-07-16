from __future__ import annotations

import inspect

from cwipss.signal import cpro_cuda, detection_cuda
from cwipss.workflows import search


def test_cpro_core_has_no_host_array_transfer() -> None:
    source = inspect.getsource(cpro_cuda)
    assert "asnumpy" not in source


def test_cuda_orchestration_keeps_cwt_resident_through_cprf() -> None:
    source = inspect.getsource(detection_cuda.prepare_block_period_chunks_cuda_power)
    assert "cp.asnumpy(result.activity)" in source
    assert "cp.asnumpy(result.window_occupancy)" in source
    assert "cp.asnumpy(result.score_map)" not in source
    assert "cp.asnumpy(result.occupancy_map)" not in source
    assert "cp.asnumpy(power" not in source
    assert "power_map=valid_power[:, :, target]" in source
    assert "cprf_normalization_threshold_cuda" in source

    pelt_source = inspect.getsource(detection_cuda.run_prepared_cuda_pelt)
    assert "pelt_mean_shift_batch" in pelt_source

    finalize_source = inspect.getsource(detection_cuda.finalize_prepared_cuda_period_chunks)
    assert "evaluate_cprf_cuda" in finalize_source
    assert "cp.asnumpy" not in finalize_source
    assert "channel.power_map = None" in finalize_source

    workflow_source = inspect.getsource(search.run_cwt_search)
    assert "pelt_executor.submit(run_prepared_cuda_pelt" in workflow_source
    assert "pending_blocks.popleft()" in workflow_source
    assert search.CWTSearchConfig().cuda_max_pending_blocks == 2
