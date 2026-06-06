from __future__ import annotations

from pathlib import Path
from threading import Event, enumerate as enumerate_threads
from types import SimpleNamespace

import numpy as np
import pytest

import cwipss.signal.cwt_cuda as cwt_cuda
import cwipss.signal.detection_cuda as detection_cuda
import cwipss.workflows.search as pipeline
from cwipss.config import CWTSearchConfig
from cwipss.data.readers import SpectrumBlock


class _FakeReader:
    filename = Path("fake.2C")
    n_records = 8
    n_channels = 4
    freqs_mhz = np.arange(4, dtype=np.float64)
    tsamp_seconds = 1.0

    def info(self) -> dict:
        return {
            "filename": str(self.filename),
            "label": "",
            "records": self.n_records,
            "channels": self.n_channels,
            "tsamp_seconds": self.tsamp_seconds,
            "freq_min_mhz": 0.0,
            "freq_max_mhz": 3.0,
        }

    def record_slice(self, start=None, stop=None) -> slice:
        return slice(0 if start is None else int(start), self.n_records if stop is None else int(stop))

    def freq_slice(self, start=None, stop=None) -> slice:
        return slice(0, self.n_channels)

    def read_block(self, record_slice: slice, channel_slice: slice) -> SpectrumBlock:
        channel_start = int(channel_slice.start)
        channel_stop = int(channel_slice.stop)
        data = np.full(
            (int(record_slice.stop - record_slice.start), channel_stop - channel_start),
            channel_start,
            dtype=np.float32,
        )
        return SpectrumBlock(
            freqs_mhz=self.freqs_mhz[channel_slice],
            data=data,
            record_range=(int(record_slice.start), int(record_slice.stop)),
            channel_range=(channel_start, channel_stop),
        )


class _FakeProgress:
    def __init__(self) -> None:
        self.updates: list[int] = []
        self.messages: list[str] = []
        self.closed = False

    def update(self, count: int) -> None:
        self.updates.append(int(count))

    def write(self, message: str) -> None:
        self.messages.append(message)

    def close(self) -> None:
        self.closed = True


def _run_fake_cuda_pipeline(
    tmp_path,
    monkeypatch,
    pending_blocks: int,
    *,
    timing: bool = False,
    progress: bool = False,
    fail_pelt: bool = False,
    progress_tracker: _FakeProgress | None = None,
) -> tuple[list[str], _FakeProgress]:
    events: list[str] = []
    second_prepared = Event()
    progress_tracker = progress_tracker or _FakeProgress()

    def fake_cwt(data, periods, **kwargs):
        block = int(data[0, 0])
        events.append(f"cwt:{block}")
        return np.zeros((len(periods), data.shape[0], data.shape[1]), dtype=np.float32)

    def fake_prepare(*, freqs_mhz, **kwargs):
        block = int(freqs_mhz[0])
        events.append(f"prepare:{block}")
        if block == 2:
            second_prepared.set()
        yield {"block": block}

    def fake_pelt(prepared_chunks):
        block = int(prepared_chunks[0]["block"])
        events.append(f"pelt_start:{block}")
        if fail_pelt and block == 0:
            raise RuntimeError("fake PELT failure")
        if pending_blocks == 2 and block == 0:
            assert second_prepared.wait(timeout=2.0)
        events.append(f"pelt_done:{block}")
        return [], 0.0

    def fake_finalize(prepared_chunks, segments_batch, **kwargs):
        block = int(prepared_chunks[0]["block"])
        events.append(f"finalize:{block}")
        return [], []

    monkeypatch.setattr(pipeline, "open_spectrum_reader", lambda path: _FakeReader())
    monkeypatch.setattr(pipeline, "_use_cuda_block_backend", lambda *args: True)
    monkeypatch.setattr(
        pipeline,
        "_channel_progress",
        lambda **kwargs: progress_tracker if kwargs["enabled"] else None,
    )
    monkeypatch.setattr(cwt_cuda, "cwt_power_cube_cuda_gpu", fake_cwt)
    monkeypatch.setattr(
        cwt_cuda,
        "_cupy",
        lambda: SimpleNamespace(
            cuda=SimpleNamespace(
                Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None))
            )
        ),
    )
    monkeypatch.setattr(detection_cuda, "prepare_block_period_chunks_cuda_power", fake_prepare)
    monkeypatch.setattr(detection_cuda, "run_prepared_cuda_pelt", fake_pelt)
    monkeypatch.setattr(detection_cuda, "finalize_prepared_cuda_period_chunks", fake_finalize)

    pipeline.run_cwt_search(
        CWTSearchConfig(
            input="fake.2C",
            output_dir=str(tmp_path),
            run_id=f"pending_{pending_blocks}",
            cwt_backend="cuda",
            block_channels=2,
            period_count=4,
            cuda_structure_batch=True,
            cuda_max_pending_blocks=pending_blocks,
            progress_enabled=progress,
            timing_enabled=timing,
            veto_enabled=False,
        )
    )
    return events, progress_tracker


def test_cuda_pending_one_finishes_before_next_block(tmp_path, monkeypatch) -> None:
    events, _progress = _run_fake_cuda_pipeline(tmp_path, monkeypatch, pending_blocks=1)

    assert events.index("finalize:0") < events.index("cwt:2")


def test_cuda_pending_two_prepares_next_block_before_finalize(tmp_path, monkeypatch) -> None:
    events, _progress = _run_fake_cuda_pipeline(tmp_path, monkeypatch, pending_blocks=2)

    assert events.index("prepare:2") < events.index("finalize:0")


def test_cuda_async_timing_and_progress_are_reported_per_block(tmp_path, monkeypatch) -> None:
    _events, progress = _run_fake_cuda_pipeline(
        tmp_path,
        monkeypatch,
        pending_blocks=2,
        timing=True,
        progress=True,
    )

    assert progress.updates == [2, 2]
    assert progress.closed is True
    timing_messages = [message for message in progress.messages if message.startswith("[CWT TIMING]")]
    assert len(timing_messages) == 2
    assert all("pelt=" in message and "pelt_wait=" in message for message in timing_messages)


def test_cuda_async_failure_closes_progress_and_executor(tmp_path, monkeypatch) -> None:
    progress = _FakeProgress()

    with pytest.raises(RuntimeError, match="fake PELT failure"):
        _run_fake_cuda_pipeline(
            tmp_path,
            monkeypatch,
            pending_blocks=1,
            progress=True,
            fail_pelt=True,
            progress_tracker=progress,
        )

    assert progress.closed is True
    assert not any(thread.name.startswith("cwipss-pelt") for thread in enumerate_threads())


def test_pipeline_requires_native_pelt_before_opening_input(tmp_path, monkeypatch) -> None:
    reader_opened = False

    def fail_native_check() -> None:
        raise RuntimeError("native PELT required")

    def fake_open_reader(path):
        nonlocal reader_opened
        reader_opened = True
        return _FakeReader()

    monkeypatch.setattr(pipeline, "require_native_pelt", fail_native_check)
    monkeypatch.setattr(pipeline, "open_spectrum_reader", fake_open_reader)

    with pytest.raises(RuntimeError, match="native PELT required"):
        pipeline.run_cwt_search(
            CWTSearchConfig(
                input="fake.2C",
                output_dir=str(tmp_path),
                progress_enabled=False,
            )
        )

    assert reader_opened is False
