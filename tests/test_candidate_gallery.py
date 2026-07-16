from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from cwipss.reporting.gallery import (
    CandidateGalleryConfig,
    select_candidate_rows,
    visualize_candidate_gallery,
)
from cwipss.data.readers import SpectrumBlock


class FakeReader:
    def __init__(self, filename: str | Path) -> None:
        self.filename = Path(filename)
        self.n_records = 256
        self.n_channels = 16
        self.freqs_mhz = np.linspace(1.0, 2.0, self.n_channels)
        self.tsamp_seconds = 1.0
        time = np.arange(self.n_records, dtype=np.float32)
        data = np.random.default_rng(123).normal(0.0, 0.2, (self.n_records, self.n_channels))
        data[:, 7] += 2.0 * np.sin(2.0 * np.pi * time / 16.0)
        self.data = data.astype(np.float32)

    def read_block(self, record_slice: slice, channel_slice: slice) -> SpectrumBlock:
        return SpectrumBlock(
            freqs_mhz=self.freqs_mhz[channel_slice],
            data=self.data[record_slice, channel_slice],
            record_range=(int(record_slice.start), int(record_slice.stop)),
            channel_range=(int(channel_slice.start), int(channel_slice.stop)),
        )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_select_candidate_rows_prefers_evidence_rank() -> None:
    rows = [
        {"candidate_id": "1", "evidence_rank": "2", "score": "100"},
        {"candidate_id": "2", "evidence_rank": "1", "score": "10"},
        {"candidate_id": "3", "candidate_status": "vetoed", "evidence_rank": "0"},
    ]

    selected = select_candidate_rows(rows, top_n=2, sort_by="auto")

    assert [row["candidate_id"] for row in selected] == ["2", "1"]


def test_visualize_candidate_gallery_writes_raw_and_cwt_figure(tmp_path: Path) -> None:
    source = tmp_path / "example.2C"
    source.touch()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.resolved.json").write_text(
        json.dumps(
            {
                "input": {"path": str(source)},
                "scan": {
                    "wavelet": "cmor1.5-1.0",
                    "cwt_method": "fft",
                    "cwt_backend": "cpu",
                    "period_min_records": 2,
                    "period_max_records": 64,
                    "period_count": 24,
                    "period_spacing": "log",
                },
            }
        )
    )
    candidate_fields = [
        "run_id",
        "source_file",
        "candidate_id",
        "candidate_status",
        "t0_rec",
        "t1_rec",
        "dur_rec",
        "p0_rec",
        "p1_rec",
        "period_rec",
        "t_peak_rec",
        "freq_mhz",
        "score",
    ]
    _write_csv(
        run_dir / "candidates_reviewed.csv",
        candidate_fields,
        [
            {
                "run_id": "run_a",
                "source_file": str(source),
                "candidate_id": "1",
                "candidate_status": "needs_validation",
                "t0_rec": "64",
                "t1_rec": "192",
                "dur_rec": "128",
                "p0_rec": "14",
                "p1_rec": "18",
                "period_rec": "16",
                "t_peak_rec": "128",
                "freq_mhz": str(np.linspace(1.0, 2.0, 16)[7]),
                "score": "20",
            }
        ],
    )
    gallery = run_dir / "candidate_gallery"

    index = visualize_candidate_gallery(
        run_dir,
        config=CandidateGalleryConfig(
            top_n=1,
            min_window_records=128,
            max_window_records=128,
            freq_context_channels=2,
            dpi=80,
            cwt_backend="cpu",
        ),
        reader_factory=FakeReader,
    )

    assert index.exists()
    assert "Stage 01 Input Matrix" in index.read_text()
    assert "Stage 02 CWT Scalogram" in index.read_text()
    filename = "0001_run_a_candidate_1.png"
    assert [path.name for path in (gallery / "raw").glob("*.png")] == [filename]
    assert [path.name for path in (gallery / "cwt").glob("*.png")] == [filename]
    assert not (gallery / "gallery.csv").exists()

    interrupted = tmp_path / "interrupted_gallery"

    def interrupt(_path):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        visualize_candidate_gallery(
            run_dir,
            interrupted,
            config=CandidateGalleryConfig(top_n=1),
            reader_factory=interrupt,
        )
    assert (interrupted / "index.md").exists()
