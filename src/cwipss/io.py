from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np


CE4_RECORD_LEN = 8287
CE4_NCHANS = 2048
PDS_NS = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}


@dataclass(frozen=True)
class SpectrumBlock:
    freqs_mhz: np.ndarray
    data: np.ndarray
    record_range: tuple[int, int]
    channel_range: tuple[int, int]


class SpectrumReader(Protocol):
    filename: Path
    n_records: int
    n_channels: int
    freqs_mhz: np.ndarray
    tsamp_seconds: float

    def info(self) -> dict:
        ...

    def freq_slice(self, f_start: float | None = None, f_stop: float | None = None) -> slice:
        ...

    def record_slice(self, t_start: int | None = None, t_stop: int | None = None) -> slice:
        ...

    def read_block(self, record_slice: slice, channel_slice: slice) -> SpectrumBlock:
        ...

    def iter_frequency_blocks(
        self,
        f_start: float | None = None,
        f_stop: float | None = None,
        t_start: int | None = None,
        t_stop: int | None = None,
        block_channels: int = 128,
    ) -> Iterator[SpectrumBlock]:
        ...


def read_ce4_2c(path: str | Path, nrec: int | None = None) -> np.memmap:
    """Return a CE4 `.2C` memmap. `mm["spec"]` has shape `(records, 2048)`."""
    dt = np.dtype(
        [
            ("frame_id", ("u1", 4)),
            ("version", "u1"),
            ("work_param", ("u1", 71)),
            ("solar_el", ">f4"),
            ("solar_az", ">f4"),
            ("cancel_period", ">u4"),
            ("cancel_num", ">u2"),
            ("accum_times", ">u2"),
            ("rec_len", ">u2"),
            ("spec", (">f4", CE4_NCHANS)),
            ("quality", "u1"),
        ],
        align=False,
    )
    if dt.itemsize != CE4_RECORD_LEN:
        raise ValueError(f"dtype itemsize={dt.itemsize} != record length={CE4_RECORD_LEN}")
    mm = np.memmap(path, dtype=dt, mode="r")
    return mm if nrec is None else mm[:nrec]


def match_2cl_for_2c(path_2c: str | Path, data_dir: str | Path | None = None) -> str | None:
    path_2c = Path(path_2c)
    data_dir = path_2c.parent if data_dir is None else Path(data_dir)
    stem = path_2c.stem
    candidates = list(data_dir.glob(stem + "*.2CL")) + list(data_dir.glob(stem + "*.xml"))
    candidates = [x for x in candidates if not x.name.startswith("._")]
    if candidates:
        return str(sorted(candidates, key=lambda x: len(x.name))[0])

    all_labels = glob.glob(str(data_dir / "*.2CL")) + glob.glob(str(data_dir / "*.xml"))
    all_labels = [x for x in all_labels if not Path(x).name.startswith("._")]
    if not all_labels:
        return None

    source_tokens = set(stem.split("_"))
    return max(all_labels, key=lambda x: len(source_tokens & set(Path(x).stem.split("_"))))


def _parse_zulu_datetime(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1]
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def infer_dt_from_2cl(path_2cl: str | None, nrec: int) -> float:
    if path_2cl is None:
        return 1.0
    try:
        root = ET.parse(path_2cl).getroot()
        start_node = root.find(".//pds:Time_Coordinates/pds:start_date_time", PDS_NS)
        stop_node = root.find(".//pds:Time_Coordinates/pds:stop_date_time", PDS_NS)
        if start_node is None or stop_node is None or start_node.text is None or stop_node.text is None:
            return 1.0
        start = _parse_zulu_datetime(start_node.text)
        stop = _parse_zulu_datetime(stop_node.text)
        dt = (stop - start).total_seconds() / float(nrec)
        return float(dt) if dt > 0 else 1.0
    except Exception:
        return 1.0


def infer_freq_axis_from_2cl(path_2cl: str | None, nchans: int) -> tuple[np.ndarray, dict | None]:
    if path_2cl is None:
        return np.arange(nchans, dtype=np.float64), None
    try:
        root = ET.parse(path_2cl).getroot()
        bands = root.find(".//pds:Instrument_Parm/pds:bands", PDS_NS)
        if bands is None or bands.text is None:
            return np.arange(nchans, dtype=np.float64), None

        raw = bands.text.strip()
        unit = (bands.attrib.get("unit") or "").strip().lower()
        unit_factor = {
            "hz": 1e-6,
            "khz": 1e-3,
            "mhz": 1.0,
            "ghz": 1e3,
        }.get(unit, 1.0)
        if "-" not in raw:
            return np.arange(nchans, dtype=np.float64), {"raw": raw, "unit": unit}

        lo, hi = [float(x.strip()) * unit_factor for x in raw.split("-", 1)]
        return np.linspace(lo, hi, nchans, dtype=np.float64), {"raw": raw, "unit": unit}
    except Exception:
        return np.arange(nchans, dtype=np.float64), None


class CE4Reader:
    """Reader for the currently supported CE4 `.2C/.2CL` input format."""

    def __init__(
        self,
        filename: str | Path,
        xml_path: str | Path | None = None,
        freq_start_mhz: float | None = None,
        freq_stop_mhz: float | None = None,
        ascending: bool = True,
    ) -> None:
        self.filename = Path(filename)
        if self.filename.suffix.lower() != ".2c":
            raise ValueError(f"Expected a CE4 .2C input-format file, got: {filename}")
        self.path_2cl = str(xml_path) if xml_path is not None else match_2cl_for_2c(self.filename)
        self.mm = read_ce4_2c(self.filename)
        self.spec = self.mm["spec"]
        self.n_records = int(self.spec.shape[0])
        self.n_channels = int(self.spec.shape[1])
        self.file_size_bytes = os.path.getsize(self.filename)

        freqs, self.band_meta = infer_freq_axis_from_2cl(self.path_2cl, self.n_channels)
        if freq_start_mhz is not None and freq_stop_mhz is not None:
            freqs = np.linspace(float(freq_start_mhz), float(freq_stop_mhz), self.n_channels)
        if not ascending:
            freqs = freqs[::-1]
        self.freqs_mhz = np.asarray(freqs, dtype=np.float64)
        self.tsamp_seconds = infer_dt_from_2cl(self.path_2cl, self.n_records)

    def info(self) -> dict:
        return {
            "filename": str(self.filename),
            "label": self.path_2cl,
            "records": self.n_records,
            "channels": self.n_channels,
            "file_size_bytes": self.file_size_bytes,
            "freq_min_mhz": float(np.nanmin(self.freqs_mhz)),
            "freq_max_mhz": float(np.nanmax(self.freqs_mhz)),
            "tsamp_seconds": float(self.tsamp_seconds),
            "band_meta": self.band_meta,
        }

    def freq_slice(self, f_start: float | None = None, f_stop: float | None = None) -> slice:
        freqs = self.freqs_mhz
        if f_start is None and f_stop is None:
            return slice(0, freqs.size)
        lo = float(np.nanmin(freqs)) if f_start is None else float(f_start)
        hi = float(np.nanmax(freqs)) if f_stop is None else float(f_stop)
        lo, hi = sorted([lo, hi])
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            raise ValueError(f"No frequency channels found in range [{lo}, {hi}] MHz")
        idx = np.where(mask)[0]
        return slice(int(idx[0]), int(idx[-1]) + 1)

    def record_slice(self, t_start: int | None = None, t_stop: int | None = None) -> slice:
        start = 0 if t_start is None else int(t_start)
        stop = self.n_records if t_stop is None else int(t_stop)
        start = max(0, min(start, self.n_records))
        stop = max(start + 1, min(stop, self.n_records))
        return slice(start, stop)

    def read_block(self, record_slice: slice, channel_slice: slice) -> SpectrumBlock:
        raw = self.spec[record_slice, channel_slice]
        data = np.asarray(raw, dtype=np.float32)
        return SpectrumBlock(
            freqs_mhz=self.freqs_mhz[channel_slice].copy(),
            data=data,
            record_range=(int(record_slice.start), int(record_slice.stop)),
            channel_range=(int(channel_slice.start), int(channel_slice.stop)),
        )

    def iter_frequency_blocks(
        self,
        f_start: float | None = None,
        f_stop: float | None = None,
        t_start: int | None = None,
        t_stop: int | None = None,
        block_channels: int = 128,
    ) -> Iterator[SpectrumBlock]:
        records = self.record_slice(t_start, t_stop)
        selected = self.freq_slice(f_start, f_stop)
        start = int(selected.start)
        stop = int(selected.stop)
        block_channels = max(1, int(block_channels))
        for block_start in range(start, stop, block_channels):
            block_stop = min(block_start + block_channels, stop)
            yield self.read_block(records, slice(block_start, block_stop))


def open_spectrum_reader(path: str | Path) -> SpectrumReader:
    """Open an input file through the available Cwipss format adapters."""
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".2c":
        return CE4Reader(input_path)
    raise ValueError(
        "Unsupported input data format. Cwipss currently supports CE4 .2C/.2CL inputs; "
        "FilterBank support is planned for the same adapter layer."
    )
