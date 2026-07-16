"""Shared plotting primitives for non-interactive review artifacts."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cwipss_matplotlib"))
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle


CANDIDATE_COLOR = "#39ff14"
TRUTH_COLOR = "#00d5ff"
VETO_COLOR = "#b23b2e"
CWT_POWER_CMAP = "inferno"
CWT_POWER_COLORBAR = "log10(CWT power)"
CWT_POWER_EPS = 1e-12
Box = tuple[float, float, float, float, str, str]
Line = tuple[str, float, str, str, str]


def number(row: Mapping[str, Any], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, default)
        return default if value in ("", None) else float(value)
    except (TypeError, ValueError):
        return default


def edges(values: np.ndarray, logarithmic: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        center = float(values[0]) if values.size else 0.5
        width = max(1e-6, abs(center) * 0.01, 1.0 if not logarithmic else 0.1)
        return np.array([center - width / 2, center + width / 2])
    result = np.empty(values.size + 1)
    result[1:-1] = np.sqrt(values[:-1] * values[1:]) if logarithmic else (values[:-1] + values[1:]) / 2
    result[0] = values[0] ** 2 / result[1] if logarithmic else 2 * values[0] - result[1]
    result[-1] = values[-1] ** 2 / result[-2] if logarithmic else 2 * values[-1] - result[-2]
    return result


def finite_percentile_limits(
    values: np.ndarray,
    percentiles: tuple[float, float] = (1.0, 99.0),
) -> tuple[float | None, float | None]:
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return None, None
    lo, hi = np.nanpercentile(finite, percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None, None
    if hi <= lo:
        center = float(lo)
        width = max(abs(center) * 1e-6, 1e-6)
        return center - width, center + width
    return float(lo), float(hi)


def cwt_power_display_values(power: np.ndarray) -> np.ndarray:
    return np.log10(np.asarray(power, dtype=np.float32) + CWT_POWER_EPS)


def row_boxes(
    rows: Iterable[Mapping[str, Any]],
    xkeys: tuple[str, ...],
    ykeys: tuple[str, ...],
    *,
    color: str,
    label: str,
    min_span: tuple[float, float] = (1.0, 1.0),
    limit: int = 100,
) -> list[Box]:
    result: list[Box] = []
    for row in list(rows)[:limit]:
        values = []
        for keys in (xkeys, ykeys):
            start = next((number(row, key) for key in keys if math.isfinite(number(row, key))), math.nan)
            stop = number(row, keys[1], start) if len(keys) > 1 else start
            values.extend((start, stop))
        if not all(math.isfinite(value) for value in values):
            continue
        x0, x1 = sorted(values[:2])
        y0, y1 = sorted(values[2:])
        if x1 <= x0:
            x0, x1 = x0 - min_span[0] / 2, x1 + min_span[0] / 2
        if y1 <= y0:
            y0, y1 = y0 - min_span[1] / 2, y1 + min_span[1] / 2
        result.append((x0, x1, y0, y1, color, label))
    return result


def save_figure(path: str | Path, dpi: int, draw: Callable[[plt.Axes], None]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    draw(ax)
    fig.savefig(path, dpi=max(72, dpi))
    plt.close(fig)
    return path


def heatmap(
    path: str | Path,
    values: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar: str,
    cmap: str = "magma",
    yscale: str | None = None,
    boxes: Iterable[Box] = (),
    lines: Iterable[Line] = (),
    shaded: Iterable[tuple[float, float]] = (),
    ylim: tuple[float, float] | None = None,
    dpi: int = 140,
) -> Path:
    values = np.asarray(values)
    limits = finite_percentile_limits(values)
    boxes, lines = list(boxes), list(lines)

    def draw(ax: plt.Axes) -> None:
        mesh = ax.pcolormesh(x_edges, y_edges, values, shading="auto", cmap=cmap, vmin=limits[0], vmax=limits[1])
        for lo, hi in shaded:
            ax.axhspan(lo, hi, color="#d9d9d9", alpha=0.18, linewidth=0)
        for x0, x1, y0, y1, color, _label in boxes:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.4))
        for axis, value, color, _label, style in lines:
            (ax.axvline if axis == "x" else ax.axhline)(value, color=color, linestyle=style, linewidth=1.1)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, yscale=yscale or "linear")
        ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
        ax.set_ylim(*(ylim or (float(y_edges[0]), float(y_edges[-1]))))
        handles, seen = [], set()
        for item in [*boxes, *lines]:
            color, label, style = (
                (item[2], item[3], item[4]) if len(item) == 5
                else (item[4], item[5], None)
            )
            if label in seen:
                continue
            seen.add(label)
            handles.append(
                plt.Line2D([], [], color=color, linestyle=style) if style
                else Patch(facecolor="none", edgecolor=color)
            )
            handles[-1].set_label(label)
        if handles:
            ax.legend(handles=handles, loc="best", fontsize="small")
        plt.colorbar(mesh, ax=ax, label=colorbar)

    return save_figure(path, dpi, draw)


def raw_view(
    path: str | Path,
    data: np.ndarray,
    freqs: np.ndarray,
    *,
    offset: int = 0,
    title: str,
    candidates: Iterable[Mapping[str, Any]] = (),
    truths: Iterable[Mapping[str, Any]] = (),
    dpi: int = 140,
) -> Path:
    step = float(np.nanmedian(np.abs(np.diff(freqs)))) if len(freqs) > 1 else 1.0
    boxes = row_boxes(
        candidates,
        ("freq_mhz",),
        ("t0_rec", "t1_rec"),
        color=CANDIDATE_COLOR,
        label="candidate",
        min_span=(step, 1),
    )
    boxes += row_boxes(
        truths,
        ("freq_start_mhz", "freq_stop_mhz"),
        ("record_start", "record_stop"),
        color=TRUTH_COLOR,
        label="truth",
        min_span=(step, 1),
    )
    return heatmap(
        path, data, edges(freqs), np.arange(offset, offset + len(data) + 1),
        title=title, xlabel="Frequency / MHz", ylabel="Record", colorbar="amplitude",
        boxes=boxes, dpi=dpi,
    )


def cwt_view(
    path: str | Path,
    power: np.ndarray,
    periods: np.ndarray,
    *,
    offset: int = 0,
    title: str,
    candidates: Iterable[Mapping[str, Any]] = (),
    truths: Iterable[Mapping[str, Any]] = (),
    refined: float = math.nan,
    ylim: tuple[float, float] | None = None,
    cmap: str = CWT_POWER_CMAP,
    colorbar: str = CWT_POWER_COLORBAR,
    log_power: bool = True,
    dpi: int = 140,
) -> Path:
    boxes = row_boxes(
        candidates,
        ("t0_rec", "t1_rec"),
        ("p0_rec", "p1_rec"),
        color=CANDIDATE_COLOR,
        label="candidate",
    )
    lines: list[Line] = [
        ("y", period, TRUTH_COLOR, "truth period", "--")
        for row in truths
        if math.isfinite(period := number(row, "period_records", number(row, "peak_period_records")))
    ]
    if math.isfinite(refined):
        lines.append(("y", refined, TRUTH_COLOR, "refined period", ":"))
    values = cwt_power_display_values(power) if log_power else np.asarray(power)
    return heatmap(
        path, values, np.arange(offset, offset + values.shape[1] + 1), edges(periods, True),
        title=title, xlabel="Record", ylabel="Period / records", colorbar=colorbar,
        cmap=cmap, yscale="log", boxes=boxes, lines=lines, ylim=ylim, dpi=dpi,
    )


class ImageIndex:
    def __init__(self, output_dir: str | Path, title: str, metadata: dict[str, Any] | None = None):
        self.output_dir, self.title = Path(output_dir), title
        self.metadata, self.items = metadata or {}, []

    def add(self, title: str, path: str | Path, note: str) -> None:
        self.items.append((title, Path(path), note))

    def write(self) -> Path:
        lines = [f"# {self.title}", "", "Diagnostic review products; not confirmed signal claims.", ""]
        for title, path, note in self.items:
            lines += [f"## {title}", "", note, "", f"![{title}]({path.relative_to(self.output_dir)})", ""]
        lines += ["## Metadata", "", "```json", json.dumps(self.metadata, indent=2, ensure_ascii=True), "```", ""]
        path = self.output_dir / "index.md"
        path.write_text("\n".join(lines))
        return path


# Backward-compatible names used by the runtime and gallery modules.
render_raw_time_frequency = raw_view
render_cwt_scalogram = cwt_view
VisualizationIndex = ImageIndex
