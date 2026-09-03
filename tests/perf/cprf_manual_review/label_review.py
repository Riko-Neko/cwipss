#!/usr/bin/env python3
"""Serve a local review UI for the fixed 1,993-case CPRF dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
EXPECTED_CASES = 1993
EDITABLE_FIELDS = (
    "notes",
    "intervals",
)
GEOMETRY_VERSION_FIELD = "label_geometry_version"
GEOMETRY_VERSION = "4"
LEGACY_INTERVAL_FIELDS = (
    "truth_t0_rec",
    "truth_t1_rec",
    "truth_left_censored",
    "truth_right_censored",
    "fp_intervals",
)
LEGACY_CLASS_FIELDS = ("label", "label_confidence")
VALID_INTERVAL_LABELS = {"", "keep", "fp", "uncertain"}
VALID_CONFIDENCE = {"", "low", "medium", "high"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=BASE_DIR / "selection.csv")
    parser.add_argument("--labels", type=Path, default=BASE_DIR / "labels.csv")
    parser.add_argument("--image-dir", type=Path, default=BASE_DIR / "artifacts/review")
    parser.add_argument("--metadata", type=Path, default=BASE_DIR / "artifacts/metadata.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--check", action="store_true", help="Validate inputs and exit")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def detect_plot_geometry(path: Path) -> dict[str, int]:
    """Locate the axes and the actual colored data extent in image pixels."""
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    y0, y1 = 40, min(280, height)
    dark = np.all(rgb[y0:y1] < 45, axis=2)
    counts = np.sum(dark, axis=0)
    columns = np.flatnonzero(counts >= int(0.72 * (y1 - y0)))
    groups: list[list[int]] = []
    for column in columns:
        value = int(column)
        if not groups or value > groups[-1][-1] + 1:
            groups.append([value])
        else:
            groups[-1].append(value)
    centers = [int(round(sum(group) / len(group))) for group in groups]
    left = [value for value in centers if value < 0.20 * width]
    right = [value for value in centers if 0.75 * width < value < 0.96 * width]
    if not left or not right:
        raise RuntimeError(f"cannot calibrate plot geometry: {path}")
    x0, x1 = max(left), min(right)
    if x1 <= x0:
        raise RuntimeError(f"invalid plot geometry: {path}")
    cwt_y0, cwt_y1 = int(0.34 * height), int(0.92 * height)
    cwt = rgb[cwt_y0:cwt_y1, x0 : x1 + 1]
    non_background = np.sum(np.min(cwt, axis=2) < 245, axis=0)
    data_columns = np.flatnonzero(non_background > int(0.35 * (cwt_y1 - cwt_y0)))
    data_groups: list[list[int]] = []
    for column in data_columns:
        value = int(column)
        if not data_groups or value > data_groups[-1][-1] + 1:
            data_groups.append([value])
        else:
            data_groups[-1].append(value)
    if not data_groups:
        raise RuntimeError(f"cannot locate colored CWT extent: {path}")
    data_group = max(data_groups, key=len)
    data_x0 = x0 + data_group[0]
    data_x1 = x0 + data_group[-1]
    if data_x1 <= data_x0:
        raise RuntimeError(f"invalid colored CWT extent: {path}")
    return {
        "image_width_px": width,
        "image_height_px": height,
        "plot_x0_px": x0,
        "plot_x1_px": x1,
        "data_x0_px": data_x0,
        "data_x1_px": data_x1,
    }


def load_geometry(
    image_dir: Path,
    selection: list[dict[str, str]],
    images: dict[str, Path],
) -> dict[str, dict[str, int]]:
    cache_path = image_dir / "review_geometry.json"
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        cached = payload.get("cases", {})
        if payload.get("version") == 3 and set(cached) == set(images):
            return {str(key): value for key, value in cached.items()}
    geometry: dict[str, dict[str, int]] = {}
    for index, row in enumerate(selection, 1):
        key = row["raw_key"]
        geometry[key] = detect_plot_geometry(images[key])
        if index == 1 or index % 100 == 0 or index == len(selection):
            print(f"[review] calibrating image axes={index}/{len(selection)}", flush=True)
    cache_path.write_text(
        json.dumps({"version": 3, "cases": geometry}, separators=(",", ":")),
        encoding="utf-8",
    )
    return geometry


def load_dataset(
    selection_path: Path,
    labels_path: Path,
    image_dir: Path,
    metadata_path: Path,
) -> tuple[list[dict[str, object]], list[str], list[dict[str, str]], dict[str, Path]]:
    _selection_fields, selection = read_csv(selection_path)
    label_fields, label_rows = read_csv(labels_path)
    metadata = {
        str(row["raw_key"]): row
        for row in json.loads(metadata_path.read_text(encoding="utf-8"))
    }
    if len(selection) != EXPECTED_CASES or len(label_rows) != EXPECTED_CASES:
        raise RuntimeError("review requires exactly 1,993 selection and label rows")
    selection_keys = [row["raw_key"] for row in selection]
    label_keys = [row["raw_key"] for row in label_rows]
    if len(set(selection_keys)) != EXPECTED_CASES or set(selection_keys) != set(label_keys):
        raise RuntimeError("selection and labels must contain identical unique raw_key values")
    schema_changed = False
    for field in (*EDITABLE_FIELDS, GEOMETRY_VERSION_FIELD):
        if field not in label_fields:
            label_fields.append(field)
            schema_changed = True
    images: dict[str, Path] = {}
    for row in selection:
        key = row["raw_key"]
        path = image_dir / f"{int(row['review_rank']):04d}_{key}.png"
        if not path.is_file():
            raise RuntimeError(f"missing review image: {path}")
        images[key] = path
    geometry = load_geometry(image_dir, selection, images)

    selection_by_key = {row["raw_key"]: row for row in selection}
    migrated = 0
    for label in label_rows:
        key = label["raw_key"]
        source_geometry_version = str(label.get(GEOMETRY_VERSION_FIELD, "")).strip()
        if source_geometry_version == GEOMETRY_VERSION:
            continue
        selection_row = selection_by_key[key]
        meta = metadata[key]
        image_geometry = geometry[key]
        requested_start = int(selection_row["extract_t0_rec"])
        requested_stop = int(selection_row["extract_t1_rec"])
        actual_start = int(meta["extract_t0_rec"])
        actual_stop = int(meta["extract_t1_rec"])
        requested_span = max(requested_stop - requested_start, 1)
        actual_span = max(actual_stop - actual_start, 1)
        axis_span = image_geometry["plot_x1_px"] - image_geometry["plot_x0_px"]
        data_span = image_geometry["data_x1_px"] - image_geometry["data_x0_px"]
        row_label = str(label.get("label", "")).strip().lower()
        row_confidence = str(label.get("label_confidence", "")).strip().lower()
        old_intervals: list[dict[str, object]] = []
        truth_start = str(label.get("truth_t0_rec", "")).strip()
        truth_stop = str(label.get("truth_t1_rec", "")).strip()
        if str(label.get("intervals", "")).strip():
            for interval in json.loads(label["intervals"]):
                old_intervals.append(
                    {
                        "t0": int(interval["t0"]),
                        "t1": int(interval["t1"]),
                        "lc": int(interval.get("lc", 0)),
                        "rc": int(interval.get("rc", 0)),
                        "label": str(interval.get("label", row_label)).strip().lower(),
                        "conf": str(interval.get("conf", row_confidence)).strip().lower(),
                    }
                )
        elif truth_start and truth_stop:
            old_intervals.append(
                {
                    "t0": int(float(truth_start)),
                    "t1": int(float(truth_stop)),
                    "lc": int(str(label.get("truth_left_censored", "0")).strip() == "1"),
                    "rc": int(str(label.get("truth_right_censored", "0")).strip() == "1"),
                    "label": row_label,
                    "conf": row_confidence,
                }
            )
        elif str(label.get("fp_intervals", "")).strip():
            for start, stop in json.loads(label["fp_intervals"]):
                old_intervals.append(
                    {
                        "t0": int(start),
                        "t1": int(stop),
                        "lc": 0,
                        "rc": 0,
                        "label": row_label,
                        "conf": row_confidence,
                    }
                )
        elif row_label:
            old_intervals.append(
                {
                    "t0": int(selection_row["t0_rec"]),
                    "t1": int(selection_row["t1_rec"]),
                    "lc": 0,
                    "rc": 0,
                    "label": row_label,
                    "conf": row_confidence,
                }
            )
        converted_intervals: list[dict[str, object]] = []
        for interval in old_intervals:
            converted: dict[str, object] = {
                "lc": interval["lc"],
                "rc": interval["rc"],
                "label": interval["label"],
                "conf": interval["conf"],
            }
            for field in ("t0", "t1"):
                if source_geometry_version in {"2", "3"} or not (truth_start and truth_stop):
                    converted[field] = int(np.clip(int(interval[field]), actual_start, actual_stop))
                else:
                    old_fraction = (int(interval[field]) - requested_start) / requested_span
                    old_pixel = image_geometry["plot_x0_px"] + old_fraction * axis_span
                    actual_fraction = (old_pixel - image_geometry["data_x0_px"]) / data_span
                    coordinate = actual_start + np.clip(actual_fraction, 0.0, 1.0) * actual_span
                    converted[field] = int(round(coordinate))
            if int(converted["t1"]) > int(converted["t0"]):
                converted_intervals.append(converted)
        label["intervals"] = json.dumps(converted_intervals, separators=(",", ":"))
        label[GEOMETRY_VERSION_FIELD] = GEOMETRY_VERSION
        migrated += int(bool(converted_intervals))
        schema_changed = True
    if any(field in label_fields for field in LEGACY_INTERVAL_FIELDS):
        label_fields = [field for field in label_fields if field not in LEGACY_INTERVAL_FIELDS]
        for label in label_rows:
            for field in LEGACY_INTERVAL_FIELDS:
                label.pop(field, None)
        schema_changed = True
    if any(field in label_fields for field in LEGACY_CLASS_FIELDS):
        label_fields = [field for field in label_fields if field not in LEGACY_CLASS_FIELDS]
        for label in label_rows:
            for field in LEGACY_CLASS_FIELDS:
                label.pop(field, None)
        schema_changed = True
    if schema_changed:
        write_labels(labels_path, label_fields, label_rows)
        print(f"[review] upgraded image/data geometry version={GEOMETRY_VERSION} labelled={migrated}")

    labels_by_key = {row["raw_key"]: row for row in label_rows}
    cases: list[dict[str, object]] = []
    for row in selection:
        label = labels_by_key[row["raw_key"]]
        cases.append(
            {
                "review_rank": int(row["review_rank"]),
                "raw_key": row["raw_key"],
                "run_id": row["run_id"],
                "channel": int(row["channel"]),
                "freq_mhz": float(row["freq_mhz"]),
                "candidate_t0_rec": int(row["t0_rec"]),
                "candidate_t1_rec": int(row["t1_rec"]),
                "record_min": int(metadata[row["raw_key"]]["extract_t0_rec"]),
                "record_max": int(metadata[row["raw_key"]]["extract_t1_rec"]),
                **geometry[row["raw_key"]],
                **{field: label.get(field, "") for field in EDITABLE_FIELDS},
            }
        )
    return cases, label_fields, label_rows, images


def parse_intervals(value: object) -> list[dict[str, object]]:
    if value in (None, ""):
        return []
    payload = json.loads(value) if isinstance(value, str) else value
    if not isinstance(payload, list):
        raise ValueError("标注边界必须是区间列表")
    intervals: list[dict[str, object]] = []
    def parse_flag(flag: object) -> int:
        return int(str(flag).strip().lower() in {"1", "true"})

    for index, interval in enumerate(payload, 1):
        if isinstance(interval, dict):
            start = int(interval.get("t0", -1))
            stop = int(interval.get("t1", -1))
            left_censored = parse_flag(interval.get("lc", False))
            right_censored = parse_flag(interval.get("rc", False))
            interval_label = str(interval.get("label", "")).strip().lower()
            confidence = str(interval.get("conf", "")).strip().lower()
        elif isinstance(interval, list) and len(interval) in {2, 4}:
            start = int(interval[0])
            stop = int(interval[1])
            left_censored = parse_flag(interval[2]) if len(interval) == 4 else 0
            right_censored = parse_flag(interval[3]) if len(interval) == 4 else 0
            interval_label = ""
            confidence = ""
        else:
            raise ValueError(f"标注框 {index} 必须包含左右两个边界")
        if start < 0 or stop <= start:
            raise ValueError(f"标注框 {index} 的右边界必须大于左边界")
        if interval_label not in VALID_INTERVAL_LABELS:
            raise ValueError(f"标注框 {index} 的分类无效")
        if confidence not in VALID_CONFIDENCE:
            raise ValueError(f"标注框 {index} 的置信度无效")
        intervals.append(
            {
                "t0": start,
                "t1": stop,
                "lc": left_censored,
                "rc": right_censored,
                "label": interval_label,
                "conf": confidence,
            }
        )
    intervals.sort(key=lambda interval: (int(interval["t0"]), int(interval["t1"])))
    if any(
        int(current["t0"]) < int(previous["t1"])
        for previous, current in zip(intervals, intervals[1:])
    ):
        raise ValueError("多个标注框不能相互重叠")
    return intervals


def validate_update(payload: dict[str, object]) -> dict[str, str]:
    intervals = parse_intervals(payload.get("intervals"))
    if not intervals:
        raise ValueError("样本至少需要一个标注框")
    has_unconfirmed_keep = any(
        interval["label"] == "keep" for interval in intervals
    ) and payload.get("boundaries_confirmed") is not True
    if has_unconfirmed_keep:
        raise ValueError("可靠样本框保存前必须确认其左右边界")
    return {
        "notes": str(payload.get("notes", "")).strip(),
        "intervals": json.dumps(intervals, separators=(",", ":")),
        GEOMETRY_VERSION_FIELD: GEOMETRY_VERSION,
    }


def write_labels(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CPRO 1993 人工审核</title>
  <style>
    :root { --ink:#172121; --paper:#f2eee3; --panel:#fffdf6; --line:#b9b19e; --keep:#147d64; --fp:#c24e36; --uncertain:#b17a12; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:linear-gradient(135deg,#e8e0ce,#f8f5ec 45%,#dfe8e3); font-family:"Avenir Next","Gill Sans",sans-serif; }
    header { display:flex; align-items:center; gap:16px; padding:12px 18px; border-bottom:1px solid var(--line); background:rgba(255,253,246,.94); position:sticky; top:0; z-index:2; }
    h1 { margin:0; font-family:"Iowan Old Style",Georgia,serif; font-size:23px; }
    .counter { font-variant-numeric:tabular-nums; }
    .spacer { flex:1; }
    .shortcut-info { position:relative; display:inline-grid; place-items:center; width:25px; height:25px; border:1px solid #7b7568; border-radius:50%; color:#625d52; font:700 15px/1 Georgia,serif; cursor:help; }
    .shortcut-tip { visibility:hidden; opacity:0; position:absolute; right:0; top:34px; width:245px; padding:10px 12px; border:1px solid var(--line); border-radius:6px; background:#fffdf6; color:var(--ink); box-shadow:0 10px 24px rgba(39,35,29,.2); font:600 12px/1.7 "Avenir Next","PingFang SC",sans-serif; white-space:nowrap; pointer-events:none; transition:opacity .12s ease; z-index:9; }
    .shortcut-info:hover .shortcut-tip { visibility:visible; opacity:1; }
    .filter-wrap { position:relative; }
    .filter-panel { display:none; position:absolute; right:0; top:44px; width:330px; padding:13px; border:1px solid var(--line); border-radius:7px; background:#fffdf6; box-shadow:0 12px 28px rgba(39,35,29,.2); z-index:8; }
    .filter-panel.open { display:block; }
    .filter-title { margin:0 0 8px; font:700 14px/1.2 "Iowan Old Style",Georgia,serif; }
    .filter-group { margin-top:9px; padding-top:8px; border-top:1px solid #d6cfbf; }
    .filter-group strong { display:block; margin-bottom:6px; font-size:12px; }
    .filter-checks { display:flex; flex-wrap:wrap; gap:6px 12px; }
    .filter-check { display:flex; align-items:center; gap:5px; margin:0; font-size:12px; font-weight:600; letter-spacing:0; text-transform:none; }
    .filter-check input { width:auto; margin:0; }
    .filter-actions { display:grid; grid-template-columns:1fr 1fr; gap:7px; margin-top:12px; }
    .filter-result { min-height:18px; margin-top:8px; color:#625d52; font-size:12px; }
    button, input, select, textarea { font:inherit; }
    button { border:1px solid #7b7568; background:#fffdf6; border-radius:5px; padding:7px 12px; cursor:pointer; }
    button:hover { background:#e9e2d4; }
    main { display:grid; grid-template-columns:minmax(0,1fr) 310px; gap:14px; padding:14px; }
    .figure { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:8px; min-height:70vh; display:flex; align-items:center; justify-content:center; overflow:auto; }
    .image-stage { position:relative; display:inline-block; width:fit-content; max-width:100%; line-height:0; user-select:none; }
    .figure img { display:block; max-width:100%; max-height:calc(100vh - 110px); pointer-events:none; }
    .no-data { position:absolute; top:0; bottom:0; z-index:1; display:none; align-items:center; justify-content:center; color:#554f45; background:repeating-linear-gradient(135deg,rgba(90,84,73,.08) 0 5px,rgba(90,84,73,.22) 5px 7px); font:700 11px/1 "Avenir Next","PingFang SC",sans-serif; letter-spacing:.08em; }
    .interval-fill { position:absolute; top:0; bottom:0; z-index:2; cursor:pointer; }
    .interval-fill.active { box-shadow:inset 0 2px var(--span-color),inset 0 -2px var(--span-color); }
    .interval-badge { position:absolute; left:50%; transform:translateX(-50%); padding:5px 8px; border-radius:4px; color:white; font:800 12px/1 "Avenir Next","PingFang SC",sans-serif; white-space:nowrap; box-shadow:0 2px 6px rgba(0,0,0,.25); }
    .boundary { position:absolute; top:0; bottom:0; width:1px; transform:translateX(-.5px); cursor:ew-resize; z-index:3; touch-action:none; outline:none; }
    .boundary::before { content:""; position:absolute; top:0; bottom:0; }
    .boundary.left::before { left:-10px; right:0; }
    .boundary.right::before { left:0; right:-10px; }
    .boundary.left { background:#0077b6; }
    .boundary.right { background:#d1495b; }
    .boundary.active .handle { outline:2px solid #fff; box-shadow:0 0 0 3px currentColor,0 2px 7px rgba(0,0,0,.28); }
    .boundary.left.active { color:#0077b6; } .boundary.right.active { color:#d1495b; }
    .handle { position:absolute; left:50%; transform:translateX(-50%); padding:6px 8px; border-radius:5px; color:white; font:700 12px/1.15 "Avenir Next","PingFang SC",sans-serif; white-space:nowrap; box-shadow:0 2px 7px rgba(0,0,0,.28); }
    .left .handle { top:66px; background:#0077b6; } .right .handle { top:104px; background:#d1495b; }
    aside { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; align-self:start; position:sticky; top:72px; }
    .meta { font-size:13px; line-height:1.5; padding-bottom:10px; border-bottom:1px solid var(--line); overflow-wrap:anywhere; }
    label { display:block; margin-top:11px; font-size:12px; font-weight:700; letter-spacing:.04em; text-transform:uppercase; }
    input, select, textarea { width:100%; margin-top:4px; border:1px solid #918a7b; border-radius:4px; padding:7px; background:white; }
    textarea { min-height:72px; resize:vertical; }
    .bounds { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:11px; }
    .bound-readout { border:1px solid #918a7b; border-radius:4px; padding:8px; background:#f8f5ec; font-variant-numeric:tabular-nums; }
    .bound-readout strong { display:block; margin-bottom:3px; font-size:12px; }
    .censor-options { margin-top:9px; display:grid; gap:5px; }
    .censor-option { display:flex; align-items:center; gap:7px; margin:0; font-size:12px; font-weight:600; letter-spacing:0; text-transform:none; }
    .censor-option input { width:auto; margin:0; }
    .span-controls { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-top:8px; }
    .span-index { grid-column:1 / -1; text-align:center; font-size:12px; color:#625d52; font-variant-numeric:tabular-nums; }
    .status { min-height:22px; margin-top:9px; font-size:13px; }
    .error { color:#a32020; font-weight:700; }
    .keys { margin-top:10px; color:#625d52; font-size:12px; line-height:1.5; }
    @media (max-width:900px) { main { grid-template-columns:1fr; } aside { position:static; } .figure img { max-height:none; } }
  </style>
</head>
<body>
<header>
  <h1>CPRO 1993 人工审核</h1>
  <button id="prev">上一个</button><button id="next">下一个</button><button id="next-unlabelled">下一个未标记</button>
  <span class="counter" id="counter"></span><span class="spacer"></span>
  <span class="shortcut-info" aria-label="快捷键提示">ⓘ<span class="shortcut-tip">N / M：新增 / 删除标注框<br>K：确认当前框现有边界<br>A / D：左 / 右删失<br>Q / E：真实 / 假阳性<br>F：假阳性并进入下一条<br>1 / 2 / 3：低 / 中 / 高置信度<br>空格：保存并进入下一条<br>← / →：微调当前边界</span></span>
  <div class="filter-wrap"><button id="filter-toggle">复核筛选</button><div class="filter-panel" id="filter-panel">
    <div class="filter-title">已标注区间复核</div>
    <div class="filter-group"><strong>区间分类</strong><div class="filter-checks">
      <label class="filter-check"><input class="filter-label" type="checkbox" value="keep">真实</label>
      <label class="filter-check"><input class="filter-label" type="checkbox" value="fp">假阳性</label>
      <label class="filter-check"><input class="filter-label" type="checkbox" value="uncertain">不确定</label>
    </div></div>
    <div class="filter-group"><strong>区间置信度</strong><div class="filter-checks">
      <label class="filter-check"><input class="filter-conf" type="checkbox" value="low">低</label>
      <label class="filter-check"><input class="filter-conf" type="checkbox" value="medium">中</label>
      <label class="filter-check"><input class="filter-conf" type="checkbox" value="high">高</label>
      <label class="filter-check"><input class="filter-conf" type="checkbox" value="">未设</label>
    </div></div>
    <div class="filter-actions"><button id="apply-filter">应用筛选</button><button id="clear-filter">清除筛选</button></div>
    <div class="filter-result" id="filter-result"></div>
  </div></div>
  <label style="margin:0">序号 <input id="rank" type="number" min="1" max="1993" style="width:88px;margin-left:5px"></label>
</header>
<main>
  <section class="figure"><div class="image-stage" id="stage"><img id="image" alt="原始通道与 CWT 审核图"><div class="no-data" id="left-no-data">无数据</div><div class="no-data" id="right-no-data">无数据</div><div id="boundaries"></div></div></section>
  <aside>
    <div class="meta" id="meta"></div>
    <label>当前框分类<select id="label"><option value=""></option><option value="keep">真实：有效持续结构</option><option value="fp">假阳性</option><option value="uncertain">不确定</option></select></label>
    <div class="bounds">
      <div class="bound-readout"><strong>标注左边界</strong><span id="t0"></span></div>
      <div class="bound-readout"><strong>标注右边界</strong><span id="t1"></span></div>
    </div>
    <div class="span-controls" id="span-controls">
      <div class="span-index" id="span-index"></div>
      <button id="add-span">新增标注框</button><button id="delete-span">删除当前框</button>
    </div>
    <div class="censor-options">
      <label class="censor-option"><input id="left-censored" type="checkbox">当前结构在观测开始前已存在（左删失）</label>
      <label class="censor-option"><input id="right-censored" type="checkbox">当前结构到观测结束仍存在（右删失）</label>
    </div>
    <label>当前框置信度<select id="confidence"><option value=""></option><option value="high">高</option><option value="medium">中</option><option value="low">低</option></select></label>
    <label>备注<textarea id="notes"></textarea></label>
    <div class="status" id="status"></div>
    <div class="keys">点击色带或细线选择当前框，再用下拉框设置分类与置信度。←/→ 每次微调 1 record，按住 Shift 时每次移动 8 records。点击“下一个”自动保存。</div>
  </aside>
</main>
<script>
let cases=[], index=0, dragging=null, active={span:0,side:'left'}, spans=[], touched=[];
let filterActive=false, filteredIndices=[], filterLabels=new Set(), filterConfs=new Set();
const $=id=>document.getElementById(id);
const spanColors=['#0077b6','#d1495b','#6a994e','#9b5de5','#bc6c25','#008c95','#c1121f','#577590'];
const labelColors={keep:'#147d64',fp:'#c24e36',uncertain:'#b17a12','':'#0077b6'};
const labelNames={keep:'真实',fp:'假阳性',uncertain:'不确定','':'未分类'};
function current(){ return cases[index]; }
function clamp(v,lo,hi){ return Math.max(lo,Math.min(hi,v)); }
function currentSpan(){ return spans[active.span]; }
function boundaryElement(span,side){ return document.querySelector(`[data-span="${span}"][data-side="${side}"]`); }
function setActive(span,side){
  active={span,side};
  document.querySelectorAll('.boundary').forEach(element=>element.classList.toggle('active',Number(element.dataset.span)===span&&element.dataset.side===side));
  document.querySelectorAll('.interval-fill').forEach(element=>element.classList.toggle('active',Number(element.dataset.span)===span));
  renderReadout();
}
function recordToPercent(record){ const c=current(), pixel=c.data_x0_px+(record-c.record_min)/(c.record_max-c.record_min)*(c.data_x1_px-c.data_x0_px); return 100*pixel/c.image_width_px; }
function clientToRecord(clientX){ const c=current(), rect=$('image').getBoundingClientRect(), imagePixel=(clientX-rect.left)/rect.width*c.image_width_px, fraction=(imagePixel-c.data_x0_px)/(c.data_x1_px-c.data_x0_px); return Math.round(c.record_min+clamp(fraction,0,1)*(c.record_max-c.record_min)); }
function renderNoData(){
  const c=current(), left=$('left-no-data'), right=$('right-no-data'), leftGap=c.data_x0_px-c.plot_x0_px, rightGap=c.plot_x1_px-c.data_x1_px;
  left.style.left=100*c.plot_x0_px/c.image_width_px+'%'; left.style.width=100*leftGap/c.image_width_px+'%'; left.style.display=leftGap>=3?'flex':'none';
  right.style.left=100*c.data_x1_px/c.image_width_px+'%'; right.style.width=100*rightGap/c.image_width_px+'%'; right.style.display=rightGap>=3?'flex':'none';
}
function renderReadout(){
  const span=currentSpan();
  if(!span)return;
  $('t0').textContent=span.left; $('t1').textContent=span.right;
  $('left-censored').checked=Boolean(span.lc); $('right-censored').checked=Boolean(span.rc);
  $('label').value=span.label||''; $('confidence').value=span.conf||'';
  $('span-index').textContent=`当前框 ${active.span+1} / ${spans.length} · ${labelNames[span.label||'']}`;
  $('delete-span').disabled=spans.length<=1;
}
function renderBounds(){
  for(let spanIndex=0;spanIndex<spans.length;spanIndex++){
    const span=spans[spanIndex], color=labelColors[span.label||''], fill=document.querySelector(`.interval-fill[data-span="${spanIndex}"]`);
    if(fill){
      fill.style.left=recordToPercent(span.left)+'%'; fill.style.width=(recordToPercent(span.right)-recordToPercent(span.left))+'%';
      fill.style.background=color+'1f'; fill.style.setProperty('--span-color',color);
      const badge=fill.querySelector('.interval-badge'); badge.style.background=color; badge.style.top=(28+spanIndex%4*28)+'px'; badge.textContent=`框 ${spanIndex+1} · ${labelNames[span.label||'']}`;
    }
    for(const side of ['left','right']){
      const element=boundaryElement(spanIndex,side), value=spans[spanIndex][side];
      if(!element)continue;
      element.style.background=color; element.style.color=color; element.querySelector('.handle').style.background=color;
      element.style.left=recordToPercent(value)+'%';
      const kind={keep:'真',fp:'假',uncertain:'?'}[spans[spanIndex].label]||'未分';
      element.querySelector('.handle').textContent=`框 ${spanIndex+1} ${kind} ${side==='left'?'左':'右'} ${value}`;
    }
  }
  renderReadout();
}
function rebuildBoundaries(){
  const container=$('boundaries'); container.innerHTML='';
  spans.forEach((span,spanIndex)=>{
    const color=labelColors[span.label||''];
    const fill=document.createElement('div'); fill.className='interval-fill'; fill.dataset.span=spanIndex;
    const badge=document.createElement('span'); badge.className='interval-badge'; fill.appendChild(badge); container.appendChild(fill);
    fill.addEventListener('click',()=>setActive(spanIndex,'left'));
    for(const side of ['left','right']){
      const element=document.createElement('div'); element.className=`boundary ${side}`; element.tabIndex=0;
      element.dataset.span=spanIndex; element.dataset.side=side; element.style.background=color; element.style.color=color;
      const handle=document.createElement('span'); handle.className='handle'; handle.style.background=color; element.appendChild(handle); container.appendChild(element);
      element.addEventListener('pointerdown',event=>{ dragging={span:spanIndex,side}; setActive(spanIndex,side); event.preventDefault(); });
      element.addEventListener('focus',()=>setActive(spanIndex,side));
    }
  });
  setActive(clamp(active.span,0,spans.length-1),active.side); renderBounds();
}
function setBoundary(spanIndex,side,value,confirmed=true){
  const c=current();
  const span=spans[spanIndex];
  if(side==='left'){ span.left=clamp(Math.round(value),c.record_min,span.right-1); if(span.left>c.record_min)span.lc=0; }
  else { span.right=clamp(Math.round(value),span.left+1,c.record_max); if(span.right<c.record_max)span.rc=0; }
  if(confirmed){
    touched[spanIndex][side]=true;
    const allConfirmed=touched.every(item=>item.left&&item.right);
    $('status').textContent=allConfirmed?'全部边界已确认，请选择分类保存':'保留项仍有边界尚未确认'; $('status').className='status';
  }
  renderBounds();
}
function parseIntervals(value){
  if(!value)return [];
  try{return JSON.parse(value).map(interval=>({left:Number(interval.t0),right:Number(interval.t1),lc:Number(interval.lc||0),rc:Number(interval.rc||0),label:interval.label||'',conf:interval.conf||''}));}catch{return [];}
}
function caseComplete(c){ const items=parseIntervals(c.intervals); return items.length>0&&items.every(interval=>interval.label); }
function intervalMatchesFilter(span){
  if(!span.label)return false;
  const labelMatches=!filterLabels.size||filterLabels.has(span.label);
  const confidenceMatches=!filterConfs.size||filterConfs.has(span.conf||'');
  return labelMatches&&confidenceMatches;
}
function caseMatchesFilter(c){ return parseIntervals(c.intervals).some(intervalMatchesFilter); }
function rebuildFilteredIndices(){
  filteredIndices=cases.map((c,i)=>caseMatchesFilter(c)?i:-1).filter(i=>i>=0);
  $('filter-result').textContent=`匹配 ${filteredIndices.length} 条样本`;
}
function updateCounter(){
  if(!filterActive){ $('counter').textContent=`${index+1} / ${cases.length}`; return; }
  const position=filteredIndices.indexOf(index);
  const currentPosition=position>=0?position+1:'-';
  $('counter').textContent=`${currentPosition} / ${filteredIndices.length} 条匹配 · 全局 ${index+1} / ${cases.length}`;
}
function show(){
  const c=current();
  updateCounter(); $('rank').value=c.review_rank;
  $('image').src=`/image/${encodeURIComponent(c.raw_key)}`; $('meta').innerHTML=`<b>${c.raw_key}</b><br>通道 ${c.channel}，${c.freq_mhz.toFixed(6)} MHz<br>旧候选窗：[${c.candidate_t0_rec}, ${c.candidate_t1_rec})<br>实际数据：${c.record_min} .. ${c.record_max}<br>斜纹白区不含数据`;
  $('notes').value=c.notes||'';
  spans=parseIntervals(c.intervals); const intervalsSaved=spans.length>0;
  if(!intervalsSaved)spans=[{left:Number(c.candidate_t0_rec),right:Number(c.candidate_t1_rec),lc:0,rc:0,label:'',conf:''}];
  touched=spans.map(()=>({left:intervalsSaved,right:intervalsSaved}));
  const matchedSpan=filterActive?spans.findIndex(intervalMatchesFilter):-1;
  active={span:matchedSpan>=0?matchedSpan:0,side:'left'}; rebuildBoundaries();
  $('status').textContent=intervalsSaved?'已加载保存的标注边界':'保留项需确认全部边界；其他分类可直接保存或新增框'; $('status').className='status';
  if($('image').complete){ renderNoData(); renderBounds(); } else $('image').onload=()=>{renderNoData();renderBounds();};
}
function go(i){ index=Math.max(0,Math.min(cases.length-1,i)); show(); }
function navigateFiltered(step){
  rebuildFilteredIndices();
  if(!filteredIndices.length){ $('status').textContent='当前筛选没有匹配项'; $('status').className='status'; updateCounter(); return; }
  let position=filteredIndices.indexOf(index);
  if(position<0){
    if(step>0){ position=filteredIndices.findIndex(i=>i>index); if(position<0)position=0; }
    else { position=filteredIndices.findLastIndex(i=>i<index); if(position<0)position=filteredIndices.length-1; }
  } else position=(position+step+filteredIndices.length)%filteredIndices.length;
  go(filteredIndices[position]);
}
async function save(advance=false){
  if(advance&&spans.some(span=>!span.label)){ $('status').textContent='请先为每个标注框选择分类'; $('status').className='status error'; return false; }
  const c=current(), payload={raw_key:c.raw_key,intervals:spans.map(span=>({t0:span.left,t1:span.right,lc:span.lc||0,rc:span.rc||0,label:span.label||'',conf:span.conf||''})),boundaries_confirmed:spans.every((span,i)=>span.label!=='keep'||(touched[i].left&&touched[i].right)),notes:$('notes').value};
  const response=await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const body=await response.json();
  if(!response.ok){ $('status').textContent=body.error; $('status').className='status error'; return false; }
  const savedIndex=index;
  Object.assign(c,body.updated); $('status').textContent='已保存'; $('status').className='status';
  if(advance){
    if(filterActive){
      rebuildFilteredIndices();
      if(!filteredIndices.length){ updateCounter(); $('status').textContent='当前筛选已复核完成'; return true; }
      const nextIndex=filteredIndices.find(i=>i>savedIndex)??filteredIndices[0];
      go(nextIndex);
    } else go(index+1);
  }
  return true;
}
function nextUnlabelled(){ for(let n=1;n<=cases.length;n++){ const i=(index+n)%cases.length; if(!caseComplete(cases[i])){ go(i); return; } } $('status').textContent='1993 条已经全部标记'; }
function selectedValues(selector){ return new Set([...document.querySelectorAll(selector)].filter(input=>input.checked).map(input=>input.value)); }
function clearFilter(closePanel=true){
  filterActive=false; filteredIndices=[]; filterLabels=new Set(); filterConfs=new Set();
  document.querySelectorAll('.filter-label,.filter-conf').forEach(input=>{input.checked=false;});
  $('filter-result').textContent='';
  if(closePanel)$('filter-panel').classList.remove('open');
  updateCounter();
}
function addSpan(){
  const c=current(), occupied=[...spans].sort((a,b)=>a.left-b.left), gaps=[];
  let cursor=Math.max(c.record_min,c.candidate_t0_rec), end=Math.min(c.record_max,c.candidate_t1_rec);
  for(const span of occupied){ if(span.left>cursor)gaps.push([cursor,Math.min(span.left,end)]); cursor=Math.max(cursor,span.right); }
  if(cursor<end)gaps.push([cursor,end]);
  let gap=gaps.filter(item=>item[1]-item[0]>=2).sort((a,b)=>(b[1]-b[0])-(a[1]-a[0]))[0];
  if(!gap)gap=[c.record_min,c.record_max];
  const margin=Math.max(1,Math.floor((gap[1]-gap[0])/4));
  const left=gap[0]+margin, right=Math.max(left+1,gap[1]-margin);
  spans.push({left,right,lc:0,rc:0,label:'',conf:''}); touched.push({left:false,right:false});
  active={span:spans.length-1,side:'left'}; rebuildBoundaries(); $('status').textContent='已新增标注框，请拖动两侧细线';
}
function deleteSpan(){ if(spans.length<=1)return; spans.splice(active.span,1); touched.splice(active.span,1); active={span:Math.min(active.span,spans.length-1),side:'left'}; rebuildBoundaries(); }
function setCurrentLabel(value){ currentSpan().label=value; $('label').value=value; renderBounds(); }
function setCurrentConfidence(value){ currentSpan().conf=value; $('confidence').value=value; }
function rejectAndAdvance(){ setCurrentLabel('fp'); save(true); }
function confirmCurrentBounds(){
  touched[active.span]={left:true,right:true};
  $('status').textContent='当前框现有边界已确认'; $('status').className='status';
}
function toggleCensor(id){ const checkbox=$(id); checkbox.checked=!checkbox.checked; checkbox.dispatchEvent(new Event('change')); }
document.addEventListener('pointermove',event=>{ if(dragging)setBoundary(dragging.span,dragging.side,clientToRecord(event.clientX)); });
document.addEventListener('pointerup',event=>{ if(dragging){setBoundary(dragging.span,dragging.side,clientToRecord(event.clientX));dragging=null;} });
$('prev').onclick=()=>filterActive?navigateFiltered(-1):go(index-1); $('next').onclick=()=>save(true); $('next-unlabelled').onclick=async()=>{if(await save(false)){clearFilter();nextUnlabelled();}};
$('rank').onchange=()=>{clearFilter();go(Number($('rank').value)-1);};
$('filter-toggle').onclick=()=>$('filter-panel').classList.toggle('open');
$('apply-filter').onclick=()=>{
  filterLabels=selectedValues('.filter-label'); filterConfs=selectedValues('.filter-conf'); filterActive=true;
  rebuildFilteredIndices();
  if(!filteredIndices.length){ $('filter-result').textContent='没有匹配项'; updateCounter(); return; }
  $('filter-panel').classList.remove('open'); go(filteredIndices[0]);
};
$('clear-filter').onclick=()=>clearFilter();
$('add-span').onclick=addSpan; $('delete-span').onclick=deleteSpan;
$('label').onchange=()=>setCurrentLabel($('label').value);
$('confidence').onchange=()=>setCurrentConfidence($('confidence').value);
$('left-censored').onchange=()=>{ const span=currentSpan(); span.lc=$('left-censored').checked?1:0; if(span.lc)setBoundary(active.span,'left',current().record_min); };
$('right-censored').onchange=()=>{ const span=currentSpan(); span.rc=$('right-censored').checked?1:0; if(span.rc)setBoundary(active.span,'right',current().record_max); };
document.addEventListener('keydown',e=>{
  if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;
  if(e.key==='ArrowLeft'||e.key==='ArrowRight'){
    e.preventDefault(); const span=currentSpan();
    setBoundary(active.span,active.side,span[active.side]+(e.key==='ArrowLeft'?-1:1)*(e.shiftKey?8:1)); return;
  }
  if(e.repeat)return;
  const key=e.key.toLowerCase();
  const actions={
    n:addSpan,m:deleteSpan,k:confirmCurrentBounds,a:()=>toggleCensor('left-censored'),d:()=>toggleCensor('right-censored'),
    q:()=>setCurrentLabel('keep'),e:()=>setCurrentLabel('fp'),f:rejectAndAdvance,
    '1':()=>setCurrentConfidence('low'),'2':()=>setCurrentConfidence('medium'),'3':()=>setCurrentConfidence('high'),
  };
  if(e.code==='Space'){e.preventDefault();save(true);return;}
  if(actions[key]){e.preventDefault();actions[key]();}
});
window.addEventListener('resize',()=>{if(cases.length){renderNoData();renderBounds();}});
fetch('/api/cases').then(r=>r.json()).then(data=>{cases=data.cases;show();});
</script>
</body></html>
"""


def make_handler(
    cases: list[dict[str, object]],
    label_path: Path,
    label_fields: list[str],
    label_rows: list[dict[str, str]],
    images: dict[str, Path],
) -> type[BaseHTTPRequestHandler]:
    rows_by_key = {row["raw_key"]: row for row in label_rows}
    cases_by_key = {str(row["raw_key"]): row for row in cases}
    write_lock = threading.Lock()

    class ReviewHandler(BaseHTTPRequestHandler):
        def send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json", status)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/cases":
                self.send_json({"cases": cases, "count": len(cases)})
                return
            if path.startswith("/image/"):
                key = unquote(path.removeprefix("/image/"))
                image_path = images.get(key)
                if image_path is None:
                    self.send_json({"error": "unknown raw_key"}, HTTPStatus.NOT_FOUND)
                    return
                self.send_bytes(image_path.read_bytes(), "image/png")
                return
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/label":
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                key = str(payload.get("raw_key", ""))
                if key not in rows_by_key:
                    raise ValueError("unknown raw_key")
                updated = validate_update(payload)
                case = cases_by_key[key]
                intervals = parse_intervals(updated["intervals"])
                record_min = int(case["record_min"])
                record_max = int(case["record_max"])
                for interval in intervals:
                    if interval["lc"]:
                        interval["t0"] = record_min
                    if interval["rc"]:
                        interval["t1"] = record_max
                    if int(interval["t0"]) < record_min or int(interval["t1"]) > record_max:
                        raise ValueError("标注框不能超出实际数据范围")
                    if int(interval["t1"]) <= int(interval["t0"]):
                        raise ValueError("删失边界仍须构成非空时间窗")
                updated["intervals"] = json.dumps(intervals, separators=(",", ":"))
                with write_lock:
                    rows_by_key[key].update(updated)
                    cases_by_key[key].update(updated)
                    write_labels(label_path, label_fields, label_rows)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"updated": updated})

        def log_message(self, format: str, *args: object) -> None:
            return

    return ReviewHandler


def main() -> None:
    args = parse_args()
    cases, fields, rows, images = load_dataset(
        args.selection,
        args.labels,
        args.image_dir,
        args.metadata,
    )
    labelled = sum(
        bool(intervals) and all(interval["label"] for interval in intervals)
        for row in rows
        if (intervals := parse_intervals(row.get("intervals")))
    )
    print(f"[review] validated cases={len(cases)} images={len(images)} labelled={labelled}")
    if args.check:
        return
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(cases, args.labels, fields, rows, images),
    )
    print(f"[review] open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
