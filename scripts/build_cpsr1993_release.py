#!/usr/bin/env python3
"""Build the deterministic CPSR-1993 GitHub Release archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


DATASET_VERSION = "1.0.0"
EXPECTED_CASES = 1993
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests" / "perf" / "cprf_manual_review"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / f"cpsr-1993-v{DATASET_VERSION}.zip",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_basename(value: object) -> str:
    return Path(str(value)).name


def validate_keys(labels: list[dict[str, str]], selection: list[dict[str, str]]) -> set[str]:
    if len(labels) != EXPECTED_CASES or len(selection) != EXPECTED_CASES:
        raise RuntimeError(f"expected {EXPECTED_CASES} label and selection rows")
    label_keys = {row["raw_key"] for row in labels}
    selection_keys = {row["raw_key"] for row in selection}
    if len(label_keys) != EXPECTED_CASES or label_keys != selection_keys:
        raise RuntimeError("labels and selection must contain identical unique raw_key values")
    return label_keys


def validate_npz(path: Path, expected_keys: set[str]) -> None:
    with zipfile.ZipFile(path) as archive:
        keys = {Path(name).stem for name in archive.namelist() if name.endswith(".npy")}
    if keys != expected_keys:
        raise RuntimeError("NPZ keys do not match the reviewed cases")


def write_dataset_readme(path: Path) -> None:
    text = (ROOT / "datasets" / "cpsr_1993" / "README.md").read_text(encoding="utf-8")
    path.write_text(text, encoding="utf-8")


def write_annotation_license(path: Path) -> None:
    path.write_text(
        "CPSR-1993 manual annotations are licensed under the Creative Commons "
        "Attribution 4.0 International license (CC BY 4.0).\n"
        "https://creativecommons.org/licenses/by/4.0/\n\n"
        "CE4 LFRS source-data rights and attribution requirements remain with "
        "the original provider, the China National Space Administration.\n",
        encoding="utf-8",
    )


def write_archive(staging: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for path in sorted(staging.iterdir(), key=lambda item: item.name):
            info = zipfile.ZipInfo(f"cpsr-1993-v{DATASET_VERSION}/{path.name}")
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.external_attr = 0o100644 << 16
            compression = zipfile.ZIP_STORED if path.suffix == ".npz" else zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes(), compress_type=compression)


def main() -> None:
    args = parse_args()
    labels_fields, labels = read_csv(SOURCE / "labels.csv")
    selection_fields, selection = read_csv(SOURCE / "selection.csv")
    keys = validate_keys(labels, selection)

    slices = SOURCE / "artifacts" / "single_channel_slices.npz"
    metadata_source = SOURCE / "artifacts" / "metadata.json"
    validate_npz(slices, keys)
    metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
    if len(metadata) != EXPECTED_CASES or {row["raw_key"] for row in metadata} != keys:
        raise RuntimeError("metadata does not match the reviewed cases")

    for row in selection:
        row["source_file"] = source_basename(row["source_file"])
    for row in metadata:
        row["source_file"] = source_basename(row["source_file"])

    with tempfile.TemporaryDirectory(prefix="cpsr-1993-") as temporary:
        staging = Path(temporary)
        write_csv(staging / "labels.csv", labels_fields, labels)
        write_csv(staging / "selection.csv", selection_fields, selection)
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        shutil.copyfile(slices, staging / slices.name)
        write_dataset_readme(staging / "README.md")
        write_annotation_license(staging / "LICENSE-ANNOTATIONS.txt")

        payloads = sorted(path for path in staging.iterdir() if path.name != "SHA256SUMS")
        checksums = "".join(f"{sha256(path)}  {path.name}\n" for path in payloads)
        (staging / "SHA256SUMS").write_text(checksums, encoding="ascii")
        write_archive(staging, args.output.resolve())

    print(f"archive={args.output.resolve()}")
    print(f"sha256={sha256(args.output.resolve())}")


if __name__ == "__main__":
    main()
