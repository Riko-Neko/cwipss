from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PERF_DIR = PROJECT_DIR / "tests" / "perf"
if str(PERF_DIR) not in sys.path:
    sys.path.insert(0, str(PERF_DIR))

from wavelet_basis_injection import (  # noqa: E402
    WaveletBasisRunConfig,
    all_continuous_wavelets,
    largest_complete_2c,
    run_wavelet_basis_injection,
)


def _path_option(pytestconfig, name: str) -> Path | None:
    value = str(pytestconfig.getoption(name) or "").strip()
    return None if not value else Path(value)


def _optional_int_zero_as_none(value: int | None) -> int | None:
    if value in (None, 0):
        return None
    return int(value)


def _optional_text(value: str | None) -> str | None:
    value = "" if value is None else str(value).strip()
    return None if not value else value


def test_wavelet_basis_injection_outputs_raw_signal_and_cwt_figures(pytestconfig) -> None:
    output_dir = Path(pytestconfig.getoption("--wavelet-basis-output"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    wavelets = tuple(str(pytestconfig.getoption("--wavelet-basis-wavelets")).split())
    config = WaveletBasisRunConfig(
        input_path=_path_option(pytestconfig, "--wavelet-basis-input"),
        input_dir=Path(pytestconfig.getoption("--wavelet-basis-input-dir")),
        injection_config=Path(pytestconfig.getoption("--wavelet-basis-injection-config")),
        cwt_config=Path(pytestconfig.getoption("--wavelet-basis-cwt-config")),
        output_dir=output_dir,
        wavelets=wavelets,
        t_start=pytestconfig.getoption("--wavelet-basis-t-start"),
        t_stop=_optional_int_zero_as_none(pytestconfig.getoption("--wavelet-basis-t-stop")),
        period_min_records=pytestconfig.getoption("--wavelet-basis-period-min"),
        period_max_records=pytestconfig.getoption("--wavelet-basis-period-max"),
        period_count=pytestconfig.getoption("--wavelet-basis-period-count"),
        period_spacing=_optional_text(pytestconfig.getoption("--wavelet-basis-period-spacing")),
        candidate_period_min_records=pytestconfig.getoption("--wavelet-basis-candidate-period-min"),
        candidate_period_max_records=pytestconfig.getoption("--wavelet-basis-candidate-period-max"),
        cwt_method=_optional_text(pytestconfig.getoption("--wavelet-basis-cwt-method")),
        cwt_backend=pytestconfig.getoption("--wavelet-basis-cwt-backend"),
        cuda_device=pytestconfig.getoption("--wavelet-basis-cuda-device"),
        max_injections=pytestconfig.getoption("--wavelet-basis-max-injections"),
        max_wavelets=pytestconfig.getoption("--wavelet-basis-max-wavelets"),
        dpi=pytestconfig.getoption("--wavelet-basis-dpi"),
        progress_every=pytestconfig.getoption("--wavelet-basis-progress-every"),
    )

    result = run_wavelet_basis_injection(config)

    assert result.input_path == (config.input_path or largest_complete_2c(config.input_dir))
    assert result.index_path.exists()
    assert result.summary_path.exists()
    assert result.truth_path.exists()
    assert result.case_count == len(result.wavelets) * result.injection_count
    if "all" in {value.lower() for value in wavelets} and config.max_wavelets == 0:
        assert len(result.wavelets) == len(all_continuous_wavelets())

    with result.summary_path.open(newline="") as fp:
        summary_rows = list(csv.DictReader(fp))
    assert len(summary_rows) == result.case_count
    assert {"wavelet", "injection_id", "profile_peak_period_records", "activity_raw_max"}.issubset(summary_rows[0])
    assert all(
        int(row["record_stop"]) - int(row["record_start"])
        <= int(row["local_records"])
        <= 2 * (int(row["record_stop"]) - int(row["record_start"]))
        for row in summary_rows
    )

    resolved = json.loads((result.output_dir / "config.resolved.json").read_text())
    assert float(resolved["config"]["detection"]["candidate_period_max_records"]) == 1000.0

    panel_images = sorted(result.output_dir.glob("*__local_cwt_diagnostic_panel.png"))
    assert len(panel_images) == result.case_count
    assert not list(result.output_dir.glob("*/local_cwt_diagnostic_panel.png"))
