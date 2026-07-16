from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def pytest_addoption(parser) -> None:
    group = parser.getgroup("cwipss wavelet-basis diagnostics")
    group.addoption(
        "--wavelet-basis-output",
        action="store",
        default="runs/pytest_wavelet_basis_smoke",
        help="Output directory for wavelet-basis diagnostic artifacts.",
    )
    group.addoption(
        "--wavelet-basis-input",
        action="store",
        default="",
        help="Specific CE4 .2C file to use. Empty value selects the largest complete .2C file under input-dir.",
    )
    group.addoption(
        "--wavelet-basis-input-dir",
        action="store",
        default=str(PROJECT_DIR / "data" / "CE4"),
        help="Directory searched for the largest complete CE4 .2C file when input is empty.",
    )
    group.addoption(
        "--wavelet-basis-injection-config",
        action="store",
        default="configs/injection_lowfreq_random_100.json",
        help="Injection configuration used by the wavelet-basis diagnostic pytest.",
    )
    group.addoption(
        "--wavelet-basis-cwt-config",
        action="store",
        default="configs/cwt_default.json",
        help="CWT pipeline configuration reused by the diagnostic pytest.",
    )
    group.addoption(
        "--wavelet-basis-wavelets",
        action="store",
        default="cmor1.5-1.0",
        help='Space-separated wavelet list, or "all" for all PyWavelets continuous wavelets.',
    )
    group.addoption("--wavelet-basis-t-start", action="store", type=int, default=None)
    group.addoption("--wavelet-basis-t-stop", action="store", type=int, default=0)
    group.addoption("--wavelet-basis-period-min", action="store", type=float, default=None)
    group.addoption("--wavelet-basis-period-max", action="store", type=float, default=2048.0)
    group.addoption("--wavelet-basis-period-count", action="store", type=int, default=128)
    group.addoption("--wavelet-basis-period-spacing", action="store", default="")
    group.addoption("--wavelet-basis-candidate-period-min", action="store", type=float, default=10.0)
    group.addoption("--wavelet-basis-candidate-period-max", action="store", type=float, default=1000.0)
    group.addoption("--wavelet-basis-cwt-method", action="store", default="")
    group.addoption("--wavelet-basis-cwt-backend", action="store", default="cpu")
    group.addoption("--wavelet-basis-cuda-device", action="store", type=int, default=None)
    group.addoption(
        "--wavelet-basis-max-injections",
        action="store",
        type=int,
        default=1,
        help="Maximum injection specs to render. Use 0 for all specs in the injection config.",
    )
    group.addoption(
        "--wavelet-basis-max-wavelets",
        action="store",
        type=int,
        default=1,
        help="Maximum wavelets to render after resolving the wavelet list. Use 0 for all requested wavelets.",
    )
    group.addoption("--wavelet-basis-dpi", action="store", type=int, default=140)
    group.addoption("--wavelet-basis-progress-every", action="store", type=int, default=10)
