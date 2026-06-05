#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cwipss.cwt import cwt_power_cube, period_grid_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small CUDA CWT smoke test and optional CPU comparison.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--records", type=int, default=1024, help="Synthetic record count.")
    parser.add_argument("--channels", type=int, default=16, help="Synthetic channel count.")
    parser.add_argument("--period-min-records", type=float, default=2.0, help="Minimum CWT period.")
    parser.add_argument("--period-max-records", type=float, default=128.0, help="Maximum CWT period.")
    parser.add_argument("--period-count", type=int, default=32, help="Number of CWT periods.")
    parser.add_argument("--wavelet", type=str, default="cmor1.5-1.0", help="CWT wavelet.")
    parser.add_argument("--seed", type=int, default=12345, help="Synthetic data RNG seed.")
    parser.add_argument("--repeat", type=int, default=3, help="CUDA timing repeats after one warmup.")
    parser.add_argument("--rtol", type=float, default=5e-3, help="CPU/CUDA relative tolerance.")
    parser.add_argument("--atol", type=float, default=5e-4, help="CPU/CUDA absolute tolerance.")
    parser.add_argument("--skip-cpu-compare", action="store_true", help="Skip CPU reference and only time CUDA.")
    return parser.parse_args()


def _cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise SystemExit("CuPy is not installed. Install cupy-cuda12x or the CuPy build matching the server CUDA runtime.") from exc
    return cp


def _device_report(cp, device: int) -> None:
    with cp.cuda.Device(device):
        props = cp.cuda.runtime.getDeviceProperties(device)
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    print(f"CUDA device {device}: {name}")
    print(f"CUDA memory: free={free_bytes / 1024**3:.2f} GiB total={total_bytes / 1024**3:.2f} GiB")


def _time_call(label: str, repeat: int, fn):
    times: list[float] = []
    result = None
    for _idx in range(max(1, int(repeat))):
        start = perf_counter()
        result = fn()
        times.append(perf_counter() - start)
    best = min(times)
    mean = sum(times) / len(times)
    print(f"{label}: best={best:.4f}s mean={mean:.4f}s repeats={len(times)}")
    return result, times


def main() -> None:
    args = parse_args()
    cp = _cupy()
    with cp.cuda.Device(args.device):
        _device_report(cp, args.device)
        rng = np.random.default_rng(args.seed)
        data = rng.normal(size=(args.records, args.channels)).astype(np.float32)
        periods = period_grid_records(
            args.period_min_records,
            args.period_max_records,
            args.period_count,
            spacing="log",
        )
        output_gib = periods.size * args.records * args.channels * np.dtype(np.float32).itemsize / 1024**3
        print(
            "case: "
            f"records={args.records} channels={args.channels} periods={periods.size} "
            f"wavelet={args.wavelet} output={output_gib:.3f} GiB"
        )

        print("warmup: cuda")
        _ = cwt_power_cube(
            data,
            periods,
            wavelet=args.wavelet,
            method="fft",
            backend="cuda",
            cuda_device=args.device,
        )
        cp.cuda.Stream.null.synchronize()

        cuda_power, cuda_times = _time_call(
            "cuda",
            args.repeat,
            lambda: cwt_power_cube(
                data,
                periods,
                wavelet=args.wavelet,
                method="fft",
                backend="cuda",
                cuda_device=args.device,
            ),
        )
        cp.cuda.Stream.null.synchronize()

        if not args.skip_cpu_compare:
            cpu_power, _cpu_times = _time_call(
                "cpu",
                1,
                lambda: cwt_power_cube(
                    data,
                    periods,
                    wavelet=args.wavelet,
                    method="fft",
                    backend="cpu",
                ),
            )
            diff = np.abs(cuda_power - cpu_power)
            denom = np.maximum(np.abs(cpu_power), args.atol)
            max_abs = float(np.nanmax(diff))
            max_rel = float(np.nanmax(diff / denom))
            ok = bool(np.allclose(cuda_power, cpu_power, rtol=args.rtol, atol=args.atol))
            print(f"compare: allclose={ok} max_abs={max_abs:.6g} max_rel={max_rel:.6g}")
            if not ok:
                raise SystemExit(1)

        elems = periods.size * args.records * args.channels
        best = min(cuda_times)
        print(f"throughput: {elems / best / 1e6:.2f} million output elems/s")
        pool = cp.get_default_memory_pool()
        pinned_pool = cp.get_default_pinned_memory_pool()
        print(
            "cupy pools: "
            f"device_used={pool.used_bytes() / 1024**3:.3f} GiB "
            f"device_total={pool.total_bytes() / 1024**3:.3f} GiB "
            f"pinned_free_blocks={pinned_pool.n_free_blocks()}"
        )


if __name__ == "__main__":
    main()
