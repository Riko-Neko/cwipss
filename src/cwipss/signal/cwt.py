"""Continuous wavelet transforms and period-grid utilities."""

from __future__ import annotations

import numpy as np
import pywt


def robust_zscore(values: np.ndarray, clip: float = 8.0) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    median = np.nanmedian(x)
    centered = x - median
    mad = np.nanmedian(np.abs(centered))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(centered)
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip(centered / scale, -clip, clip).astype(np.float32)


def robust_zscore_channels(values: np.ndarray, clip: float = 8.0) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("values must have shape (records, channels)")
    median = np.nanmedian(matrix, axis=0, keepdims=True)
    centered = matrix - median
    mad = np.nanmedian(np.abs(centered), axis=0, keepdims=True)
    scale = 1.4826 * mad
    fallback = np.nanstd(centered, axis=0, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    return np.clip(centered / scale, -clip, clip).astype(np.float32)


def period_grid_records(
    min_period: float,
    max_period: float,
    count: int,
    spacing: str = "log",
) -> np.ndarray:
    lo = max(float(min_period), 1.0)
    hi = max(lo, float(max_period))
    n = max(1, int(count))
    if n == 1:
        return np.array([lo], dtype=np.float64)
    if spacing == "linear":
        periods = np.linspace(lo, hi, n, dtype=np.float64)
    elif spacing == "log":
        periods = np.geomspace(lo, hi, n, dtype=np.float64)
    else:
        raise ValueError(f"Unknown CWT period spacing: {spacing}")
    return np.unique(np.round(periods, decimals=6)).astype(np.float64)


def periods_to_scales(periods: np.ndarray, wavelet: str) -> np.ndarray:
    central = float(pywt.central_frequency(wavelet))
    if not np.isfinite(central) or central <= 0:
        raise ValueError(f"Cannot derive CWT central frequency for wavelet: {wavelet}")
    return np.asarray(periods, dtype=np.float64) * central


def cwt_power_cube(
    data: np.ndarray,
    periods: np.ndarray,
    wavelet: str = "cmor1.5-1.0",
    normalize_channels: bool = True,
    method: str = "fft",
    backend: str = "cpu",
    cuda_device: int = 0,
) -> np.ndarray:
    """Return CWT power as `(periods, records, channels)`.

    The default CPU backend is the original PyWavelets implementation. The CUDA
    backend is optional and must be requested explicitly or via `backend="auto"`.
    """
    backend_name = str(backend or "cpu").lower()
    if backend_name == "cpu":
        return _cwt_power_cube_cpu(
            data,
            periods,
            wavelet=wavelet,
            normalize_channels=normalize_channels,
            method=method,
        )
    if backend_name == "cuda":
        from .cwt_cuda import cwt_power_cube_cuda

        return cwt_power_cube_cuda(
            data,
            periods,
            wavelet=wavelet,
            normalize_channels=normalize_channels,
            method=method,
            device=int(cuda_device),
        )
    if backend_name == "auto":
        if method != "fft":
            return _cwt_power_cube_cpu(
                data,
                periods,
                wavelet=wavelet,
                normalize_channels=normalize_channels,
                method=method,
            )
        try:
            from .cwt_cuda import cuda_available, cwt_power_cube_cuda
        except ImportError:
            return _cwt_power_cube_cpu(
                data,
                periods,
                wavelet=wavelet,
                normalize_channels=normalize_channels,
                method=method,
            )

        if cuda_available(device=int(cuda_device)):
            return cwt_power_cube_cuda(
                data,
                periods,
                wavelet=wavelet,
                normalize_channels=normalize_channels,
                method=method,
                device=int(cuda_device),
            )
        return _cwt_power_cube_cpu(
            data,
            periods,
            wavelet=wavelet,
            normalize_channels=normalize_channels,
            method=method,
        )
    raise ValueError(f"Unknown CWT backend: {backend}")


def _cwt_power_cube_cpu(
    data: np.ndarray,
    periods: np.ndarray,
    wavelet: str = "cmor1.5-1.0",
    normalize_channels: bool = True,
    method: str = "fft",
) -> np.ndarray:
    """Return CWT power as `(periods, records, channels)`.

    Periods are expressed in input records. With `sampling_period=1`, PyWavelets
    maps `scale -> frequency`; `period = 1 / frequency`.
    """
    matrix = np.asarray(data, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("data must have shape (records, channels)")
    period_values = np.asarray(periods, dtype=np.float64)
    if period_values.ndim != 1 or period_values.size == 0:
        raise ValueError("periods must be a non-empty 1D array")
    scales = periods_to_scales(period_values, wavelet)
    if normalize_channels:
        matrix = robust_zscore_channels(matrix)
    coeffs, _freqs = pywt.cwt(matrix, scales, wavelet, sampling_period=1.0, method=method, axis=0)
    coeffs = np.asarray(coeffs)
    real = coeffs.real.astype(np.float32, copy=False)
    imag = coeffs.imag.astype(np.float32, copy=False)
    return (real * real + imag * imag).astype(np.float32, copy=False)


def aggregate_cwt_time(
    power: np.ndarray,
    method: str = "p95",
    percentile: float = 95.0,
) -> np.ndarray:
    """Collapse `(period, time, channel)` CWT power to `(period, channel)`."""
    values = np.asarray(power, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("power must have shape (periods, records, channels)")
    if method == "max":
        return np.nanmax(values, axis=1).astype(np.float32)
    if method == "mean":
        return np.nanmean(values, axis=1).astype(np.float32)
    if method == "median":
        return np.nanmedian(values, axis=1).astype(np.float32)
    if method.startswith("p"):
        q = float(method[1:]) if len(method) > 1 else float(percentile)
        return np.nanpercentile(values, q, axis=1).astype(np.float32)
    if method == "percentile":
        return np.nanpercentile(values, float(percentile), axis=1).astype(np.float32)
    raise ValueError(f"Unknown CWT time aggregation method: {method}")
