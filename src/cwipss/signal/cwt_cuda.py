"""CUDA implementation of the CWT compute backend."""

from __future__ import annotations

from functools import lru_cache
from math import ceil, floor
from typing import Any

import numpy as np
from pywt._cwt import integrate_wavelet, next_fast_len
from pywt._extensions._pywt import ContinuousWavelet, DiscreteContinuousWavelet, Wavelet, _check_dtype

from .cwt import MIN_ROBUST_SCALE, periods_to_scales


def _cupy() -> Any:
    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError("CuPy is required for cwt_backend='cuda'.") from exc
    return cp


def cuda_available(device: int = 0) -> bool:
    try:
        cp = _cupy()
        cp.cuda.Device(int(device)).use()
        cp.cuda.runtime.getDevice()
        return True
    except Exception:
        return False


def _wavelet_object(wavelet: str | ContinuousWavelet | Wavelet):
    if isinstance(wavelet, (ContinuousWavelet, Wavelet)):
        return wavelet
    return DiscreteContinuousWavelet(wavelet)


def _build_integrated_wavelet(wavelet: str | ContinuousWavelet | Wavelet, data_dtype: np.dtype):
    wavelet_obj = _wavelet_object(wavelet)
    dt_cplx = np.result_type(data_dtype, np.complex64)
    precision = 10
    int_psi, x = integrate_wavelet(wavelet_obj, precision=precision)
    int_psi = np.conj(int_psi) if wavelet_obj.complex_cwt else int_psi
    dt_psi = dt_cplx if int_psi.dtype.kind == "c" else data_dtype
    return (
        wavelet_obj,
        np.asarray(int_psi, dtype=dt_psi),
        np.asarray(x, dtype=np.dtype(data_dtype).type),
    )


@lru_cache(maxsize=16)
def _cached_integrated_wavelet(wavelet: str, dtype_string: str):
    wavelet_obj, int_psi, x = _build_integrated_wavelet(wavelet, np.dtype(dtype_string))
    int_psi.setflags(write=False)
    x.setflags(write=False)
    return wavelet_obj, int_psi, x


def _integrated_wavelet(wavelet: str | ContinuousWavelet | Wavelet, data_dtype: np.dtype):
    if isinstance(wavelet, str):
        return _cached_integrated_wavelet(wavelet, np.dtype(data_dtype).str)
    return _build_integrated_wavelet(wavelet, data_dtype)


def _scaled_integrated_wavelet(int_psi: np.ndarray, x: np.ndarray, scale: float) -> np.ndarray:
    step = x[1] - x[0]
    j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step)
    j = j.astype(int)
    if j[-1] >= int_psi.size:
        j = np.extract(j < int_psi.size, j)
    return int_psi[j][::-1]


def cwt_power_cube_cuda(
    data: np.ndarray,
    periods: np.ndarray,
    wavelet: str = "cmor1.5-1.0",
    normalize_channels: bool = True,
    method: str = "fft",
    device: int = 0,
) -> np.ndarray:
    """Return CWT power via a CuPy FFT backend.

    This mirrors the PyWavelets FFT CWT construction used by the CPU backend:
    integrated wavelet sampling, FFT convolution, finite differencing, and
    center cropping. This compatibility API explicitly returns a NumPy
    `float32` power cube. Device-resident consumers can use `cwt_power_cube_cuda_gpu`
    instead, so its period-time cube remains on CUDA.
    """
    if method != "fft":
        raise ValueError("CUDA CWT backend currently supports method='fft' only.")
    cp = _cupy()
    power_gpu = cwt_power_cube_cuda_gpu(
        data,
        periods,
        wavelet=wavelet,
        normalize_channels=normalize_channels,
        method=method,
        device=device,
    )
    return cp.asnumpy(power_gpu)


def cwt_power_cube_cuda_gpu(
    data,
    periods: np.ndarray,
    wavelet: str = "cmor1.5-1.0",
    normalize_channels: bool = True,
    method: str = "fft",
    device: int = 0,
):
    """Return CWT power as a CuPy `(periods, records, channels)` array."""
    if method != "fft":
        raise ValueError("CUDA CWT backend currently supports method='fft' only.")
    cp = _cupy()
    cp.cuda.Device(int(device)).use()
    input_is_cuda = isinstance(data, cp.ndarray)
    if input_is_cuda:
        matrix_gpu = cp.asarray(data, dtype=cp.float32)
        if matrix_gpu.ndim != 2:
            raise ValueError("data must have shape (records, channels)")
        data_dtype = np.dtype(np.float32)
        records, channels = map(int, matrix_gpu.shape)
    else:
        matrix = np.asarray(data, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("data must have shape (records, channels)")
        data_dtype = np.dtype(_check_dtype(matrix))
        matrix = np.asarray(matrix, dtype=data_dtype)
        records, channels = matrix.shape
    period_values = np.asarray(periods, dtype=np.float64)
    if period_values.ndim != 1 or period_values.size == 0:
        raise ValueError("periods must be a non-empty 1D array")
    scales = periods_to_scales(period_values, wavelet)
    wavelet_obj, int_psi, x = _integrated_wavelet(wavelet, data_dtype)

    out = cp.empty((period_values.size, records, channels), dtype=cp.float32)
    # PyWavelets transforms axis 0 by swapping it to the last axis and
    # batching all channels as independent 1D signals.
    if input_is_cuda:
        data_gpu = cp.ascontiguousarray(matrix_gpu.T)
    else:
        data_gpu = cp.asarray(np.ascontiguousarray(matrix.T))
    if normalize_channels:
        median = cp.nanmedian(data_gpu, axis=1, keepdims=True)
        centered = data_gpu - median
        mad = cp.nanmedian(cp.abs(centered), axis=1, keepdims=True)
        scale = 1.4826 * mad
        valid_scale = cp.isfinite(scale)
        scale = cp.where(valid_scale, cp.maximum(scale, MIN_ROBUST_SCALE), cp.nan)
        data_gpu = cp.where(valid_scale & cp.isfinite(data_gpu), centered / scale, 0.0)
        data_gpu = cp.clip(data_gpu, -8.0, 8.0).astype(cp.float32, copy=False)
    fft_data = None
    size_scale0 = -1
    for scale_index, scale in enumerate(scales):
        int_psi_scale = _scaled_integrated_wavelet(int_psi, x, float(scale))
        size_scale = next_fast_len(records + int_psi_scale.size - 1)
        if size_scale != size_scale0:
            fft_data = cp.fft.fft(data_gpu, size_scale, axis=-1)
            size_scale0 = size_scale
        wav_gpu = cp.asarray(int_psi_scale)
        fft_wav = cp.fft.fft(wav_gpu, size_scale, axis=-1)
        conv = cp.fft.ifft(fft_wav[cp.newaxis, :] * fft_data, axis=-1)
        conv = conv[..., : records + int_psi_scale.size - 1]
        coef = -cp.sqrt(cp.asarray(scale, dtype=data_gpu.real.dtype)) * cp.diff(conv, axis=-1)
        if not wavelet_obj.complex_cwt:
            coef = coef.real
        d = (coef.shape[-1] - records) / 2.0
        if d > 0:
            coef = coef[..., floor(d):-ceil(d)]
        elif d < 0:
            raise ValueError(f"Selected scale of {scale} too small.")
        real = coef.real.astype(cp.float32, copy=False)
        if coef.dtype.kind == "c":
            imag = coef.imag.astype(cp.float32, copy=False)
            power = real * real + imag * imag
        else:
            power = real * real
        out[scale_index, :, :] = power.T.astype(cp.float32, copy=False)
        del wav_gpu, fft_wav, conv, coef, real, power
    del data_gpu, fft_data
    return out
