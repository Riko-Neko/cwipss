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


def pad_to_swt_multiple(values: np.ndarray, levels: int) -> tuple[np.ndarray, int]:
    """Pad a 1D series so PyWavelets SWT divisibility requirements are met."""
    x = np.asarray(values, dtype=np.float32)
    multiple = 2 ** int(levels)
    pad = (-x.size) % multiple
    if pad == 0:
        return x, 0
    mode = "reflect" if x.size > 2 else "edge"
    return np.pad(x, (0, pad), mode=mode).astype(np.float32), int(pad)


def swt_detail_power_matrix(
    data: np.ndarray,
    wavelet: str = "db4",
    levels: int = 5,
    normalize_channels: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return SWT detail power as `(levels, records, channels)`.

    SWT does not downsample. The returned level numbers are the physical SWT
    levels reported from coarse to fine by PyWavelets, e.g. `[5, 4, ..., 1]`.
    """
    if levels < 1:
        raise ValueError("levels must be >= 1")
    pywt.Wavelet(wavelet)
    matrix = np.asarray(data, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("data must have shape (records, channels)")

    n_records, n_channels = matrix.shape
    powers = np.empty((levels, n_records, n_channels), dtype=np.float32)
    for channel_idx in range(n_channels):
        series = matrix[:, channel_idx]
        if normalize_channels:
            series = robust_zscore(series)
        padded, pad = pad_to_swt_multiple(series, levels)
        coeffs = pywt.swt(padded, wavelet=wavelet, level=levels, trim_approx=False, norm=True)
        for level_idx, (_approx, detail) in enumerate(coeffs):
            detail = np.asarray(detail, dtype=np.float32)
            if pad:
                detail = detail[:-pad]
            powers[level_idx, :, channel_idx] = detail[:n_records] ** 2
    level_numbers = np.arange(levels, 0, -1, dtype=np.int16)
    return powers, level_numbers


def approximate_scale_records(level_number: int) -> int:
    """Approximate SWT scale in records. Validation must refine true periods."""
    return int(2 ** int(level_number))
