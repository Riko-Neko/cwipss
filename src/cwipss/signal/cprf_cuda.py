"""CUDA implementation of the Concentrated Periodic Ridge Filter."""

from __future__ import annotations

import numpy as np

from .cprf import CPRFParameters, CPRFResult, MIN_POSITIVE


def _cupy_modules():
    try:
        import cupy as cp
        from cupyx.scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("CPRF CUDA requires CuPy with cupyx.scipy") from exc
    return cp, ndimage


def cprf_normalization_threshold_cuda(
    power,
    *,
    noise_std,
    noise_gain: np.ndarray,
    params: CPRFParameters,
):
    """Compute the independent CPRF normalization threshold on CUDA."""
    params.validate()
    cp, _ndimage = _cupy_modules()
    values = cp.asarray(power, dtype=cp.float32)
    gain = cp.asarray(noise_gain, dtype=cp.float32)
    if values.ndim != 2 or gain.shape != (values.shape[0],):
        raise ValueError("power and noise_gain must match on the period axis")
    sigma = cp.asarray(noise_std, dtype=cp.float64)
    if not bool(cp.isfinite(sigma) & (sigma > 0.0)):
        raise ValueError("noise_std must be finite and positive")
    calibrated = cp.maximum(cp.where(cp.isfinite(values), values, 0.0), 0.0) / cp.maximum(
        sigma**2 * gain[:, None], MIN_POSITIVE
    )
    threshold = cp.asarray(params.threshold_snr, dtype=cp.float64)
    if params.texture_quantile > 0.0:
        threshold = cp.maximum(threshold, cp.quantile(calibrated, params.texture_quantile))
    return threshold


def _masked_median(profile, mask, cp):
    count = cp.sum(mask, axis=1, dtype=cp.int64)
    ordered = cp.sort(cp.where(mask, profile[None, :], cp.inf), axis=1)
    lower = cp.maximum((count - 1) // 2, 0)
    upper = count // 2
    lower_value = cp.take_along_axis(ordered, lower[:, None], axis=1)[:, 0]
    upper_value = cp.take_along_axis(ordered, upper[:, None], axis=1)[:, 0]
    return cp.where(count > 0, 0.5 * (lower_value + upper_value), 0.0), count


def _harmonic_scores(
    profile,
    periods,
    peaks,
    band_mask,
    *,
    order: int,
    window_scale: float,
    cp,
):
    frequencies = 1.0 / periods
    center_frequency = float(order) / periods[peaks]
    band_max = cp.max(cp.where(band_mask, frequencies[None, :], -cp.inf), axis=1)
    band_min = cp.min(cp.where(band_mask, frequencies[None, :], cp.inf), axis=1)
    main_half_width = 0.5 * cp.abs(band_max - band_min)
    if int(frequencies.size) > 1:
        nearest = cp.argmin(cp.abs(frequencies[None, :] - center_frequency[:, None]), axis=1)
        neighbor = cp.where(nearest == 0, 1, nearest - 1)
        grid_half_width = 0.5 * cp.abs(frequencies[nearest] - frequencies[neighbor])
    else:
        grid_half_width = cp.zeros_like(main_half_width)
    half_width = cp.maximum(float(order) * main_half_width * window_scale, grid_half_width)
    harmonic_mask = cp.abs(frequencies[None, :] - center_frequency[:, None]) <= half_width[:, None]
    harmonic_peak = cp.max(cp.where(harmonic_mask, profile[None, :], 0.0), axis=1)
    return cp.clip(harmonic_peak / cp.maximum(profile[peaks], MIN_POSITIVE), 0.0, 1.0)


def evaluate_cprf_cuda(
    power_window,
    periods,
    *,
    noise_std,
    noise_gain,
    normalization_threshold,
    params: CPRFParameters | None = None,
) -> CPRFResult:
    """Evaluate one PELT window on CUDA and return only final scalars."""
    params = params or CPRFParameters()
    params.validate()
    cp, ndimage = _cupy_modules()
    power = cp.asarray(power_window, dtype=cp.float32)
    period_values = cp.asarray(periods, dtype=cp.float64)
    gain = cp.asarray(noise_gain, dtype=cp.float32)
    if power.ndim != 2 or power.shape[0] != period_values.size or power.shape[1] == 0:
        raise ValueError("power_window must have shape (periods, non-empty records)")
    if gain.shape != (power.shape[0],):
        raise ValueError("noise_gain must match power_window's period axis")

    values = cp.maximum(cp.where(cp.isfinite(power), power, 0.0), 0.0) / cp.maximum(
        cp.asarray(noise_std, dtype=cp.float64) ** 2
        * gain[:, None]
        * cp.asarray(normalization_threshold, dtype=cp.float64),
        MIN_POSITIVE,
    )
    occupied = values >= 1.0
    occupancy = cp.mean(occupied, axis=1, dtype=cp.float64).astype(cp.float32)
    occupied_mean = cp.sum(cp.where(occupied, values, 0.0), axis=1, dtype=cp.float64) / cp.maximum(
        cp.count_nonzero(occupied, axis=1), 1
    )
    profile = (occupied_mean * cp.sqrt(occupancy)).astype(cp.float32)
    if params.smooth_bins > 1:
        profile = ndimage.uniform_filter1d(
            profile,
            size=min(int(params.smooth_bins), int(profile.size)),
            mode="nearest",
        ).astype(cp.float32, copy=False)

    if int(profile.size) == 1:
        peaks = cp.zeros(1, dtype=cp.int64)
    else:
        peak_mask = cp.concatenate(
            (
                (profile[:1] >= profile[1:2]),
                (profile[1:-1] > profile[:-2]) & (profile[1:-1] >= profile[2:]),
                (profile[-1:] > profile[-2:-1]),
            )
        )
        peaks = cp.flatnonzero(peak_mask)
        if int(peaks.size) == 0:
            peaks = cp.argmax(profile)[None]
    peaks = peaks[cp.argsort(profile[peaks])[::-1]][: params.max_peak_hypotheses]

    positive = profile[profile > 0.0]
    global_baseline = cp.median(positive) if int(positive.size) else cp.asarray(0.0, dtype=cp.float32)
    band_threshold = global_baseline + params.peak_band_fraction * cp.maximum(
        profile[peaks] - global_baseline, 0.0
    )
    period_indices = cp.arange(profile.size, dtype=cp.int64)
    above = profile[None, :] >= band_threshold[:, None]
    left_blocker = cp.max(
        cp.where((period_indices[None, :] < peaks[:, None]) & ~above, period_indices[None, :], -1),
        axis=1,
    )
    right_blocker = cp.min(
        cp.where(
            (period_indices[None, :] > peaks[:, None]) & ~above,
            period_indices[None, :],
            int(profile.size),
        ),
        axis=1,
    )
    starts = left_blocker + 1
    stops = right_blocker
    widths = cp.maximum(stops - starts, 1)
    band_mask = (period_indices[None, :] >= starts[:, None]) & (
        period_indices[None, :] < stops[:, None]
    )
    side_mask = (
        (period_indices[None, :] >= (starts - 2 * widths)[:, None])
        & (period_indices[None, :] < starts[:, None])
    ) | (
        (period_indices[None, :] >= stops[:, None])
        & (period_indices[None, :] < (stops + 2 * widths)[:, None])
    )
    baseline, side_count = _masked_median(profile, side_mask, cp)
    outside_mask = ~band_mask
    outside_baseline, _outside_count = _masked_median(profile, outside_mask, cp)
    baseline = cp.where(side_count > 0, baseline, outside_baseline)

    band_values = cp.where(band_mask, profile[None, :], 0.0)
    band_sum = cp.sum(band_values, axis=1, dtype=cp.float64)
    band_mean = band_sum / widths
    excess = cp.where(band_mask, cp.maximum(profile[None, :] - baseline[:, None], 0.0), 0.0)
    peak_strength = cp.maximum(profile[peaks] - baseline, 0.0)
    integrated_strength = cp.sum(excess, axis=1, dtype=cp.float64) / cp.sqrt(widths)
    concentration = band_sum / cp.maximum(cp.sum(profile, dtype=cp.float64), MIN_POSITIVE)
    persistence = cp.sum(
        cp.where(band_mask, occupancy[None, :] * cp.maximum(profile[None, :], 0.0), 0.0),
        axis=1,
        dtype=cp.float64,
    ) / cp.maximum(cp.sum(cp.maximum(band_values, 0.0), axis=1, dtype=cp.float64), MIN_POSITIVE)
    local_contrast = (band_mean - baseline) / cp.maximum(cp.abs(baseline), 1.0)
    harmonic_2 = _harmonic_scores(
        profile,
        period_values,
        peaks,
        band_mask,
        order=2,
        window_scale=params.harmonic_window_scale,
        cp=cp,
    )
    harmonic_3 = _harmonic_scores(
        profile,
        period_values,
        peaks,
        band_mask,
        order=3,
        window_scale=params.harmonic_window_scale,
        cp=cp,
    )
    base_score = (
        cp.maximum(peak_strength, MIN_POSITIVE) ** 0.40
        * cp.maximum(integrated_strength, MIN_POSITIVE) ** 0.30
        * cp.maximum(concentration, MIN_POSITIVE) ** 0.15
        * cp.maximum(persistence, MIN_POSITIVE) ** 0.15
    )
    total_score = base_score * (
        1.0 + params.harmonic_weight * 0.5 * (harmonic_2 + harmonic_3)
    )
    selected = cp.argmax(total_score)
    accepted = (
        (widths[selected] >= params.min_width_bins)
        & (peak_strength[selected] >= params.min_peak_strength)
        & (integrated_strength[selected] >= params.min_integrated_strength)
        & (persistence[selected] >= params.min_band_persistence)
        & (concentration[selected] >= params.min_band_concentration)
        & (local_contrast[selected] >= params.min_local_contrast)
    )
    peak = peaks[selected]
    start = starts[selected]
    stop = stops[selected]
    packed = cp.stack(
        (
            accepted,
            peak,
            start,
            stop,
            widths[selected],
            peak_strength[selected],
            integrated_strength[selected],
            concentration[selected],
            persistence[selected],
            local_contrast[selected],
            harmonic_2[selected],
            harmonic_3[selected],
            base_score[selected],
            total_score[selected],
            cp.asarray(normalization_threshold),
        )
    )
    host = cp.asnumpy(packed)
    peak_index = int(host[1])
    start_index = int(host[2])
    stop_index = int(host[3])
    period_host = np.asarray(periods, dtype=np.float64)
    harmonic_support = int(host[10] >= params.harmonic_min_relative) + int(
        host[11] >= params.harmonic_min_relative
    )
    return CPRFResult(
        accepted=bool(host[0]),
        peak_period_records=float(period_host[peak_index]),
        period_start_records=float(period_host[start_index]),
        period_stop_records=float(period_host[stop_index - 1]),
        peak_index=peak_index,
        band_start_index=start_index,
        band_stop_index=stop_index,
        width_bins=int(host[4]),
        peak_strength=float(host[5]),
        integrated_strength=float(host[6]),
        band_concentration=float(host[7]),
        band_persistence=float(host[8]),
        local_contrast=float(host[9]),
        harmonic_2_score=float(host[10]),
        harmonic_3_score=float(host[11]),
        harmonic_support_count=harmonic_support,
        base_score=float(host[12]),
        total_score=float(host[13]),
        normalization_threshold=float(host[14]),
        profile=None,
    )
