"""Light processing for shot gathers (spec section 84).

Deliberately minimal: this platform is not trying to reproduce an
industrial processing sequence.  What is here is what a mechanistic 4D
experiment actually needs - muting the direct arrival, band-limiting, and
display gain.

Display gain is display gain.  AGC and normalisation exist for looking at
gathers and must never be applied to data that will be migrated or
differenced, because they destroy the amplitude relationships a 4D
difference is made of.  Every function returns a new array; the raw record
is never modified.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import ConfigError


def bandpass(traces: np.ndarray, dt: float, low: float, high: float,
             taper: float = 5.0) -> np.ndarray:
    """Zero-phase trapezoidal band-pass along the last axis.

    Applied in the frequency domain with cosine tapers of width ``taper`` Hz
    on each corner, so it adds no phase and therefore no apparent time shift
    to a 4D difference.
    """
    if not 0 <= low < high:
        raise ConfigError(f"require 0 <= low < high, got ({low}, {high})")
    data = np.asarray(traces, dtype=float)
    n = data.shape[-1]
    freq = np.fft.rfftfreq(n, d=dt)
    gain = np.ones_like(freq)
    gain[freq < low - taper] = 0.0
    gain[freq > high + taper] = 0.0
    rising = (freq >= low - taper) & (freq < low)
    falling = (freq > high) & (freq <= high + taper)
    if taper > 0:
        gain[rising] = 0.5 * (1 - np.cos(np.pi * (freq[rising] - (low - taper)) / taper))
        gain[falling] = 0.5 * (1 + np.cos(np.pi * (freq[falling] - high) / taper))
    return np.fft.irfft(np.fft.rfft(data, axis=-1) * gain, n=n, axis=-1)


def direct_wave_mute(traces: np.ndarray, offsets: np.ndarray, dt: float,
                     velocity: float, pad: float = 0.05,
                     taper_samples: int = 20) -> np.ndarray:
    """Mute everything before ``offset / velocity + pad`` on each trace.

    The direct arrival dominates a short-offset gather and carries no
    reservoir information, so removing it before migration keeps the image
    from being swamped by the injection points.
    """
    data = np.asarray(traces, dtype=float).copy()
    off = np.asarray(offsets, dtype=float).ravel()
    if off.size != data.shape[0]:
        raise ConfigError(
            f"{off.size} offsets for {data.shape[0]} traces"
        )
    nt = data.shape[-1]
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(taper_samples) / max(taper_samples, 1)))
    for i, offset in enumerate(off):
        cut = int(np.ceil((offset / velocity + pad) / dt))
        cut = max(0, min(cut, nt))
        data[i, :cut] = 0.0
        end = min(cut + taper_samples, nt)
        data[i, cut:end] *= ramp[: end - cut]
    return data


def taper_mute(traces: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Zero samples outside ``[start, stop)`` with a cosine edge."""
    data = np.asarray(traces, dtype=float).copy()
    data[..., :start] = 0.0
    data[..., stop:] = 0.0
    return data


def agc(traces: np.ndarray, window_samples: int) -> np.ndarray:
    """Automatic gain control - **for display only**.

    Divides each sample by the RMS in a sliding window.  This destroys the
    amplitude information that a 4D difference is made of, so it must never
    be applied to data that will be migrated or differenced.
    """
    if window_samples < 2:
        raise ConfigError(f"AGC window must be >= 2 samples, got {window_samples}")
    data = np.asarray(traces, dtype=float)
    kernel = np.ones(window_samples) / window_samples
    power = np.apply_along_axis(
        lambda s: np.convolve(s**2, kernel, mode="same"), -1, data)
    scale = np.sqrt(np.where(power > 0, power, 1.0))
    return data / scale


def normalise(traces: np.ndarray, per_trace: bool = True) -> np.ndarray:
    """Scale to unit maximum - **for display only**, for the same reason as AGC."""
    data = np.asarray(traces, dtype=float)
    axis = -1 if per_trace else None
    peak = np.max(np.abs(data), axis=axis, keepdims=per_trace)
    return data / np.where(peak > 0, peak, 1.0)
