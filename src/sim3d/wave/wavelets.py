"""Source time functions (spec section 64).

In full-wave mode the wavelet is injected into the wave equation as a
source term; it is never used as a convolution kernel.  The functions here
only build the time series - :mod:`sim3d.wave.acoustic` does the injecting.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import ConfigError


def ricker(t: np.ndarray, f_peak: float, t0: float | None = None) -> np.ndarray:
    r"""Ricker wavelet, the negative second derivative of a Gaussian.

    .. math:: r(t) = \big(1 - 2\pi^2 f^2 \tau^2\big)\, e^{-\pi^2 f^2 \tau^2},
              \quad \tau = t - t_0

    ``f_peak`` is the peak of the amplitude spectrum in Hz.  The default
    delay ``t0 = 1/f_peak`` puts the wavelet's onset near ``t = 0`` with
    negligible truncation (about 1e-4 of peak amplitude).
    """
    if f_peak <= 0:
        raise ConfigError(f"peak frequency must be positive, got {f_peak}")
    t = np.asarray(t, dtype=float)
    t0 = 1.0 / f_peak if t0 is None else float(t0)
    a = (np.pi * f_peak * (t - t0)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def ricker_fmax(f_peak: float, fraction: float = 0.05) -> float:
    """Frequency above which a Ricker spectrum falls below ``fraction`` of its peak.

    The Ricker amplitude spectrum is ``A(f) ~ (f/f_p)^2 exp(1 - (f/f_p)^2)``.
    Solving ``A(f) = fraction`` gives the practical bandwidth used for
    grid-sampling decisions - a Ricker is *not* band-limited at ``f_peak``,
    and sampling the grid for ``f_peak`` alone under-samples the model.
    """
    if not 0 < fraction < 1:
        raise ConfigError(f"fraction must be in (0, 1), got {fraction}")
    # Solve u^2 exp(1 - u^2) = fraction for u = f / f_peak, u > 1.
    lo, hi = 1.0, 20.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid**2 * np.exp(1.0 - mid**2) > fraction:
            lo = mid
        else:
            hi = mid
    return float(hi * f_peak)


def ormsby(t: np.ndarray, f1: float, f2: float, f3: float, f4: float,
           t0: float | None = None) -> np.ndarray:
    """Ormsby trapezoidal band-pass wavelet with corners ``f1<f2<f3<f4`` (Hz)."""
    if not f1 < f2 < f3 < f4:
        raise ConfigError(f"Ormsby corners must increase, got {(f1, f2, f3, f4)}")
    t = np.asarray(t, dtype=float)
    t0 = 0.5 * (t[0] + t[-1]) if t0 is None else float(t0)
    tau = t - t0
    a = (np.pi * f4) ** 2 / (np.pi * (f4 - f3)) * np.sinc(f4 * tau) ** 2
    b = (np.pi * f3) ** 2 / (np.pi * (f4 - f3)) * np.sinc(f3 * tau) ** 2
    c = (np.pi * f2) ** 2 / (np.pi * (f2 - f1)) * np.sinc(f2 * tau) ** 2
    d = (np.pi * f1) ** 2 / (np.pi * (f2 - f1)) * np.sinc(f1 * tau) ** 2
    w = (a - b) - (c - d)
    peak = np.max(np.abs(w))
    return w / peak if peak > 0 else w


def amplitude_spectrum(w: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(frequencies_hz, amplitude)`` of a real time series."""
    w = np.asarray(w, dtype=float)
    spec = np.abs(np.fft.rfft(w))
    freq = np.fft.rfftfreq(w.size, d=dt)
    return freq, spec


def dominant_frequency(w: np.ndarray, dt: float) -> float:
    """Spectral-peak frequency of a time series, in Hz."""
    freq, spec = amplitude_spectrum(w, dt)
    return float(freq[int(np.argmax(spec))])


#: Wavelets ``source.type`` accepts.
WAVELETS = ("ricker", "ormsby")

#: Ormsby corners used when a configuration asks for one without saying which.
DEFAULT_ORMSBY_CORNERS = (5.0, 10.0, 40.0, 50.0)


def ormsby_delay(f1: float) -> float:
    """Onset delay that contains an Ormsby's low-frequency tail.

    An Ormsby is zero-phase and its tails decay only as ``1/tau^2``, so it
    needs far more lead-in than a Ricker.  One period of the lowest corner
    leaves the truncated amplitude around a percent of peak; cutting closer
    puts a step at ``t = 0`` and rings across the whole passband.
    """
    if f1 <= 0:
        raise ConfigError(f"lowest Ormsby corner must be positive, got {f1}")
    return 1.0 / f1


def build_wavelet(t: np.ndarray, kind: str = "ricker", frequency: float = 20.0,
                  corners=None) -> np.ndarray:
    """The source time function a configuration asks for.

    The delay is set here rather than left to each wavelet's own default,
    because the callers build these on arrays as long as a whole record and
    a wavelet centred in the middle of one would fire the shot half a second
    late.
    """
    kind = str(kind).lower()
    if kind not in WAVELETS:
        raise ConfigError(
            f"source type {kind!r} is not available; valid types are "
            f"{sorted(WAVELETS)}")
    if kind == "ricker":
        return ricker(t, frequency)
    f1, f2, f3, f4 = _corners(corners)
    return ormsby(t, f1, f2, f3, f4, t0=ormsby_delay(f1))


def wavelet_fmax(kind: str = "ricker", frequency: float = 20.0, corners=None,
                 fraction: float = 0.05) -> float:
    """Practical maximum frequency, for grid-sampling and aliasing checks.

    An Ormsby is band-limited by construction, so its answer is exactly the
    top corner rather than a spectral-fraction heuristic: asking where a
    trapezoid has fallen to 5% of its peak is asking a question it has
    already answered.
    """
    kind = str(kind).lower()
    if kind not in WAVELETS:
        raise ConfigError(
            f"source type {kind!r} is not available; valid types are "
            f"{sorted(WAVELETS)}")
    if kind == "ricker":
        return ricker_fmax(frequency, fraction)
    return float(_corners(corners)[3])


def _corners(corners) -> tuple[float, float, float, float]:
    if corners is None:
        return DEFAULT_ORMSBY_CORNERS
    values = [float(c) for c in corners]
    if len(values) != 4:
        raise ConfigError(
            f"Ormsby needs four corner frequencies [f1, f2, f3, f4], got "
            f"{len(values)}")
    if not values[0] < values[1] < values[2] < values[3]:
        raise ConfigError(f"Ormsby corners must increase, got {tuple(values)}")
    return tuple(values)
