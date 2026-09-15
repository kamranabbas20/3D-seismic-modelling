"""4D time shifts: the true one, the measurable one, and the alignment.

A velocity change does two things to a monitor survey.  It changes the
reflection *amplitudes* where the properties changed, and it changes the
*traveltime* to everything below - so every reflector under a softened
reservoir arrives late, whether or not its own rock changed at all.  The
second effect is not a nuisance: overburden and underburden time shifts
are a primary 4D observable, and on a compacting field they are often the
only one that survives.

Differencing a monitor against a baseline without accounting for it mixes
the two.  A shifted copy of a wavelet minus the wavelet is a derivative-
shaped difference that can dwarf the real amplitude change and sits at the
wrong depth, which is how a time shift gets interpreted as a fluid front.

This module produces three things:

``true_time_shift``
    Exact, from the two two-way-time cubes the forward model already
    builds by integrating each scenario's own slowness.  A forward model
    knows the answer; estimating something it can compute is a choice to
    be less accurate.  This is the reference.
``estimated_time_shift``
    What a processor would recover: a windowed cross-correlation down each
    trace pair.  Its disagreement with the true shift is the measurement
    error, which is the interesting quantity and cannot be seen without
    both.
``align``
    Warps a monitor back onto the baseline's time axis so the remaining
    difference is amplitude only.

Shifts are in seconds and positive means the monitor arrives *later* - a
slowdown, the classic softening signature.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import ConfigError


def true_time_shift(twt_baseline, twt_monitor, times) -> np.ndarray:
    """Exact shift as a function of baseline two-way time.

    ``twt_*`` are ``(..., nz)`` two-way times down each column, as
    :func:`sim3d.processing.preview.time_from_depth` returns them, and the
    result is on the seismic ``times`` axis: for every sample, how much
    later the monitor's copy of that same *depth* arrives.

    Sampling on the baseline's time axis - not the monitor's, and not
    depth - is what makes the result subtractable from a measured shift
    and applicable to a recorded trace, neither of which knows where any
    depth went.
    """
    base = np.asarray(twt_baseline, dtype=float)
    mon = np.asarray(twt_monitor, dtype=float)
    if base.shape != mon.shape:
        raise ConfigError(f"shapes must match, got {base.shape} and {mon.shape}")
    t = np.asarray(times, dtype=float)
    delta = mon - base
    flat_base = base.reshape(-1, base.shape[-1])
    flat_delta = delta.reshape(-1, delta.shape[-1])
    out = np.empty((flat_base.shape[0], t.size))
    for i in range(flat_base.shape[0]):
        # np.interp needs an increasing x; two-way time down a column is
        # monotonic by construction, but a constant-velocity column can
        # repeat values, which interp tolerates and extrapolation clamps.
        out[i] = np.interp(t, flat_base[i], flat_delta[i])
    return out.reshape(base.shape[:-1] + (t.size,))


def estimated_time_shift(baseline, monitor, dt: float, window: float = 0.12,
                         step: float = 0.02, max_shift: float = 0.05,
                         min_energy: float = 0.05):
    """Windowed cross-correlation shift down each trace pair.

    Returns ``(shifts, centres)``: the shift at each window centre, and the
    centre times.  A single shift per trace - which is what a whole-trace
    correlation gives - cannot represent the thing being measured, because
    the shift accumulates with depth and is flat above the change.

    A window holding less than ``min_energy`` of the trace's RMS reports
    zero rather than the lag that happens to maximise the correlation of
    two pieces of nothing.  The time axis runs from the surface and the
    model does not, so every trace has dead ends; left to guess, they
    return shifts as large as the search range and swamp any average taken
    over the cube.
    """
    base = np.asarray(baseline, dtype=float)
    mon = np.asarray(monitor, dtype=float)
    if base.shape != mon.shape:
        raise ConfigError(f"shapes must match, got {base.shape} and {mon.shape}")
    if dt <= 0:
        raise ConfigError(f"dt must be positive, got {dt}")
    nt = base.shape[-1]
    half = max(int(round(0.5 * window / dt)), 1)
    hop = max(int(round(step / dt)), 1)
    max_lag = max(int(round(max_shift / dt)), 1)
    centres_idx = np.arange(half, max(nt - half, half + 1), hop)
    flat_base = base.reshape(-1, nt)
    flat_mon = mon.reshape(-1, nt)
    reference = np.sqrt(np.mean(flat_base**2, axis=1))
    shifts = np.zeros((flat_base.shape[0], centres_idx.size))
    for j, c in enumerate(centres_idx):
        lo, hi = c - half, c + half + 1
        a = flat_base[:, lo:hi]
        b = flat_mon[:, lo:hi]
        live = (np.sqrt(np.mean(a**2, axis=1)) > min_energy * reference) & \
               (np.sqrt(np.mean(b**2, axis=1)) > min_energy * reference)
        if live.any():
            shifts[live, j] = _window_shift(a[live], b[live], max_lag) * dt
    return (shifts.reshape(base.shape[:-1] + (centres_idx.size,)),
            centres_idx * dt)


def _window_shift(a: np.ndarray, b: np.ndarray, max_lag: int) -> np.ndarray:
    """Sub-sample lag maximising correlation, for a stack of window pairs."""
    lags = np.arange(-max_lag, max_lag + 1)
    scores = np.empty((a.shape[0], lags.size))
    for k, lag in enumerate(lags):
        if lag > 0:
            x, y = a[:, :-lag], b[:, lag:]
        elif lag < 0:
            x, y = a[:, -lag:], b[:, :lag]
        else:
            x, y = a, b
        num = np.einsum("ij,ij->i", x, y)
        den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
        scores[:, k] = np.where(den > 0, num / np.where(den > 0, den, 1.0), -np.inf)
    peak = np.argmax(scores, axis=1)
    out = lags[peak].astype(float)
    # Parabolic refinement, skipped at the ends where there is no triple.
    inner = (peak > 0) & (peak < lags.size - 1)
    if inner.any():
        rows = np.nonzero(inner)[0]
        p = peak[rows]
        y0 = scores[rows, p - 1]
        y1 = scores[rows, p]
        y2 = scores[rows, p + 1]
        den = y0 - 2.0 * y1 + y2
        out[rows] += np.where(den != 0, 0.5 * (y0 - y2) / np.where(den != 0, den, 1.0), 0.0)
    return out


def align(cube, shifts, dt: float) -> np.ndarray:
    """Warp a monitor onto the baseline's time axis.

    ``shifts`` is per sample, as :func:`true_time_shift` returns it: the
    monitor's sample for baseline time ``t`` sits at ``t + shift(t)``, so
    undoing the shift means *reading* the monitor there.
    """
    data = np.asarray(cube, dtype=float)
    lag = np.asarray(shifts, dtype=float)
    if lag.shape != data.shape:
        raise ConfigError(
            f"a shift per sample is required, got {lag.shape} for {data.shape}")
    nt = data.shape[-1]
    t = np.arange(nt) * dt
    flat = data.reshape(-1, nt)
    flat_lag = lag.reshape(-1, nt)
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        out[i] = np.interp(t + flat_lag[i], t, flat[i])
    return out.reshape(data.shape)


def resample_shift(shifts, centres, times) -> np.ndarray:
    """Put a windowed shift back on the full sample axis, for :func:`align`."""
    s = np.asarray(shifts, dtype=float)
    c = np.asarray(centres, dtype=float)
    t = np.asarray(times, dtype=float)
    flat = s.reshape(-1, s.shape[-1])
    out = np.array([np.interp(t, c, row) for row in flat])
    return out.reshape(s.shape[:-1] + (t.size,))
