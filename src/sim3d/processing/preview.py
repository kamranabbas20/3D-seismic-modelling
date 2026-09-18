"""Fast 1D convolution preview mode (spec sections 82-83).

This is the *screening* path, not the scientific one.  It exists so that
thousands of rock-physics cases can be triaged cheaply before a handful are
put through full-wave modelling and RTM, and so that a convolution result
can be compared side by side with a migrated one to show what the
approximation costs.

Everything it produces carries :data:`PREVIEW_LABEL`.  Nothing in this
module may be described as a seismic image: it is a property cube filtered
by a wavelet.

The chain is::

    Vp, rho -> AI -> vertical two-way time -> normal-incidence reflectivity
            -> convolution with the wavelet -> optional mapping back to depth

What it contains: vertical reflectivity, bandwidth, and tuning between
closely spaced interfaces.

What it does not contain, and cannot: lateral wave propagation, diffraction,
refraction and head waves, transmission losses, geometric spreading,
illumination and acquisition effects, migration, and every finite-frequency
phenomenon that is not one-dimensional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.errors import ConfigError
from ..core.grid import Grid3D
from ..wave.acoustic import AcousticModel

PREVIEW_LABEL = ("Fast 1D/Convolution Approximation - Not Full 3D Wave Modelling")


def time_from_depth(vp: np.ndarray, dz: float) -> np.ndarray:
    """Vertical two-way traveltime at each node, seconds.

    Integrates ``2 dz / Vp`` down each column with the trapezoidal rule, so
    the time of the first sample is zero and each interval contributes the
    average slowness of its two ends.

    The last axis is depth; everything before it is columns, so this takes
    a ``(nx, ny, nz)`` cube or a ``(K, nz)`` handful of well locations.
    """
    slowness = 2.0 / np.asarray(vp, dtype=float)
    interval = 0.5 * (slowness[..., 1:] + slowness[..., :-1]) * dz
    return np.concatenate(
        [np.zeros(slowness.shape[:-1] + (1,)), np.cumsum(interval, axis=-1)], axis=-1
    )


def reflectivity(impedance: np.ndarray) -> np.ndarray:
    """Normal-incidence reflection coefficients between vertical neighbours.

    ``R_k = (AI_{k+1} - AI_k) / (AI_{k+1} + AI_k)``, returned on the
    interfaces, so the last axis is one shorter than the input.
    """
    ai = np.asarray(impedance, dtype=float)
    upper, lower = ai[..., :-1], ai[..., 1:]
    return (lower - upper) / (lower + upper)


def synthetic_columns(twt: np.ndarray, impedance: np.ndarray, wavelet: np.ndarray,
                      dt: float, nt: int, map_to_depth: bool = True):
    """Impedance -> reflectivity -> the chain in :func:`convolve_reflectivity`."""
    return convolve_reflectivity(twt, reflectivity(impedance), wavelet, dt, nt,
                                 map_to_depth=map_to_depth)


def convolve_reflectivity(twt: np.ndarray, rc: np.ndarray, wavelet: np.ndarray,
                          dt: float, nt: int, map_to_depth: bool = True):
    """Reflectivity -> time series -> convolution -> optional depth mapping.

    The shared core of every synthetic mode.  ``twt`` is ``(..., nz)`` and
    ``rc`` is ``(..., nz - 1)``, defined on the interfaces between depth
    nodes; every axis before the last is an independent column, so the same
    code serves the full cube, an angle stack and a handful of well
    locations, and the timing convention below cannot drift between them.

    Taking reflectivity rather than impedance is what lets the angle-
    dependent modes in :mod:`sim3d.processing.sim2seis` share this chain:
    their coefficients are not the contrast of any single quantity.

    Returns ``(traces, times, depth_traces)``, the last being ``None`` when
    ``map_to_depth`` is false.
    """
    if rc.shape[-1] != twt.shape[-1] - 1:
        raise ConfigError(
            f"reflectivity is defined on interfaces, so its last axis must be "
            f"one shorter than the depth axis; got {rc.shape} against {twt.shape}")
    # Place each coefficient at the midpoint time of its interface.
    rc_time = 0.5 * (twt[..., 1:] + twt[..., :-1])

    lead = twt.shape[:-1]
    n_col = int(np.prod(lead)) if lead else 1
    rc_flat = rc.reshape(n_col, -1)
    index = np.clip(np.round(rc_time / dt).astype(int), 0, nt - 1).reshape(n_col, -1)

    # ``bincount`` rather than ``np.add.at``, which is an order of magnitude
    # slower; both accumulate coefficients that land in the same sample.
    series = np.stack([np.bincount(index[i], weights=rc_flat[i], minlength=nt)
                       for i in range(n_col)])

    w = np.asarray(wavelet, dtype=float)
    # Full convolution truncated to the input length, not ``mode="same"``:
    # centring on the full convolution would shift every event earlier by
    # half the wavelet length. The wavelet's own delay is kept, so the
    # timing convention matches the finite-difference solver, where the
    # source wavelet carries the same delay.
    traces = np.stack([np.convolve(s, w)[:nt] for s in series])
    times = np.arange(nt) * dt

    depth_traces = None
    if map_to_depth:
        # Sample the trace at ``twt + delay``, not at ``twt``.  The trace
        # deliberately keeps the wavelet's own delay so its time axis matches
        # the finite-difference solver's, which means the event from an
        # interface peaks a delay *after* that interface's two-way time.
        # Reading the trace at ``twt`` therefore lands every reflector too
        # deep by roughly ``delay * V / 2`` - 125 m for a 12 Hz Ricker at
        # 2,500 m/s, which puts a reservoir event below the reservoir.
        delay = float(np.argmax(np.abs(w)) * dt) if w.size else 0.0
        twt_flat = twt.reshape(n_col, -1) + delay
        depth_traces = np.stack(
            [np.interp(twt_flat[i], times, traces[i]) for i in range(n_col)]
        ).reshape(lead + (twt.shape[-1],))
    return traces.reshape(lead + (nt,)), times, depth_traces


@dataclass
class ConvolutionPreview:
    """The output of :func:`convolution_preview`, labelled for what it is."""

    time_traces: np.ndarray       #: ``(nx, ny, nt)`` in the time domain
    times: np.ndarray             #: time axis, seconds
    depth_traces: np.ndarray      #: the same cube mapped back onto the depth grid
    grid: Grid3D
    label: str = PREVIEW_LABEL

    def describe(self) -> str:
        return (f"{self.label}\n"
                f"  cube {self.depth_traces.shape}, "
                f"{self.times.size} time samples at "
                f"{(self.times[1] - self.times[0]) * 1e3:.2f} ms")


def convolution_preview(model: AcousticModel, wavelet: np.ndarray, dt: float,
                        t_max: float | None = None,
                        map_to_depth: bool = True) -> ConvolutionPreview:
    """Build a 1D convolution cube from a Vp/rho model.

    Parameters
    ----------
    model:
        The earth model to screen.
    wavelet:
        Source wavelet sampled at ``dt``.
    dt:
        Time sample interval, seconds.
    t_max:
        Length of the time axis; defaults to the deepest two-way time in
        the model.
    map_to_depth:
        Also resample the result back onto the depth grid, which is what
        allows a direct comparison with an RTM image.
    """
    if dt <= 0:
        raise ConfigError(f"dt must be positive, got {dt}")
    grid = model.grid
    twt = time_from_depth(model.vp, grid.dz)
    t_max = float(t_max if t_max is not None else twt.max())
    nt = int(np.ceil(t_max / dt)) + 1

    traces, times, depth = synthetic_columns(twt, model.impedance, wavelet, dt, nt,
                                             map_to_depth=map_to_depth)
    depth_traces = depth if depth is not None else np.zeros(grid.shape)
    return ConvolutionPreview(time_traces=traces, times=times,
                              depth_traces=depth_traces, grid=grid)
