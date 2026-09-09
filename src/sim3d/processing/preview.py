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
    """
    slowness = 2.0 / np.asarray(vp, dtype=float)
    interval = 0.5 * (slowness[:, :, 1:] + slowness[:, :, :-1]) * dz
    return np.concatenate(
        [np.zeros(vp.shape[:2] + (1,)), np.cumsum(interval, axis=2)], axis=2
    )


def reflectivity(impedance: np.ndarray) -> np.ndarray:
    """Normal-incidence reflection coefficients between vertical neighbours.

    ``R_k = (AI_{k+1} - AI_k) / (AI_{k+1} + AI_k)``, returned on the
    interfaces, so the last axis is one shorter than the input.
    """
    ai = np.asarray(impedance, dtype=float)
    upper, lower = ai[:, :, :-1], ai[:, :, 1:]
    return (lower - upper) / (lower + upper)


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
    ai = model.impedance
    twt = time_from_depth(model.vp, grid.dz)
    rc = reflectivity(ai)
    # Place each coefficient at the midpoint time of its interface.
    rc_time = 0.5 * (twt[:, :, 1:] + twt[:, :, :-1])

    t_max = float(t_max if t_max is not None else twt.max())
    nt = int(np.ceil(t_max / dt)) + 1
    times = np.arange(nt) * dt

    nx, ny = grid.nx, grid.ny
    series = np.zeros((nx, ny, nt))
    index = np.clip(np.round(rc_time / dt).astype(int), 0, nt - 1)
    for ix in range(nx):
        for iy in range(ny):
            np.add.at(series[ix, iy], index[ix, iy], rc[ix, iy])

    w = np.asarray(wavelet, dtype=float)
    # Full convolution truncated to the input length, not ``mode="same"``:
    # centring on the full convolution would shift every event earlier by
    # half the wavelet length. The wavelet's own delay is kept, so the
    # timing convention matches the finite-difference solver, where the
    # source wavelet carries the same delay.
    traces = np.apply_along_axis(lambda s: np.convolve(s, w)[:nt], 2, series)

    depth_traces = np.zeros(grid.shape)
    if map_to_depth:
        for ix in range(nx):
            for iy in range(ny):
                depth_traces[ix, iy] = np.interp(twt[ix, iy], times, traces[ix, iy])

    return ConvolutionPreview(time_traces=traces, times=times,
                              depth_traces=depth_traces, grid=grid)
