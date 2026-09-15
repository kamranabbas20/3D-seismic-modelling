"""Source terms injected into the wave equation (spec section 64).

In full-wave mode the wavelet is a term in the pressure equation, not a
convolution kernel.  A pressure (explosive) point source adds

.. math:: p \\mathrel{+}= \\Delta t\\,\\kappa(\\mathbf x)\\,
          \\frac{s(t)}{\\Delta x\\,\\Delta y\\,\\Delta z}\\,w(\\mathbf x)

where ``w`` are the trilinear weights of the source position and
:math:`\\kappa = \\rho V_p^2`.  Spreading by the cell volume keeps the
radiated amplitude independent of the grid spacing, so a preview run and a
fine run are directly comparable.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import ConfigError
from ..core.grid import Grid3D
from .interp import PointSet


class SourceTerm:
    """Base class: something that adds to the pressure field each time step."""

    def inject(self, p: np.ndarray, step: int, dt_kappa_over_cell: np.ndarray) -> None:
        raise NotImplementedError

    @property
    def n_steps(self) -> int:
        raise NotImplementedError


class PointSource(SourceTerm):
    """A single explosive source at an arbitrary position."""

    def __init__(self, grid: Grid3D, position, wavelet: np.ndarray):
        self.points = PointSet(grid, np.asarray(position, dtype=float).reshape(1, 3), "sources")
        self.wavelet = np.asarray(wavelet, dtype=float).ravel()
        self.position = tuple(float(c) for c in np.asarray(position).ravel())

    @property
    def n_steps(self) -> int:
        return int(self.wavelet.size)

    def inject(self, p: np.ndarray, step: int, dt_kappa_over_cell: np.ndarray) -> None:
        if 0 <= step < self.wavelet.size:
            self.points.scatter(p, [self.wavelet[step]], scale=dt_kappa_over_cell)


class MultiPointSource(SourceTerm):
    """Many simultaneous point sources, each with its own time series.

    Used to back-propagate a recorded shot gather during RTM: every
    receiver becomes a source driven by its own time-reversed trace.
    """

    def __init__(self, grid: Grid3D, positions, traces: np.ndarray):
        self.points = PointSet(grid, positions, "back-propagated receivers")
        self.traces = np.asarray(traces, dtype=float)
        if self.traces.ndim != 2 or self.traces.shape[0] != len(self.points):
            raise ConfigError(
                f"traces must have shape (n_points, nt); got {self.traces.shape} "
                f"for {len(self.points)} points"
            )

    @property
    def n_steps(self) -> int:
        return int(self.traces.shape[1])

    def inject(self, p: np.ndarray, step: int, dt_kappa_over_cell: np.ndarray) -> None:
        if 0 <= step < self.traces.shape[1]:
            self.points.scatter(p, self.traces[:, step], scale=dt_kappa_over_cell)


class NoSource(SourceTerm):
    """Placeholder for runs driven purely by an initial condition."""

    @property
    def n_steps(self) -> int:
        return 0

    def inject(self, p: np.ndarray, step: int, dt_kappa_over_cell: np.ndarray) -> None:
        return
