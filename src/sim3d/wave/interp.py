"""Trilinear scatter/gather between physical coordinates and the grid.

Sources and receivers sit at arbitrary metre coordinates, not on grid
nodes.  Snapping them to the nearest node introduces a position error of
up to half a cell, which at reservoir scale is a real traveltime error and
a real 4D repeatability error, so both injection and recording use
trilinear weights over the eight surrounding nodes.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import ConfigError
from ..core.grid import Grid3D


class PointSet:
    """Pre-computed trilinear stencils for a set of points on one grid.

    Building the weights once and reusing them every time step keeps the
    inner loop free of coordinate arithmetic.
    """

    def __init__(self, grid: Grid3D, points: np.ndarray, name: str = "points"):
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ConfigError(f"{name} must have shape (n, 3), got {pts.shape}")
        outside = [tuple(p) for p in pts if not grid.contains(p)]
        if outside:
            raise ConfigError(
                f"{len(outside)} {name} lie outside the propagation domain "
                f"{grid.bounds}; first offender {outside[0]}. Enlarge the "
                f"propagation domain or move the geometry - sim3d will not "
                f"clip the acquisition to fit."
            )
        self.grid = grid
        self.points = pts
        self.name = name

        frac = np.stack([np.asarray(grid.to_index(p)) for p in pts])
        base = np.floor(frac).astype(int)
        # Keep the +1 neighbour inside the grid for points exactly on the far face.
        base = np.minimum(base, np.array(grid.shape) - 2)
        base = np.maximum(base, 0)
        self._base = base
        self._t = frac - base  # in [0, 1]

        # Eight corner offsets and their weights, shape (n, 8).
        offsets = np.array(
            [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=int
        )
        t = self._t
        w = np.ones((len(pts), 8), dtype=float)
        for c, (oi, oj, ok) in enumerate(offsets):
            wx = t[:, 0] if oi else 1.0 - t[:, 0]
            wy = t[:, 1] if oj else 1.0 - t[:, 1]
            wz = t[:, 2] if ok else 1.0 - t[:, 2]
            w[:, c] = wx * wy * wz
        self._offsets = offsets
        self.weights = w

        idx = base[:, None, :] + offsets[None, :, :]  # (n, 8, 3)
        self.ix = idx[..., 0]
        self.iy = idx[..., 1]
        self.iz = idx[..., 2]

    def __len__(self) -> int:
        return len(self.points)

    def gather(self, field: np.ndarray) -> np.ndarray:
        """Interpolate ``field`` at every point; returns shape ``(n,)``."""
        return np.einsum("pc,pc->p", field[self.ix, self.iy, self.iz], self.weights)

    def scatter(self, field: np.ndarray, values: np.ndarray, scale: np.ndarray | float = 1.0) -> None:
        """Add ``values`` into ``field`` at every point, spread trilinearly.

        ``scale`` may be a full grid array (e.g. ``dt * kappa``) that is
        sampled at the same eight nodes, which is what lets a source term be
        injected with the local medium properties applied.
        """
        vals = np.asarray(values, dtype=float).reshape(-1, 1) * self.weights
        if np.ndim(scale) == 3:
            vals = vals * scale[self.ix, self.iy, self.iz]
        else:
            vals = vals * scale
        np.add.at(field, (self.ix, self.iy, self.iz), vals)
