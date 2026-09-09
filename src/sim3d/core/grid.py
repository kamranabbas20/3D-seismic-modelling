"""Regular Cartesian grids and the three-domain scale hierarchy.

Spec section 6 requires that three domains stay architecturally distinct:

``geology``
    the full synthetic geological / reservoir model;
``propagation``
    the sub-volume actually stepped by the finite-difference solver;
``target``
    the window in which the seismic response is analysed.

They are *not* assumed identical.  :class:`DomainSet` holds all three and
checks that each is contained in the previous one, so a target window can
never silently sit inside an absorbing boundary or outside the modelled
volume.

Axis convention: ``x``, ``y`` are map coordinates in metres and ``z``
increases downwards from ``z=0`` at the top of the model.  Arrays are
indexed ``[ix, iy, iz]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import ConfigError


@dataclass(frozen=True)
class Grid3D:
    """A regular 3D Cartesian grid, node-centred at ``origin``.

    Parameters
    ----------
    origin:
        ``(xmin, ymin, zmin)`` in metres.
    spacing:
        ``(dx, dy, dz)`` in metres.
    shape:
        ``(nx, ny, nz)`` node counts.
    """

    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        if any(n < 2 for n in self.shape):
            raise ConfigError(f"each grid axis needs >= 2 nodes, got {self.shape}")
        if any(d <= 0 for d in self.spacing):
            raise ConfigError(f"grid spacing must be positive, got {self.spacing}")

    # -- derived geometry ------------------------------------------------
    @property
    def nx(self) -> int:
        return self.shape[0]

    @property
    def ny(self) -> int:
        return self.shape[1]

    @property
    def nz(self) -> int:
        return self.shape[2]

    @property
    def dx(self) -> float:
        return self.spacing[0]

    @property
    def dy(self) -> float:
        return self.spacing[1]

    @property
    def dz(self) -> float:
        return self.spacing[2]

    @property
    def n_cells(self) -> int:
        """Total number of grid nodes."""
        return int(self.nx) * int(self.ny) * int(self.nz)

    @property
    def extent(self) -> tuple[float, float, float]:
        """Physical size ``(lx, ly, lz)`` in metres, node-to-node."""
        return tuple(float(d) * (n - 1) for d, n in zip(self.spacing, self.shape))

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        """``((xmin, xmax), (ymin, ymax), (zmin, zmax))`` in metres."""
        return tuple(
            (float(o), float(o) + float(d) * (n - 1))
            for o, d, n in zip(self.origin, self.spacing, self.shape)
        )

    def axis(self, which: int | str) -> np.ndarray:
        """Coordinate vector along axis ``which`` (``0/'x'``, ``1/'y'``, ``2/'z'``)."""
        idx = {"x": 0, "y": 1, "z": 2}.get(which, which) if isinstance(which, str) else which
        o, d, n = self.origin[idx], self.spacing[idx], self.shape[idx]
        return o + d * np.arange(n, dtype=float)

    @property
    def coords(self) -> dict[str, np.ndarray]:
        """xarray-style coordinate mapping for this grid."""
        return {"x": self.axis(0), "y": self.axis(1), "z": self.axis(2)}

    # -- point queries ---------------------------------------------------
    def contains(self, point, tol: float = 1e-9) -> bool:
        """True if ``point`` (x, y, z in metres) lies inside the grid bounds."""
        return all(
            lo - tol <= float(c) <= hi + tol
            for c, (lo, hi) in zip(point, self.bounds)
        )

    def to_index(self, point) -> tuple[float, float, float]:
        """Fractional grid index of a physical point (no rounding, no clipping)."""
        return tuple(
            (float(c) - o) / d for c, o, d in zip(point, self.origin, self.spacing)
        )

    def nearest_index(self, point) -> tuple[int, int, int]:
        """Nearest node index to a physical point.

        Raises :class:`ConfigError` when the point is outside the grid: a
        receiver quietly snapped to the model edge would corrupt the
        experiment rather than the geometry.
        """
        if not self.contains(point):
            raise ConfigError(
                f"point {tuple(float(c) for c in point)} is outside grid bounds {self.bounds}"
            )
        return tuple(int(round(v)) for v in self.to_index(point))

    # -- construction helpers -------------------------------------------
    @classmethod
    def from_bounds(cls, bounds, spacing) -> "Grid3D":
        """Build the smallest grid covering ``bounds`` at the given ``spacing``.

        ``bounds`` is ``((xmin, xmax), (ymin, ymax), (zmin, zmax))``.  The
        upper bound is extended, never truncated, so the requested volume is
        always fully contained.
        """
        origin, shape = [], []
        for (lo, hi), d in zip(bounds, spacing):
            if hi <= lo:
                raise ConfigError(f"bounds {(lo, hi)} are not increasing")
            origin.append(float(lo))
            shape.append(int(np.ceil((hi - lo) / d)) + 1)
        return cls(tuple(origin), tuple(float(d) for d in spacing), tuple(shape))

    def resampled(self, spacing) -> "Grid3D":
        """Same physical extent, different spacing (extent may grow slightly)."""
        return Grid3D.from_bounds(self.bounds, spacing)

    def padded(self, pad_cells) -> "Grid3D":
        """Grow the grid by ``pad_cells`` nodes on both ends of every axis.

        Used to wrap a propagation domain in absorbing-boundary padding
        without moving the physical target.
        """
        pad = (pad_cells,) * 3 if np.isscalar(pad_cells) else tuple(pad_cells)
        origin = tuple(o - p * d for o, d, p in zip(self.origin, self.spacing, pad))
        shape = tuple(n + 2 * p for n, p in zip(self.shape, pad))
        return Grid3D(origin, self.spacing, shape)

    def describe(self) -> str:
        """One-line human summary, metres."""
        (x0, x1), (y0, y1), (z0, z1) = self.bounds
        return (
            f"{self.nx}x{self.ny}x{self.nz} nodes, "
            f"d=({self.dx:g}, {self.dy:g}, {self.dz:g}) m, "
            f"x[{x0:g},{x1:g}] y[{y0:g},{y1:g}] z[{z0:g},{z1:g}] m, "
            f"{self.n_cells:,} cells"
        )


@dataclass(frozen=True)
class DomainSet:
    """The geology / propagation / target hierarchy of spec section 6.

    ``propagation`` must lie inside ``geology`` (you cannot propagate through
    rock you never built) and ``target`` inside ``propagation`` (you cannot
    analyse an image where no wavefield was computed).
    """

    geology: Grid3D
    propagation: Grid3D
    target: Grid3D

    def __post_init__(self) -> None:
        _require_contained(self.propagation, self.geology, "propagation", "geology")
        _require_contained(self.target, self.propagation, "target", "propagation")

    def summary(self) -> str:
        """Multi-line report of all three domains and their cell counts."""
        lines = ["Domain hierarchy (spec section 6):"]
        for name in ("geology", "propagation", "target"):
            lines.append(f"  {name:12s} {getattr(self, name).describe()}")
        ratio = self.propagation.n_cells / max(self.geology.n_cells, 1)
        lines.append(f"  propagation domain is {100 * ratio:.1f}% of the geological volume")
        return "\n".join(lines)


def _require_contained(inner: Grid3D, outer: Grid3D, inner_name: str, outer_name: str) -> None:
    tol = 1e-6
    for axis, ((ilo, ihi), (olo, ohi)) in enumerate(zip(inner.bounds, outer.bounds)):
        if ilo < olo - tol or ihi > ohi + tol:
            ax = "xyz"[axis]
            raise ConfigError(
                f"{inner_name} domain {ax}[{ilo:g},{ihi:g}] m is not contained in "
                f"{outer_name} domain {ax}[{olo:g},{ohi:g}] m"
            )
