"""Parametric horizon surfaces (spec sections 18, 21).

A surface maps map coordinates to a depth, ``z = f(x, y)``, with ``z``
increasing downwards.  Surfaces compose additively, so an anticline on a
regional dip is ``Dipping(...) + Anticline(...)`` and needs no special case.

The primitives here cover the structures the spec asks for in the first
version: horizontal and dipping layering, anticlines, synclines, and -
through :class:`Composite` and per-layer thickness - wedges, lenses and
pinchouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.errors import ConfigError
from ..core.geometry import bearing_vector, to_principal


class Surface:
    """Base class: something that returns a depth for each map position."""

    def depth(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Depth in metres on the ``(x, y)`` mesh, positive downwards."""
        raise NotImplementedError

    def on_grid(self, grid) -> np.ndarray:
        """Evaluate on a :class:`~sim3d.core.grid.Grid3D`, returning ``(nx, ny)``."""
        x, y = np.meshgrid(grid.axis(0), grid.axis(1), indexing="ij")
        return self.depth(x, y)

    def __add__(self, other: "Surface") -> "Composite":
        return Composite([self, other])

    def shifted(self, dz: float) -> "Composite":
        """The same surface moved down by ``dz`` metres."""
        return Composite([self, Flat(dz)])


@dataclass
class Flat(Surface):
    """A horizontal surface at constant depth."""

    z0: float

    def depth(self, x, y):
        return np.full(np.broadcast(x, y).shape, float(self.z0))


@dataclass
class Dipping(Surface):
    """A planar surface dipping at ``dip`` degrees towards ``azimuth``.

    ``azimuth`` is measured in degrees clockwise from ``+y`` (map north),
    the geological convention.  ``z0`` is the depth at ``origin``.
    """

    z0: float
    dip: float = 2.0
    azimuth: float = 90.0
    origin: tuple[float, float] = (0.0, 0.0)

    def depth(self, x, y):
        if not -90.0 < self.dip < 90.0:
            raise ConfigError(f"dip must be within (-90, 90) degrees, got {self.dip}")
        slope = np.tan(np.radians(self.dip))
        ux, uy = bearing_vector(self.azimuth)  # down-dip direction
        return self.z0 + slope * ((x - self.origin[0]) * ux + (y - self.origin[1]) * uy)


@dataclass
class Anticline(Surface):
    """A Gaussian dome: the surface shallows by ``amplitude`` over the crest.

    ``radius`` is a ``(major, minor)`` pair of Gaussian half-widths in
    metres, with ``azimuth`` the bearing of the major axis clockwise from
    ``+y``, so an elongated fold is expressed directly.
    """

    z0: float
    amplitude: float = 80.0
    centre: tuple[float, float] = (1500.0, 1500.0)
    radius: tuple[float, float] = (900.0, 1400.0)
    azimuth: float = 0.0

    def depth(self, x, y):
        rx, ry = self.radius
        if rx <= 0 or ry <= 0:
            raise ConfigError(f"fold radii must be positive, got {self.radius}")
        u, v = to_principal(x - self.centre[0], y - self.centre[1], self.azimuth)
        return self.z0 - self.amplitude * np.exp(-((u / rx) ** 2 + (v / ry) ** 2))


@dataclass
class Syncline(Anticline):
    """A Gaussian basin: the mirror image of :class:`Anticline`."""

    def depth(self, x, y):
        crest = Anticline.depth(self, x, y)
        return 2.0 * self.z0 - crest


@dataclass
class Wedge(Surface):
    """A *thickness* that ramps linearly to zero along a bearing.

    Not a horizon but a term to add to one: ``top + Wedge(...)`` is the base
    of a unit that thins out in the ``azimuth`` direction and is gone beyond
    ``end``.  Because the thickness reaches zero and never goes negative, the
    base touches the top rather than crossing it, which is exactly what a
    pinchout is and what :func:`~sim3d.geology.builder.build_geology`
    requires of ordered horizons.

    ``start`` and ``end`` are distances in metres along the bearing from
    ``origin``: full thickness at and before ``start``, zero at and beyond
    ``end``.
    """

    thickness: float
    azimuth: float = 90.0
    start: float = 0.0
    end: float = 1000.0
    origin: tuple[float, float] = (0.0, 0.0)

    def depth(self, x, y):
        if self.thickness < 0:
            raise ConfigError(f"a wedge thickness cannot be negative, got "
                              f"{self.thickness}")
        if self.end <= self.start:
            raise ConfigError(
                f"a wedge must thin over a positive distance, got start "
                f"{self.start} and end {self.end}")
        ux, uy = bearing_vector(self.azimuth)
        along = (x - self.origin[0]) * ux + (y - self.origin[1]) * uy
        taper = np.clip((along - self.start) / (self.end - self.start), 0.0, 1.0)
        return self.thickness * (1.0 - taper)


@dataclass
class Lens(Surface):
    """A *thickness* that tapers to zero away from a centre.

    The radial counterpart of :class:`Wedge`: thickest at ``centre``, gone at
    the edge of the ellipse set by ``radius`` and ``azimuth``.  Adding it to
    a horizon gives a lens or a channel body that pinches out all round
    rather than in one direction.
    """

    thickness: float
    centre: tuple[float, float] = (1500.0, 1500.0)
    radius: tuple[float, float] = (900.0, 500.0)
    azimuth: float = 0.0
    #: Fraction of the radius over which the thickness falls from full to
    #: zero.  1.0 is a smooth taper from the centre; 0.2 is a flat-topped
    #: body with steep edges.
    taper: float = 1.0

    def depth(self, x, y):
        rx, ry = self.radius
        if rx <= 0 or ry <= 0:
            raise ConfigError(f"lens radii must be positive, got {self.radius}")
        if not 0.0 < self.taper <= 1.0:
            raise ConfigError(f"lens taper must lie in (0, 1], got {self.taper}")
        u, v = to_principal(x - self.centre[0], y - self.centre[1], self.azimuth)
        r = np.sqrt((u / rx) ** 2 + (v / ry) ** 2)
        edge = np.clip((1.0 - r) / self.taper, 0.0, 1.0)
        return self.thickness * edge


@dataclass
class Truncated(Surface):
    """A surface clamped so it can touch the one above but never cross it.

    The explicit unconformity the builder asks for when a structure would
    otherwise drive one horizon through another.  Where ``surface`` would go
    shallower than ``limit`` it is held at ``limit``, leaving zero thickness
    there instead of a crossing horizon.
    """

    surface: Surface
    limit: Surface

    def depth(self, x, y):
        return np.maximum(self.surface.depth(x, y), self.limit.depth(x, y))


@dataclass
class Composite(Surface):
    """The sum of several surfaces, taking the first one's ``z0`` as the base.

    Adding surfaces adds their depths, so combining a dipping surface with
    an anticline of the same ``z0`` would double-count the datum.
    :meth:`Surface.__add__` therefore subtracts each additional surface's
    value at its own crest, leaving each term contributing only its relief.
    """

    parts: list[Surface] = field(default_factory=list)

    def depth(self, x, y):
        if not self.parts:
            raise ConfigError("a Composite surface needs at least one part")
        total = self.parts[0].depth(x, y)
        for part in self.parts[1:]:
            total = total + part.depth(x, y)
        return total

    def __add__(self, other: Surface) -> "Composite":
        return Composite([*self.parts, other])


@dataclass
class Relief(Surface):
    """A surface's departure from its own datum, for use as an additive term.

    ``Flat(1200) + Relief(Anticline(0, amplitude=80, ...))`` gives a flat
    horizon at 1200 m with 80 m of fold relief, without the fold's own
    datum entering twice.
    """

    surface: Surface
    datum: float = 0.0

    def depth(self, x, y):
        return self.surface.depth(x, y) - self.datum
