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
class PickedThickness(Surface):
    """A *thickness* interpolated between picked points along one axis.

    What a drawn horizon becomes.  The user clicks where they want the base
    of a unit to sit on a section; the depth they clicked is turned into a
    thickness relative to the unit's own top and stored here, clamped at
    zero.  Storing the thickness rather than the depth is what makes a drawn
    stratigraphy safe: thicknesses are non-negative, so the horizons that
    accumulate from them can touch but never cross, however the picks fall.

    ``points`` are ``(position, thickness)`` pairs in metres, ``position``
    measured along ``axis`` (0 for x, 1 for y).  Between picks the thickness
    is linear; beyond the outermost picks it is held flat, so a section
    drawn across part of the model does not imply anything about the rest of
    it beyond continuing as it ended.

    The thickness varies only along ``axis``.  That is the honest reading of
    a pick made on one section: nothing was said about the other direction,
    so nothing is invented for it.
    """

    points: list[tuple[float, float]] = field(default_factory=list)
    axis: int = 0

    def depth(self, x, y):
        if not self.points:
            raise ConfigError(
                "a picked thickness needs at least one point; draw one on the "
                "section or give the unit a constant thickness instead")
        if self.axis not in (0, 1):
            raise ConfigError(f"axis must be 0 (x) or 1 (y), got {self.axis}")
        ordered = sorted((float(a), float(b)) for a, b in self.points)
        positions = np.array([a for a, _ in ordered], dtype=float)
        # Clamped here rather than trusted: a pick above the unit's own top
        # is a thickness of zero - a pinchout - not a negative thickness.
        values = np.clip([b for _, b in ordered], 0.0, None)
        along = x if self.axis == 0 else y
        return np.interp(np.asarray(along, dtype=float), positions, values)


@dataclass
class SectionThickness(Surface):
    """A *thickness* defined on several sections, interpolated between them.

    The general form of :class:`PickedThickness`.  Each section is a line
    across the model at a fixed position on the other axis, carrying its own
    knee points:

    ``{"at": 500.0, "points": [(0.0, 90.0), (2000.0, 30.0)]}``

    Thickness is interpolated in two separable steps - along each section
    between its own knee points, then across the model between the sections
    - so an edit made on one section stays put and only its neighbourhood
    moves.  Outside the outermost sections the nearest one is held flat, for
    the same reason the knee points are: a section says nothing about ground
    beyond the ones that were drawn, so nothing is invented for it.

    Every thickness stays non-negative through both steps, which is what
    keeps the horizons that accumulate from it from ever crossing.
    """

    sections: list = field(default_factory=list)
    axis: int = 0

    def _profiles(self, along):
        ordered = sorted(self.sections, key=lambda s: float(s["at"]))
        positions = np.array([float(s["at"]) for s in ordered], dtype=float)
        if len(np.unique(positions)) != len(positions):
            raise ConfigError(
                "two sections sit at the same position; move one or merge them")
        stack = []
        for section in ordered:
            points = sorted((float(a), float(b))
                            for a, b in (section.get("points") or []))
            if not points:
                raise ConfigError(
                    f"the section at {section.get('at')} has no knee points")
            stack.append(np.interp(
                along, [a for a, _ in points],
                np.clip([b for _, b in points], 0.0, None)))
        return positions, np.stack(stack, axis=-1)

    def depth(self, x, y):
        if not self.sections:
            raise ConfigError(
                "a section thickness needs at least one section; draw one or "
                "give the unit a constant thickness instead")
        if self.axis not in (0, 1):
            raise ConfigError(f"axis must be 0 (x) or 1 (y), got {self.axis}")
        along = np.asarray(x if self.axis == 0 else y, dtype=float)
        across = np.asarray(y if self.axis == 0 else x, dtype=float)
        positions, stack = self._profiles(along)
        if positions.size == 1:
            return stack[..., 0]

        upper = np.clip(np.searchsorted(positions, across), 1, positions.size - 1)
        lower = upper - 1
        span = positions[upper] - positions[lower]
        # Clipped, so beyond the outermost sections the nearest one is held
        # flat rather than extrapolated into ground nobody drew.
        weight = np.clip((across - positions[lower]) / span, 0.0, 1.0)
        first = np.take_along_axis(stack, lower[..., None], axis=-1)[..., 0]
        second = np.take_along_axis(stack, upper[..., None], axis=-1)[..., 0]
        return first * (1.0 - weight) + second * weight


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
