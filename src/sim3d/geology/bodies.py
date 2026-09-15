"""Hand-drawn geobodies painted into a built geological model.

A template gives a layered earth; this gives the thing an interpreter
actually wants to test - *this* channel, here, with this width, cutting
this reservoir.  The body is drawn in map view as a centreline, given a
width and a vertical interval, and its facies replaces whatever the
template put there.

The point is that it reaches the flow simulation.  A body that only
repainted the display would be a drawing; changing porosity, permeability,
net-to-gross and the reservoir mask means the water goes where the channel
goes, the 4D anomaly follows it, and the seismic sees the difference.  So
the edit lands on the property cube the simulator reads, before anything
downstream of geology has run.

Geometry is deliberately simple and explicit: distance to a polyline in
map view, a depth window, and a hard boundary.  There is no smoothing and
no taper, because a channel margin that fades over three cells would make
the flow barrier it represents ambiguous, and a mechanistic laboratory is
better served by an edge you can point at.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.errors import ConfigError
from ..core.units import MILLIDARCY
from .facies import FACIES, Facies, get_facies

#: Body kinds ``geology.bodies`` accepts.
BODY_TYPES = ("channel", "lens")


@dataclass
class GeoBody:
    """One drawn body: a map-view path, a width and a depth interval.

    ``path`` is the centreline for a ``channel`` and the outline for a
    ``lens``; a lens with two points is a line and has no interior, so it
    is refused rather than drawn as nothing.
    """

    name: str = "channel_1"
    type: str = "channel"
    #: ``[[x, y], ...]`` in model coordinates, metres.
    path: list = field(default_factory=list)
    #: Full width of a channel, metres.  Ignored by a lens.
    width: float = 300.0
    top: float = 1200.0
    thickness: float = 40.0
    facies: str = "clean_sandstone"
    #: Explicit properties; ``None`` takes the facies midpoint.
    porosity: float | None = None
    vsh: float | None = None
    ntg: float | None = None
    permeability_md: float | None = None

    def __post_init__(self) -> None:
        if self.type not in BODY_TYPES:
            raise ConfigError(
                f"geobody {self.name!r} has type {self.type!r}; "
                f"valid types are {sorted(BODY_TYPES)}")
        self.path = [[float(x), float(y)] for x, y in self.path]
        minimum = 2 if self.type == "channel" else 3
        if len(self.path) < minimum:
            raise ConfigError(
                f"geobody {self.name!r} is a {self.type} and needs at least "
                f"{minimum} path points, got {len(self.path)}")
        if self.type == "channel" and self.width <= 0.0:
            raise ConfigError(f"geobody {self.name!r} needs a positive width")
        if self.thickness <= 0.0:
            raise ConfigError(f"geobody {self.name!r} needs a positive thickness")

    @property
    def base(self) -> float:
        return self.top + self.thickness

    def rock(self, catalogue: dict[str, Facies] | None = None) -> dict:
        """Resolved properties, facies midpoints filling anything unset."""
        f = get_facies(self.facies, catalogue or FACIES)
        mid = lambda pair: 0.5 * (pair[0] + pair[1])       # noqa: E731
        return {
            "code": f.code,
            "porosity": mid(f.porosity) if self.porosity is None else float(self.porosity),
            "vsh": mid(f.vsh) if self.vsh is None else float(self.vsh),
            "ntg": mid(f.ntg) if self.ntg is None else float(self.ntg),
            "permeability_md": (mid(f.permeability) if self.permeability_md is None
                                else float(self.permeability_md)),
            "is_reservoir": f.is_reservoir,
        }

    def describe(self) -> str:
        length = float(np.sum(np.hypot(*np.diff(np.asarray(self.path), axis=0).T)))
        if self.type == "channel":
            return (f"{self.name}: {self.type}, {length:,.0f} m long, "
                    f"{self.width:,.0f} m wide, {self.top:,.0f}–{self.base:,.0f} m, "
                    f"{self.facies}")
        return (f"{self.name}: {self.type}, {len(self.path)}-point outline, "
                f"{self.top:,.0f}–{self.base:,.0f} m, {self.facies}")


def _distance_to_path(x: np.ndarray, y: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Shortest distance from each (x, y) to a polyline, vectorised.

    Point-to-segment rather than point-to-vertex: a coarse centreline drawn
    with four clicks would otherwise pinch to nothing between them.
    """
    best = np.full(x.shape, np.inf)
    for a, b in zip(path[:-1], path[1:]):
        d = b - a
        length2 = float(d @ d)
        if length2 == 0.0:
            best = np.minimum(best, np.hypot(x - a[0], y - a[1]))
            continue
        t = np.clip(((x - a[0]) * d[0] + (y - a[1]) * d[1]) / length2, 0.0, 1.0)
        best = np.minimum(best, np.hypot(x - (a[0] + t * d[0]),
                                         y - (a[1] + t * d[1])))
    return best


def _inside_polygon(x: np.ndarray, y: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Ray-casting point-in-polygon, vectorised over the map."""
    inside = np.zeros(x.shape, dtype=bool)
    x1, y1 = path[:, 0], path[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for ax, ay, bx, by in zip(x1, y1, x2, y2):
        if ay == by:
            continue
        crosses = ((ay > y) != (by > y))
        cut = ax + (y - ay) * (bx - ax) / (by - ay)
        inside ^= crosses & (x < cut)
    return inside


def body_mask(body: GeoBody, grid) -> np.ndarray:
    """Boolean cube marking the cells the body occupies."""
    xs, ys, zs = grid.axis(0), grid.axis(1), grid.axis(2)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    path = np.asarray(body.path, dtype=float)
    if body.type == "channel":
        plan = _distance_to_path(X, Y, path) <= 0.5 * body.width
    else:
        plan = _inside_polygon(X, Y, path)
    depth = (zs >= body.top) & (zs < body.base)
    return plan[:, :, None] & depth[None, None, :]


def apply_bodies(model, bodies, catalogue: dict[str, Facies] | None = None):
    """Paint every body into ``model``'s property cubes, in order.

    Later bodies overwrite earlier ones where they overlap, which is the
    same rule a drawing program uses and the only one that makes "draw
    another channel across that one" mean what it looks like.

    The model is edited in place and returned: it is built immediately
    before this in the pipeline and has no other reader yet, so copying a
    full property cube per body would cost memory to protect nobody.
    """
    for body in bodies:
        mask = body_mask(body, model.grid)
        if not mask.any():
            raise ConfigError(
                f"geobody {body.name!r} does not intersect the geological "
                f"model: its path or its {body.top:,.0f}–{body.base:,.0f} m "
                f"interval lies outside the grid")
        rock = body.rock(catalogue)
        model.facies_code[mask] = rock["code"]
        model.porosity[mask] = rock["porosity"]
        model.vsh[mask] = rock["vsh"]
        model.ntg[mask] = rock["ntg"]
        model.permeability[mask] = rock["permeability_md"] * MILLIDARCY
        model.reservoir_mask[mask] = rock["is_reservoir"]
    return model


def build_bodies(entries) -> list[GeoBody]:
    """Turn configuration mappings into bodies, naming anything wrong."""
    bodies = []
    seen = set()
    for i, entry in enumerate(entries or []):
        if not isinstance(entry, dict):
            raise ConfigError(
                f"geology.bodies[{i}] must be a mapping of settings, got "
                f"{type(entry).__name__}")
        known = {f for f in GeoBody.__dataclass_fields__}
        unknown = set(entry) - known
        if unknown:
            raise ConfigError(
                f"unknown key(s) {sorted(unknown)} in geology.bodies[{i}]; "
                f"valid keys are {sorted(known)}")
        body = GeoBody(**entry)
        if body.name in seen:
            raise ConfigError(f"duplicate geobody name {body.name!r}")
        seen.add(body.name)
        bodies.append(body)
    return bodies
