r"""Well completions defined by geological unit, not by typed depths.

Requirement section 2.  Production and injection never happen through the
whole wellbore: every well carries an explicit set of open intervals, and
those intervals are chosen by naming geological units.  The software then
computes where the well actually meets each unit.

That indirection is the point.  A completion typed as "1180-1240 m" is
wrong the moment the layer is dipped, folded or faulted, and wrong
differently at every well in the pattern.  A completion recorded as "open
in the reservoir and the lower sand" stays correct through all of it,
because the depths are recomputed from the model each time.

Because the initial wells are vertical, the intersection is the column of
the layer-index volume beneath the wellhead, which already carries the
local structural depth: dip, fold relief and fault throw are all baked into
that column by :func:`sim3d.geology.builder.build_geology`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.errors import ConfigError, ValidationError
from ..core.units import MILLIDARCY


@dataclass(frozen=True)
class LayerIntersection:
    """Where a vertical well meets one geological unit."""

    index: int
    name: str
    top: float            #: local top depth at this well, m
    base: float           #: local base depth at this well, m
    is_reservoir: bool
    porosity: float
    permeability: float   #: thickness-averaged, m^2
    ntg: float

    @property
    def thickness(self) -> float:
        """Gross thickness at the well, m."""
        return self.base - self.top

    @property
    def net_thickness(self) -> float:
        """Net reservoir thickness at the well, m."""
        return self.thickness * self.ntg

    @property
    def permeability_md(self) -> float:
        return self.permeability / MILLIDARCY

    def describe(self) -> str:
        flag = "reservoir" if self.is_reservoir else "non-reservoir"
        return (f"{self.name:22s} {self.top:7.1f} - {self.base:7.1f} m "
                f"({self.thickness:5.1f} m gross, {self.net_thickness:5.1f} m net) "
                f"phi {self.porosity:.3f}  k {self.permeability_md:8.2f} mD  {flag}")


@dataclass
class Completion:
    """One open (or shut) interval, named by its geological unit.

    ``top`` and ``base`` are optional manual adjustments *within* that unit,
    for the case where only part of a thick layer is perforated.  They are
    validated against the unit's own local depths, so a completion can never
    silently drift outside the rock it claims to be in.
    """

    layer: str
    open: bool = True
    top: float | None = None
    base: float | None = None

    def resolve(self, intersection: LayerIntersection) -> tuple[float, float]:
        """The actual open interval, clipped to and validated against the unit."""
        top = intersection.top if self.top is None else float(self.top)
        base = intersection.base if self.base is None else float(self.base)
        if base <= top:
            raise ConfigError(
                f"completion in {self.layer!r} has base {base:g} m at or above "
                f"top {top:g} m")
        tolerance = 1e-6
        if top < intersection.top - tolerance or base > intersection.base + tolerance:
            raise ValidationError(
                f"completion {top:g}-{base:g} m in {self.layer!r} falls outside "
                f"that unit at this well, which spans "
                f"{intersection.top:g}-{intersection.base:g} m here. Layer depths "
                f"vary with structure, so an interval typed for one well is not "
                f"valid at another."
            )
        return top, base


def layer_intersections(well, geology, min_thickness: float = 0.0
                        ) -> list[LayerIntersection]:
    """Every geological unit a vertical well passes through, top to bottom.

    Units thinner than ``min_thickness`` at this well are omitted, which is
    how a pinchout stops offering a completion that would be a single cell
    thick.
    """
    grid = geology.grid
    if not grid.contains((well.x, well.y, 0.5 * sum(grid.bounds[2]))):
        raise ValidationError(
            f"well {well.name} at ({well.x:g}, {well.y:g}) m is outside the "
            f"geological model {grid.bounds[0]} x {grid.bounds[1]}")
    ix = int(np.argmin(np.abs(grid.axis(0) - well.x)))
    iy = int(np.argmin(np.abs(grid.axis(1) - well.y)))

    column = geology.layer_index[ix, iy, :]
    z = grid.axis(2)
    out: list[LayerIntersection] = []
    for index in np.unique(column):
        cells = np.flatnonzero(column == index)
        # Cell centres to cell boundaries: the unit extends half a cell past
        # its outermost node in each direction.
        top = float(z[cells[0]] - 0.5 * grid.dz)
        base = float(z[cells[-1]] + 0.5 * grid.dz)
        if base - top < min_thickness:
            continue
        layer = geology.layers[int(index)]
        out.append(LayerIntersection(
            index=int(index), name=layer.name, top=top, base=base,
            is_reservoir=bool(geology.reservoir_mask[ix, iy, cells].any()),
            porosity=float(geology.porosity[ix, iy, cells].mean()),
            permeability=float(geology.permeability[ix, iy, cells].mean()),
            ntg=float(geology.ntg[ix, iy, cells].mean()),
        ))
    return sorted(out, key=lambda item: item.top)


def default_completions(well, geology, reservoir_only: bool = True) -> list[Completion]:
    """Open every reservoir unit the well passes through.

    A sensible starting point that the user then edits, rather than a blank
    form: the model already knows which units are reservoir.
    """
    return [Completion(layer=item.name, open=True)
            for item in layer_intersections(well, geology)
            if item.is_reservoir or not reservoir_only]


def resolve_completions(well, geology, completions: list[Completion]
                        ) -> list[tuple[float, float, LayerIntersection]]:
    """Turn named completions into ``(top, base, unit)`` intervals in metres.

    Raises if a completion names a unit the well does not reach - which is
    the common consequence of moving a well across a fault, and exactly the
    case that must not pass silently.
    """
    found = {item.name: item for item in layer_intersections(well, geology)}
    intervals = []
    for completion in completions:
        if not completion.open:
            continue
        item = found.get(completion.layer)
        if item is None:
            raise ValidationError(
                f"well {well.name} is completed in {completion.layer!r}, but does "
                f"not intersect that unit at ({well.x:g}, {well.y:g}) m. It passes "
                f"through: {', '.join(found)}."
            )
        top, base = completion.resolve(item)
        intervals.append((top, base, item))
    return intervals


def completion_mask(well, geology, completions: list[Completion]) -> np.ndarray:
    """Boolean volume marking the cells a well is open to.

    One column of cells beneath the wellhead, limited to the resolved
    intervals - the connection list a flow model needs.
    """
    grid = geology.grid
    mask = np.zeros(grid.shape, dtype=bool)
    ix = int(np.argmin(np.abs(grid.axis(0) - well.x)))
    iy = int(np.argmin(np.abs(grid.axis(1) - well.y)))
    z = grid.axis(2)
    for top, base, _ in resolve_completions(well, geology, completions):
        # ``top`` and ``base`` are already cell boundaries, so node centres
        # inside them are exactly the unit's cells. Adding a half-cell
        # tolerance here would reach one cell into the units either side.
        mask[ix, iy, (z >= top) & (z <= base)] = True
    return mask


def completion_summary(well, geology, completions: list[Completion]) -> str:
    """Human-readable statement of where a well is open, with local depths."""
    lines = [f"{well.name} ({well.role}) at ({well.x:,.0f}, {well.y:,.0f}) m"]
    open_units = {c.layer for c in completions if c.open}
    for item in layer_intersections(well, geology):
        marker = "OPEN " if item.name in open_units else "shut "
        lines.append(f"  {marker}{item.describe()}")
    intervals = resolve_completions(well, geology, completions)
    net = sum((base - top) * item.ntg for top, base, item in intervals)
    lines.append(f"  total net completed thickness: {net:.1f} m")
    return "\n".join(lines)
