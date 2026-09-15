"""Sparse vertical synthetic traces: the second, cheap seismic mode.

The scientific path in this package is wave propagation, acquisition and
migration.  This module is the other end of the scale: instead of a cube
that came out of an imaging operator, it produces **K vertical traces** at
chosen locations, each built by the one-dimensional convolutional chain.

Why it exists alongside :mod:`sim3d.processing.preview`, which already
convolves the whole cube: the full preview costs ``nx * ny`` columns and
answers a map-shaped question, while most of the 4D questions an engineer
actually asks are asked *at a well* - does the flood front show up on the
monitor survey at P1, and by how much.  K columns is a few thousand times
less arithmetic than the cube and hundreds of thousands of times less than
the RTM, which is what makes the 4D loop interactive.

What a trace here contains: vertical normal-incidence reflectivity,
bandwidth, and tuning between closely spaced interfaces.

What it does not contain, and cannot: lateral wave propagation,
diffraction, refraction, transmission loss, geometric spreading,
illumination, offset and azimuth, migration, and every finite-frequency
effect that is not one-dimensional.  A difference measured here is the
difference the rock physics put into the vertical impedance profile, not
the difference a survey would record.  That is a genuinely useful thing to
know - it is the 4D signal before the acquisition and the operator get to
it - but it is not a substitute for the migrated result, and nothing this
module returns may be presented as an image.  Everything it produces
carries :data:`SPARSE_LABEL`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.errors import ConfigError
from ..core.grid import Grid3D
from ..wave.acoustic import AcousticModel
from .preview import synthetic_columns, time_from_depth

SPARSE_LABEL = ("Sparse 1D Synthetic Traces - Not Full 3D Wave Modelling "
                "or Migration")

#: How the K trace locations are chosen.
LAYOUTS = ("wells", "points", "grid")


@dataclass(frozen=True)
class TraceLocation:
    """One vertical trace position, named so a plot can be read."""

    name: str
    x: float
    y: float
    ix: int
    iy: int

    def describe(self) -> str:
        return f"{self.name} at ({self.x:,.0f}, {self.y:,.0f}) m"


@dataclass
class SparseSynthetic:
    """K vertical synthetic traces, labelled for what they are."""

    locations: tuple[TraceLocation, ...]
    traces: np.ndarray            #: ``(K, nt)`` in the time domain
    times: np.ndarray             #: time axis, seconds
    depth_traces: np.ndarray      #: ``(K, nz)`` mapped back onto the depth grid
    depths: np.ndarray            #: depth axis, metres
    twt: np.ndarray               #: ``(K, nz)`` two-way time at each depth node
    label: str = SPARSE_LABEL

    @property
    def n_traces(self) -> int:
        return len(self.locations)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(location.name for location in self.locations)

    def index(self, name: str) -> int:
        for i, location in enumerate(self.locations):
            if location.name == name:
                return i
        raise KeyError(f"no trace named {name!r}; have {list(self.names)}")

    def trace(self, name: str) -> np.ndarray:
        """The time-domain trace at one named location."""
        return self.traces[self.index(name)]

    def depth_trace(self, name: str) -> np.ndarray:
        """The same trace resampled onto the depth axis."""
        return self.depth_traces[self.index(name)]

    def describe(self) -> str:
        dt = (self.times[1] - self.times[0]) * 1e3 if self.times.size > 1 else 0.0
        return (f"{self.label}\n"
                f"  {self.n_traces} traces, {self.times.size} time samples "
                f"at {dt:.2f} ms\n"
                + "\n".join(f"    {loc.describe()}" for loc in self.locations))


# --------------------------------------------------------------- locations
def _snap(name: str, x: float, y: float, grid: Grid3D) -> TraceLocation:
    """Nearest column to ``(x, y)``; raises if the point is off the grid.

    Snapping silently is the one degradation worth allowing here, because a
    trace has to live on a column of the model, and the offset is at most
    half a cell.  Being *outside* the model is a different matter and is an
    error, per the containment rule the domains already enforce.
    """
    (_, _), (_, _), (z0, _) = grid.bounds
    try:
        ix, iy, _ = grid.nearest_index((x, y, z0))
    except Exception as exc:
        raise ConfigError(
            f"trace {name!r} at ({x:,.0f}, {y:,.0f}) m is outside the model "
            f"being sampled: {exc}") from None
    return TraceLocation(name=name, x=float(x), y=float(y), ix=int(ix), iy=int(iy))


def locations_from_wells(wells, grid: Grid3D) -> tuple[TraceLocation, ...]:
    """One trace per well, named after the well.

    This is the default because it is the layout that answers the question
    the wells were placed to ask.
    """
    found = list(wells) if wells is not None else []
    if not found:
        raise ConfigError(
            "the 'wells' trace layout needs at least one well; place a well "
            "first, or choose layout 'grid' with a count, or 'points' with "
            "explicit coordinates")
    return tuple(_snap(w.name, w.x, w.y, grid) for w in found)


def locations_from_points(points, grid: Grid3D) -> tuple[TraceLocation, ...]:
    """Traces at explicit ``(x, y)`` coordinates, numbered in order given."""
    listed = [tuple(p) for p in (points or ())]
    if not listed:
        raise ConfigError(
            "the 'points' trace layout needs at least one [x, y] coordinate")
    for i, point in enumerate(listed):
        if len(point) != 2:
            raise ConfigError(
                f"trace point {i} is {point!r}; each point is [x, y] in metres "
                f"(the trace is vertical, so it carries no z)")
    return tuple(_snap(f"T{i + 1}", float(x), float(y), grid)
                 for i, (x, y) in enumerate(listed))


def locations_on_grid(count: int, grid: Grid3D,
                      region: Grid3D | None = None) -> tuple[TraceLocation, ...]:
    """Exactly ``count`` traces on a near-square lattice inside ``region``.

    The lattice is ``ceil(sqrt(count))`` columns wide and as many rows as
    that needs; when ``count`` is not a perfect rectangle the last row is
    short rather than the count being rounded, because the user asked for
    K traces and K is what they get.  Points sit strictly inside the
    bounds, so a single trace lands at the centre.
    """
    if count < 1:
        raise ConfigError(f"trace count must be at least 1, got {count}")
    n_col = int(np.ceil(np.sqrt(count)))
    n_row = int(np.ceil(count / n_col))
    (x0, x1), (y0, y1), _ = (region or grid).bounds
    xs = np.linspace(x0, x1, n_col + 2)[1:-1]
    ys = np.linspace(y0, y1, n_row + 2)[1:-1]
    out = []
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            if len(out) == count:
                break
            out.append(_snap(f"T{len(out) + 1}", float(x), float(y), grid))
    return tuple(out)


def resolve_locations(grid: Grid3D, layout: str = "wells", wells=None,
                      points=(), count: int = 9,
                      region: Grid3D | None = None) -> tuple[TraceLocation, ...]:
    """Pick the K trace locations according to the configured layout.

    Parameters
    ----------
    grid:
        The model the traces are sampled on.  Indices and the containment
        check are both against this, so a trace always names a real column.
    region:
        Where the ``grid`` layout spreads its lattice - the region of
        interest, normally the target domain.  Defaults to ``grid``.  The
        ``wells`` and ``points`` layouts ignore it: those positions are
        given rather than chosen, and a well is perfectly entitled to sit
        outside the imaging target while still being worth a trace.
    """
    if layout not in LAYOUTS:
        raise ConfigError(
            f"unknown trace layout {layout!r}; choose from {list(LAYOUTS)}")
    if layout == "wells":
        return locations_from_wells(wells, grid)
    if layout == "points":
        return locations_from_points(points, grid)
    return locations_on_grid(count, grid, region=region)


def snap_to(locations, grid: Grid3D) -> tuple[TraceLocation, ...]:
    """Re-index existing locations onto another grid, keeping their names.

    Layout and sampling ask different grids different questions: *where*
    the traces belong is a question about the region of interest, while
    ``ix``/``iy`` have to index the model actually being sampled, which is
    the padded propagation grid.  Doing that in one step would silently mix
    the two.
    """
    return tuple(_snap(site.name, site.x, site.y, grid) for site in locations)


# --------------------------------------------------------------- synthetic
def sparse_synthetic(model: AcousticModel, locations, wavelet: np.ndarray,
                     dt: float, t_max: float | None = None,
                     map_to_depth: bool = True) -> SparseSynthetic:
    """Build K vertical synthetic traces from a Vp/rho model.

    Parameters
    ----------
    model:
        The earth model to sample.
    locations:
        The trace positions, from :func:`resolve_locations`.
    wavelet:
        Source wavelet sampled at ``dt``.
    dt:
        Time sample interval, seconds.
    t_max:
        Length of the time axis; defaults to the deepest two-way time at
        the sampled columns.  Pass the survey's record length to keep the
        traces directly comparable with modelled gathers.
    map_to_depth:
        Also resample onto the depth axis, which is what allows a trace to
        be plotted beside the property profile it came from.
    """
    if dt <= 0:
        raise ConfigError(f"dt must be positive, got {dt}")
    sites = tuple(locations)
    if not sites:
        raise ConfigError("at least one trace location is required")

    grid = model.grid
    ix = np.array([site.ix for site in sites], dtype=int)
    iy = np.array([site.iy for site in sites], dtype=int)
    vp = np.asarray(model.vp, dtype=float)[ix, iy, :]
    impedance = np.asarray(model.impedance, dtype=float)[ix, iy, :]

    twt = time_from_depth(vp, grid.dz)
    t_max = float(t_max if t_max is not None else twt.max())
    nt = int(np.ceil(t_max / dt)) + 1

    traces, times, depth = synthetic_columns(twt, impedance, wavelet, dt, nt,
                                             map_to_depth=map_to_depth)
    return SparseSynthetic(
        locations=sites, traces=traces, times=times,
        depth_traces=depth if depth is not None else np.zeros_like(vp),
        depths=grid.axis("z"), twt=twt,
    )
