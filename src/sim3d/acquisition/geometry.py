"""Synthetic acquisition, starting with sparse 3D OBN (spec sections 66-72).

Ocean-bottom nodes are the first geometry because they suit reservoir-scale
mechanistic studies: the geometry is simple, azimuth coverage is good, the
node positions are repeatable between surveys, and the shot count - not the
receiver count - sets the cost of a full-wave experiment.

Nothing here is hard-coded.  Node and shot spacings are parameters, and the
values in the docstrings are the typical CPU-research starting points of
section 67, not defaults the physics depends on.

Fold and offset/azimuth distributions are geometric only.  Real
illumination is a wavefield property, and the platform's point is that a
weak 4D anomaly may be an illumination effect rather than a geological one,
so these summaries are a first screen and not the answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.errors import ConfigError
from ..core.grid import Grid3D


@dataclass
class Acquisition:
    """Explicit source and receiver coordinates, in metres."""

    sources: np.ndarray       #: ``(n_sources, 3)``
    receivers: np.ndarray     #: ``(n_receivers, 3)``
    name: str = "acquisition"

    def __post_init__(self) -> None:
        for label, arr in (("sources", self.sources), ("receivers", self.receivers)):
            a = np.atleast_2d(np.asarray(arr, dtype=float))
            if a.ndim != 2 or a.shape[1] != 3:
                raise ConfigError(f"{label} must have shape (n, 3), got {a.shape}")
            setattr(self, label, a)

    @property
    def n_sources(self) -> int:
        return int(self.sources.shape[0])

    @property
    def n_receivers(self) -> int:
        return int(self.receivers.shape[0])

    @property
    def n_traces(self) -> int:
        return self.n_sources * self.n_receivers

    def offsets(self) -> np.ndarray:
        """Source-receiver offsets for every trace, ``(n_sources, n_receivers)``."""
        d = self.sources[:, None, :2] - self.receivers[None, :, :2]
        return np.hypot(d[..., 0], d[..., 1])

    def azimuths(self) -> np.ndarray:
        """Trace azimuths in degrees clockwise from ``+y``, in ``[0, 360)``."""
        d = self.receivers[None, :, :2] - self.sources[:, None, :2]
        return np.degrees(np.arctan2(d[..., 0], d[..., 1])) % 360.0

    def within(self, grid: Grid3D) -> bool:
        """True when every source and receiver lies inside ``grid``."""
        return all(grid.contains(p) for p in (*self.sources, *self.receivers))

    def outside(self, grid: Grid3D) -> list[tuple[float, float, float]]:
        """The positions that fall outside ``grid`` - to be reported, not clipped."""
        return [tuple(p) for p in (*self.sources, *self.receivers)
                if not grid.contains(p)]

    def decimated(self, source_step: int = 1, receiver_step: int = 1) -> "Acquisition":
        """A sparser copy.  Decimation is an explicit user act, never automatic."""
        if source_step < 1 or receiver_step < 1:
            raise ConfigError("decimation steps must be >= 1")
        return Acquisition(self.sources[::source_step], self.receivers[::receiver_step],
                           name=f"{self.name}|decimated({source_step},{receiver_step})")

    def summary(self) -> str:
        off = self.offsets()
        return "\n".join([
            f"Acquisition '{self.name}': {self.n_sources} sources, "
            f"{self.n_receivers} receivers, {self.n_traces:,} traces",
            f"  offsets  {off.min():.0f} - {off.max():.0f} m "
            f"(median {np.median(off):.0f} m)",
            f"  sources  z = {np.unique(np.round(self.sources[:, 2], 3))} m",
            f"  nodes    z = {np.unique(np.round(self.receivers[:, 2], 3))} m",
        ])


@dataclass
class OBNGeometry:
    """A sparse 3D ocean-bottom-node survey.

    Typical CPU-research starting values are 100-250 m node spacing and
    50-150 m shot spacing (section 67); the defaults below sit in that range
    but carry no special status.
    """

    centre: tuple[float, float] = (1500.0, 1500.0)
    receiver_spacing: float = 200.0
    receiver_extent: float = 1600.0
    source_spacing: float = 100.0
    source_line_spacing: float = 200.0
    source_extent: float = 2000.0
    receiver_depth: float = 400.0
    source_depth: float = 380.0
    name: str = "obn"

    def __post_init__(self) -> None:
        for label in ("receiver_spacing", "source_spacing", "source_line_spacing",
                      "receiver_extent", "source_extent"):
            if getattr(self, label) <= 0:
                raise ConfigError(f"{label} must be positive, got {getattr(self, label)}")

    def _axis(self, spacing: float, extent: float, centre: float) -> np.ndarray:
        n = max(int(np.floor(extent / spacing)) + 1, 1)
        return centre + (np.arange(n) - (n - 1) / 2) * spacing

    def build(self) -> Acquisition:
        """Generate the node grid and the shot carpet."""
        rx = self._axis(self.receiver_spacing, self.receiver_extent, self.centre[0])
        ry = self._axis(self.receiver_spacing, self.receiver_extent, self.centre[1])
        gx, gy = np.meshgrid(rx, ry, indexing="ij")
        receivers = np.column_stack([
            gx.ravel(), gy.ravel(), np.full(gx.size, self.receiver_depth)])

        sx = self._axis(self.source_spacing, self.source_extent, self.centre[0])
        sy = self._axis(self.source_line_spacing, self.source_extent, self.centre[1])
        hx, hy = np.meshgrid(sx, sy, indexing="ij")
        sources = np.column_stack([
            hx.ravel(), hy.ravel(), np.full(hx.size, self.source_depth)])
        return Acquisition(sources=sources, receivers=receivers, name=self.name)

    def aperture_for(self, target_depth: float, target_half_width: float) -> float:
        """Half-aperture needed to illuminate a target to 45 degrees.

        A rough sizing aid: a reflector at ``target_depth`` beneath the
        acquisition plane needs roughly ``depth`` of extra aperture on each
        side of the target to reach 45 degrees of incidence.
        """
        return target_half_width + max(target_depth - self.receiver_depth, 0.0)


def fold_map(acquisition: Acquisition, grid: Grid3D, depth: float,
             bin_size: float = 50.0, max_offset: float | None = None):
    """Common-midpoint fold in map view at a given target depth.

    Returns ``(x_centres, y_centres, fold)``.  This is straight-ray CMP
    counting, which is a geometric proxy: it says nothing about whether the
    wavefield actually reaches the target.
    """
    src = acquisition.sources
    rec = acquisition.receivers
    mx = 0.5 * (src[:, None, 0] + rec[None, :, 0])
    my = 0.5 * (src[:, None, 1] + rec[None, :, 1])
    keep = np.ones(mx.shape, dtype=bool)
    if max_offset is not None:
        keep &= acquisition.offsets() <= max_offset

    (x0, x1), (y0, y1) = grid.bounds[0], grid.bounds[1]
    xedges = np.arange(x0, x1 + bin_size, bin_size)
    yedges = np.arange(y0, y1 + bin_size, bin_size)
    fold, _, _ = np.histogram2d(mx[keep], my[keep], bins=(xedges, yedges))
    return (0.5 * (xedges[:-1] + xedges[1:]),
            0.5 * (yedges[:-1] + yedges[1:]),
            fold)


def offset_distribution(acquisition: Acquisition, n_bins: int = 20):
    """Histogram of source-receiver offsets: ``(centres, counts)``."""
    off = acquisition.offsets().ravel()
    counts, edges = np.histogram(off, bins=n_bins)
    return 0.5 * (edges[:-1] + edges[1:]), counts


def azimuth_distribution(acquisition: Acquisition, n_bins: int = 36,
                         min_offset: float = 0.0):
    """Histogram of trace azimuths in degrees: ``(centres, counts)``.

    Near-zero-offset traces have an ill-defined azimuth, so ``min_offset``
    excludes them rather than letting them pile into one bin.
    """
    az = acquisition.azimuths()
    off = acquisition.offsets()
    counts, edges = np.histogram(az[off >= min_offset], bins=n_bins, range=(0.0, 360.0))
    return 0.5 * (edges[:-1] + edges[1:]), counts
