"""Wells and well patterns (spec sections 26-29).

Initial trajectories are vertical; the :class:`Well` interface carries a
trajectory so that deviated and horizontal wells slot in later without the
reservoir and analysis code changing.

Minimum spacing is a *warning*, not a prohibition (spec section 27): a
deliberately close pair is a legitimate research case, and the software's
job is to say so, not to refuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

from ..core.errors import ConfigError

#: Default minimum horizontal well spacing, metres.
MIN_SPACING_DEFAULT = 500.0

ROLES = ("injector", "producer", "observation")


@dataclass
class Well:
    """A single well."""

    name: str
    role: str
    x: float
    y: float
    #: Perforated interval as ``(z_top, z_base)`` in metres.
    perforation: tuple[float, float] = (1200.0, 1350.0)
    #: Reservoir layer names this well is completed in; empty means all.
    active_layers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ConfigError(f"well role must be one of {ROLES}, got {self.role!r}")
        top, base = self.perforation
        if base <= top:
            raise ConfigError(
                f"well {self.name}: perforation {self.perforation} must increase downwards"
            )

    @property
    def position(self) -> tuple[float, float, float]:
        """Mid-perforation position, used as the centre of well-centred effects."""
        return (self.x, self.y, 0.5 * (self.perforation[0] + self.perforation[1]))

    def trajectory(self, z: np.ndarray) -> np.ndarray:
        """Map coordinates at each depth; vertical for now, ``(n, 2)``."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        return np.column_stack([np.full(z.size, self.x), np.full(z.size, self.y)])

    def horizontal_distance(self, x, y) -> np.ndarray:
        """Map-view distance from the well to each point, metres."""
        return np.hypot(np.asarray(x, dtype=float) - self.x,
                        np.asarray(y, dtype=float) - self.y)

    def perforation_mask(self, z) -> np.ndarray:
        """True where a depth lies within the perforated interval."""
        z = np.asarray(z, dtype=float)
        return (z >= self.perforation[0]) & (z <= self.perforation[1])


@dataclass
class WellSet:
    """A collection of wells, with the spacing diagnostics of section 27."""

    wells: list[Well] = field(default_factory=list)
    min_spacing: float = MIN_SPACING_DEFAULT

    def __iter__(self) -> Iterator[Well]:
        return iter(self.wells)

    def __len__(self) -> int:
        return len(self.wells)

    def __getitem__(self, key):
        if isinstance(key, str):
            for well in self.wells:
                if well.name == key:
                    return well
            raise KeyError(f"no well named {key!r}; have {[w.name for w in self.wells]}")
        return self.wells[key]

    def by_role(self, role: str) -> list[Well]:
        if role not in ROLES:
            raise ConfigError(f"unknown role {role!r}; choose from {ROLES}")
        return [w for w in self.wells if w.role == role]

    @property
    def injectors(self) -> list[Well]:
        return self.by_role("injector")

    @property
    def producers(self) -> list[Well]:
        return self.by_role("producer")

    def distance_matrix(self) -> np.ndarray:
        """Pairwise horizontal distances, metres."""
        p = np.array([[w.x, w.y] for w in self.wells], dtype=float)
        if len(p) == 0:
            return np.zeros((0, 0))
        return np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)

    def spacing_violations(self) -> list[tuple[str, str, float]]:
        """Well pairs closer than ``min_spacing``, as ``(name_a, name_b, metres)``."""
        d = self.distance_matrix()
        out = []
        for i in range(len(self.wells)):
            for j in range(i + 1, len(self.wells)):
                if d[i, j] < self.min_spacing:
                    out.append((self.wells[i].name, self.wells[j].name, float(d[i, j])))
        return out

    def check_spacing(self) -> list[str]:
        """Warnings for close pairs.  Never raises: close wells are allowed."""
        return [
            f"{a} and {b} are {dist:.0f} m apart, below the {self.min_spacing:.0f} m "
            f"minimum spacing; allowed, but confirm this is intended"
            for a, b, dist in self.spacing_violations()
        ]

    def summary(self) -> str:
        lines = [f"{len(self.wells)} wells "
                 f"({len(self.injectors)} injectors, {len(self.producers)} producers)"]
        for w in self.wells:
            lines.append(f"  {w.name:6s} {w.role:12s} ({w.x:7.0f}, {w.y:7.0f}) m, "
                         f"perforated {w.perforation[0]:.0f}-{w.perforation[1]:.0f} m")
        d = self.distance_matrix()
        if len(self.wells) > 1:
            off_diagonal = d[~np.eye(len(d), dtype=bool)]
            lines.append(f"  spacing: min {off_diagonal.min():.0f} m, "
                         f"max {off_diagonal.max():.0f} m")
        lines += [f"  warning: {w}" for w in self.check_spacing()]
        return "\n".join(lines)


# --- pattern templates (spec section 28) ----------------------------------
def _grid_positions(centre, spacing, nx, ny):
    cx, cy = centre
    xs = cx + (np.arange(nx) - (nx - 1) / 2) * spacing
    ys = cy + (np.arange(ny) - (ny - 1) / 2) * spacing
    return xs, ys


def line_drive(centre=(1500.0, 1500.0), spacing=600.0, n_injectors=2, n_producers=3,
               perforation=(1200.0, 1350.0), separation=700.0):
    """A row of injectors facing a row of producers."""
    wells = []
    for i in range(n_injectors):
        y = centre[1] + (i - (n_injectors - 1) / 2) * spacing
        wells.append(Well(f"I{i + 1}", "injector", centre[0] - separation / 2, y, perforation))
    for i in range(n_producers):
        y = centre[1] + (i - (n_producers - 1) / 2) * spacing
        wells.append(Well(f"P{i + 1}", "producer", centre[0] + separation / 2, y, perforation))
    return WellSet(wells)


def five_spot(centre=(1500.0, 1500.0), spacing=700.0, perforation=(1200.0, 1350.0)):
    """Four producers at the corners around one central injector."""
    half = spacing / 2
    wells = [Well("I1", "injector", centre[0], centre[1], perforation)]
    for i, (dx, dy) in enumerate([(-half, -half), (half, -half), (half, half), (-half, half)]):
        wells.append(Well(f"P{i + 1}", "producer", centre[0] + dx, centre[1] + dy, perforation))
    return WellSet(wells)


def inverted_five_spot(centre=(1500.0, 1500.0), spacing=700.0,
                       perforation=(1200.0, 1350.0)):
    """Four injectors around one central producer."""
    ws = five_spot(centre, spacing, perforation)
    flipped = []
    for w in ws:
        role = "producer" if w.role == "injector" else "injector"
        prefix = "P" if role == "producer" else "I"
        flipped.append(Well(f"{prefix}{w.name[1:]}", role, w.x, w.y, w.perforation))
    return WellSet(flipped)


def staggered_line_drive(centre=(1500.0, 1500.0), spacing=600.0, separation=700.0,
                         perforation=(1200.0, 1350.0)):
    """A line drive with the producer row offset by half a spacing."""
    ws = line_drive(centre, spacing, 2, 3, perforation, separation)
    for w in ws:
        if w.role == "producer":
            w.y += spacing / 2
    return ws


def demonstration_pattern(centre=(1500.0, 1500.0), perforation=(1200.0, 1350.0)):
    """The section 29 reference pattern: 2 injectors, 3 producers, >= 500 m apart."""
    wells = [
        Well("I1", "injector", centre[0] - 550.0, centre[1] - 320.0, perforation),
        Well("I2", "injector", centre[0] + 480.0, centre[1] + 430.0, perforation),
        Well("P1", "producer", centre[0] + 20.0, centre[1] - 560.0, perforation),
        Well("P2", "producer", centre[0] - 520.0, centre[1] + 500.0, perforation),
        Well("P3", "producer", centre[0] + 600.0, centre[1] - 180.0, perforation),
    ]
    return WellSet(wells)


#: Pattern name -> builder.
PATTERNS: dict[str, Callable[..., WellSet]] = {
    "line_drive": line_drive,
    "five_spot": five_spot,
    "inverted_five_spot": inverted_five_spot,
    "staggered_line_drive": staggered_line_drive,
    "demonstration": demonstration_pattern,
}


def pattern(name: str, **kwargs) -> WellSet:
    """Build a named well pattern."""
    try:
        return PATTERNS[name](**kwargs)
    except KeyError:
        raise ConfigError(
            f"unknown well pattern {name!r}; choose from {sorted(PATTERNS)}"
        ) from None
