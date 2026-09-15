"""Facies definitions and their default property ranges (spec section 24)."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.errors import ConfigError


@dataclass(frozen=True)
class Facies:
    """A rock type with default property ranges and a mineral composition.

    The ranges are typical clastic-basin values used to seed a mechanistic
    model; they are starting points for an experiment, not measurements of
    any particular field.
    """

    name: str
    code: int
    composition: dict[str, float]
    porosity: tuple[float, float]
    vsh: tuple[float, float]
    ntg: tuple[float, float]
    #: Indicative permeability range in mD, carried for flow-style reasoning.
    permeability: tuple[float, float] = (0.001, 1.0)
    is_reservoir: bool = False

    def __post_init__(self) -> None:
        total = sum(self.composition.values())
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(
                f"facies {self.name}: mineral fractions sum to {total:g}, not 1"
            )
        for label, (lo, hi) in (("porosity", self.porosity), ("vsh", self.vsh),
                                ("ntg", self.ntg)):
            if not 0.0 <= lo <= hi <= 1.0:
                raise ConfigError(f"facies {self.name}: {label} range {(lo, hi)} invalid")

    def midpoint(self, attribute: str) -> float:
        lo, hi = getattr(self, attribute)
        return 0.5 * (lo + hi)


#: Default facies catalogue.  Extend or override per project.
FACIES: dict[str, Facies] = {
    "shale": Facies(
        "shale", 1, {"clay": 0.75, "quartz": 0.25},
        porosity=(0.05, 0.15), vsh=(0.7, 0.95), ntg=(0.0, 0.05),
        permeability=(1e-5, 1e-2),
    ),
    "sandy_shale": Facies(
        "sandy_shale", 2, {"clay": 0.45, "quartz": 0.55},
        porosity=(0.10, 0.20), vsh=(0.35, 0.65), ntg=(0.1, 0.4),
        permeability=(0.1, 10.0),
    ),
    "shaly_sandstone": Facies(
        "shaly_sandstone", 3, {"clay": 0.20, "quartz": 0.80},
        porosity=(0.15, 0.25), vsh=(0.15, 0.35), ntg=(0.5, 0.8),
        permeability=(10.0, 200.0), is_reservoir=True,
    ),
    "clean_sandstone": Facies(
        "clean_sandstone", 4, {"clay": 0.05, "quartz": 0.95},
        porosity=(0.20, 0.32), vsh=(0.0, 0.12), ntg=(0.9, 1.0),
        permeability=(200.0, 3000.0), is_reservoir=True,
    ),
    "carbonate": Facies(
        "carbonate", 5, {"calcite": 0.90, "dolomite": 0.10},
        porosity=(0.05, 0.20), vsh=(0.0, 0.10), ntg=(0.4, 0.9),
        permeability=(0.1, 500.0), is_reservoir=True,
    ),
    "coal": Facies(
        "coal", 6, {"clay": 0.5, "quartz": 0.5},
        porosity=(0.02, 0.10), vsh=(0.3, 0.6), ntg=(0.0, 0.0),
        permeability=(1e-4, 1.0),
    ),
    "salt": Facies(
        "salt", 7, {"halite": 1.0},
        porosity=(0.001, 0.02), vsh=(0.0, 0.0), ntg=(0.0, 0.0),
        permeability=(1e-9, 1e-6),
    ),
    "basement": Facies(
        "basement", 8, {"quartz": 0.45, "feldspar": 0.55},
        porosity=(0.001, 0.03), vsh=(0.0, 0.1), ntg=(0.0, 0.0),
        permeability=(1e-6, 1e-3),
    ),
}


def get_facies(name: str, catalogue: dict[str, Facies] | None = None) -> Facies:
    table = catalogue or FACIES
    try:
        return table[name]
    except KeyError:
        raise ConfigError(
            f"unknown facies {name!r}; known: {sorted(table)}. Define a Facies "
            f"rather than substituting a different rock type."
        ) from None
