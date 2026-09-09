"""The four earth models of a 4D experiment (spec sections 48-54).

Pressure/saturation separation is a first-class concept here, not a display
option.  Every baseline-to-monitor comparison produces four *distinct
physical earth models*:

============================  ===========================================
``baseline``                  P, Sw, So, Sg all at their initial values
``pressure_only``             monitor pressure, baseline saturations
``saturation_only``           baseline pressure, monitor saturations
``combined``                  monitor pressure and monitor saturations
============================  ===========================================

Each is run through the complete rock-physics chain independently
(section 156): none is derived by scaling another, and each is capable of
its own forward modelling, acquisition and migration.  The interaction term
that falls out of this - the difference between the combined response and
the sum of the two isolated ones - is the quantity that says how nonlinear
the coupled response really is, and it cannot be recovered from a scaled
difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.errors import ConfigError
from ..core.grid import Grid3D
from ..geology.builder import GeologyModel
from ..geology.faults import FaultSet
from ..reservoir.mechanistic import ReservoirScenario
from ..reservoir.state import ReservoirState
from ..rockphysics.model import RockPhysicsConfig, RockPhysicsResult, elastic_from_state
from ..wave.acoustic import AcousticModel
from ..wells.well import WellSet

SCENARIO_NAMES = ("baseline", "pressure_only", "saturation_only", "combined")

GRAVITY = 9.81


@dataclass
class FourDStates:
    """The four reservoir states."""

    baseline: ReservoirState
    pressure_only: ReservoirState
    saturation_only: ReservoirState
    combined: ReservoirState

    def __getitem__(self, name: str) -> ReservoirState:
        if name not in SCENARIO_NAMES:
            raise ConfigError(f"unknown scenario {name!r}; choose from {SCENARIO_NAMES}")
        return getattr(self, name)

    def items(self):
        return ((name, self[name]) for name in SCENARIO_NAMES)

    def check_isolation(self, tolerance: float = 1e-12) -> None:
        """Assert that each scenario changed only what it was supposed to.

        This is the section 132-133 test built into the construction path:
        the pressure-only state must have baseline saturations exactly, and
        the saturation-only state must have baseline pressure exactly.
        """
        d = self.pressure_only.difference(self.baseline)
        for key in ("dSw", "dSo", "dSg"):
            if np.max(np.abs(d[key])) > tolerance:
                raise ConfigError(
                    f"the pressure-only scenario changed {key} by up to "
                    f"{np.max(np.abs(d[key])):.3e}; it must change pressure alone"
                )
        d = self.saturation_only.difference(self.baseline)
        if np.max(np.abs(d["dP"])) > tolerance:
            raise ConfigError(
                f"the saturation-only scenario changed pressure by up to "
                f"{np.max(np.abs(d['dP'])):.3e} Pa; it must change saturation alone"
            )


@dataclass
class FourDEarth:
    """Rock-physics results and acoustic earth models for the four scenarios."""

    states: FourDStates
    rock_physics: dict[str, RockPhysicsResult]
    models: dict[str, AcousticModel]
    confining_pressure: np.ndarray

    def property_deltas(self, attribute: str) -> dict[str, np.ndarray]:
        """``{scenario: value - baseline_value}`` for a rock-physics attribute."""
        base = getattr(self.rock_physics["baseline"], attribute)
        return {name: getattr(self.rock_physics[name], attribute) - base
                for name in SCENARIO_NAMES[1:]}

    def summary(self) -> str:
        lines = ["4D earth models:"]
        for name in SCENARIO_NAMES:
            m = self.models[name]
            lines.append(f"  {name:16s} Vp {m.vmin:7.1f} - {m.vmax:7.1f} m/s, "
                         f"rho {m.rho.min():7.1f} - {m.rho.max():7.1f} kg/m3")
        for attribute, scale, unit in (("vp", 1.0, "m/s"), ("ai", 1e6, "1e6 kg/m2/s")):
            for name, delta in self.property_deltas(attribute).items():
                lines.append(f"  d{attribute.upper()} {name:16s} "
                             f"{delta.min() / scale:+9.4f} to {delta.max() / scale:+9.4f} {unit}")
        return "\n".join(lines)


def build_states(baseline: ReservoirState, scenario: ReservoirScenario,
                 wells: WellSet, faults: FaultSet | None = None) -> FourDStates:
    """Build the four reservoir states from one baseline and one scenario."""
    states = FourDStates(
        baseline=baseline.copy(name="baseline"),
        pressure_only=scenario.apply(baseline, wells, faults,
                                     include_saturation=False),
        saturation_only=scenario.apply(baseline, wells, faults,
                                       include_pressure=False),
        combined=scenario.apply(baseline, wells, faults),
    )
    states.pressure_only.name = "pressure_only"
    states.saturation_only.name = "saturation_only"
    states.combined.name = "combined"
    states.check_isolation()
    return states


def confining_pressure_from_density(grid: Grid3D, density: np.ndarray,
                                    surface_pressure: float = 101325.0) -> np.ndarray:
    """Lithostatic stress by integrating a density column, in Pa.

    ``P_conf(z) = P_surface + g * integral_{z_top}^{z} rho dz'``, integrated
    with the trapezoidal rule down each column.  ``surface_pressure`` is the
    stress at the **top of this grid**, so a grid that starts below the
    surface must be given the weight of the section above it - otherwise the
    overburden is silently missing and every effective stress is too low.

    The integral uses the model's own density, which itself depends on
    effective stress, so :func:`build_earth_models` runs one refinement pass:
    a constant-density first guess, then a re-integration through the
    resulting density.  One pass is enough because the overburden density is
    set by compaction, not by the reservoir's own few tens of bar of change.
    """
    rho = np.asarray(density, dtype=float)
    if rho.shape != grid.shape:
        raise ConfigError(f"density shape {rho.shape} does not match grid {grid.shape}")
    # Trapezoidal rule: dz * (rho_0/2 + rho_1 + ... + rho_{k-1} + rho_k/2).
    column = (np.cumsum(rho, axis=2) - 0.5 * (rho + rho[:, :, :1])) * grid.dz
    return surface_pressure + GRAVITY * column


def build_earth_models(states: FourDStates, geology: GeologyModel,
                       config: RockPhysicsConfig | None = None,
                       overburden_density: float = 2300.0,
                       refine_confining: bool = True) -> FourDEarth:
    """Run every scenario through the rock-physics chain, independently.

    The confining stress is computed once, from the *baseline* density, and
    then held fixed across the four scenarios.  That is deliberate: a
    monitor survey's overburden has not changed, so letting the confining
    stress drift with the reservoir's own density would put a spurious
    stress change into the pressure-only case.
    """
    config = config or RockPhysicsConfig()
    grid = states.baseline.grid
    composition = geology.composition

    depth = grid.axis(2)[None, None, :] * np.ones(grid.shape)
    confining = confining_pressure_from_density(
        grid, np.full(grid.shape, overburden_density))

    def run(state: ReservoirState, conf: np.ndarray) -> RockPhysicsResult:
        return elastic_from_state(
            porosity=state.porosity, composition=composition,
            saturations=state.saturations, pore_pressure=state.pressure,
            confining_pressure=conf, config=config,
        )

    if refine_confining:
        first = run(states.baseline, confining)
        confining = confining_pressure_from_density(grid, first.rho)

    rock_physics = {name: run(state, confining) for name, state in states.items()}
    models = {
        name: AcousticModel(grid, result.vp, result.rho, name=name)
        for name, result in rock_physics.items()
    }
    return FourDEarth(states=states, rock_physics=rock_physics, models=models,
                      confining_pressure=confining)
