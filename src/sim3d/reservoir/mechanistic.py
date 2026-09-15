r"""Mechanistic pressure and saturation generators (spec sections 30-38).

Pressure halos and saturation fronts are generated **independently**, which
is the point: around an injector the pressure response usually extends far
beyond the water front, and around a producer the depletion extends beyond
any saturation change at all.  A tool that tied them together could not
pose the question the platform exists to answer.

Shapes
------
Every perturbation is defined on an anisotropic normalised radius

.. math:: r = \sqrt{(u/R_x)^2 + (v/R_y)^2 + (\Delta z/R_z)^2}

with :math:`u, v` the map offsets rotated into the perturbation's own
azimuth, so preferential spreading along strike, along a
high-permeability direction, or within a layer is expressed directly
(spec section 33).  The radial profile is then one of

===============  =====================================
``gaussian``     :math:`e^{-r^2}`
``exponential``  :math:`e^{-r}`
``power``        :math:`e^{-r^n}` - the section 32 form
``linear``       :math:`\max(1-r, 0)`
``step``         :math:`1` for :math:`r<1`, smoothly tapered outside
===============  =====================================

Faults
------
Crossing a fault multiplies a perturbation by that fault's
transmissibility (spec section 37), so a sealing fault truncates a
pressure halo and stops a flood front.  This is a geometric approximation
to compartmentalisation, not a flow solution, and it is applied by
comparing which side of each fault the well and the target cell are on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.errors import ConfigError, ValidationError
from ..core.geometry import to_principal
from ..geology.faults import FaultSet
from ..wells.well import Well, WellSet
from .state import ReservoirState

PROFILES = ("gaussian", "exponential", "power", "linear", "step")

#: Pseudo-time states of spec section 38, as multipliers on radii and amplitudes.
TIME_STATES: dict[str, float] = {
    "T0": 0.0,   # baseline
    "T1": 0.35,  # early response
    "T2": 0.60,  # intermediate
    "T3": 0.85,  # breakthrough
    "T4": 1.00,  # mature flood / depletion
}


def _profile(r: np.ndarray, kind: str, exponent: float = 2.0,
             taper: float = 0.25) -> np.ndarray:
    if kind == "gaussian":
        return np.exp(-(r**2))
    if kind == "exponential":
        return np.exp(-r)
    if kind == "power":
        return np.exp(-(r**exponent))
    if kind == "linear":
        return np.clip(1.0 - r, 0.0, 1.0)
    if kind == "step":
        # Smoothstep from 1 inside the front to 0 across a taper of width `taper`.
        t = np.clip((r - 1.0) / max(taper, 1e-9), 0.0, 1.0)
        return 1.0 - t * t * (3.0 - 2.0 * t)
    raise ConfigError(f"unknown profile {kind!r}; choose from {PROFILES}")


@dataclass
class _WellCentred:
    """Shared geometry for anything centred on a well."""

    well: str
    #: ``(major, minor, vertical)`` radii in metres.
    radius: tuple[float, float, float] = (600.0, 600.0, 120.0)
    #: Bearing of the major radius, degrees clockwise from +y (map north).
    azimuth: float = 0.0
    profile: str = "gaussian"
    exponent: float = 2.0
    taper: float = 0.25
    #: Restrict the effect to these depths; ``None`` uses the perforated interval.
    z_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise ConfigError(f"unknown profile {self.profile!r}; choose from {PROFILES}")
        if any(r <= 0 for r in self.radius):
            raise ConfigError(f"radii must be positive, got {self.radius}")

    def shape_field(self, grid, well: Well, faults: FaultSet | None,
                    scale: float = 1.0) -> np.ndarray:
        """Normalised 0-1 influence of this perturbation over the grid.

        ``scale`` grows the radii with pseudo-time.  At ``scale <= 0`` the
        perturbation does not exist at all: shrinking the radii towards zero
        instead would leave a one-cell spike at the well, which is not a
        small flood front but a different object entirely.
        """
        if scale <= 0.0:
            return np.zeros(grid.shape)
        x, y, z = np.meshgrid(grid.axis(0), grid.axis(1), grid.axis(2), indexing="ij")
        zc = 0.5 * sum(self.z_range) if self.z_range else well.position[2]
        dz = z - zc
        u, v = to_principal(x - well.x, y - well.y, self.azimuth)
        rx, ry, rz = (max(r * scale, 1e-6) for r in self.radius)
        r = np.sqrt((u / rx) ** 2 + (v / ry) ** 2 + (dz / rz) ** 2)
        shape = _profile(r, self.profile, self.exponent, self.taper)

        if self.z_range is not None:
            shape = np.where((z >= self.z_range[0]) & (z <= self.z_range[1]), shape, 0.0)
        if faults is not None and len(faults):
            shape = shape * _fault_attenuation(faults, well, x, y, z)
        return shape


def _fault_attenuation(faults: FaultSet, well: Well, x, y, z) -> np.ndarray:
    """Multiplier for crossing faults between the well and each cell.

    A cell on the opposite side of a fault from the well is multiplied by
    that fault's transmissibility.  Sealing faults therefore truncate the
    perturbation at the fault plane.
    """
    wx, wy, wz = well.position
    out = np.ones(np.asarray(x).shape)
    for fault in faults:
        well_side = float(fault.hanging_wall_fraction(
            np.array([wx]), np.array([wy]), np.array([wz]))[0]) > 0.5
        cell_side = fault.hanging_wall_fraction(x, y, z) > 0.5
        crosses = cell_side != well_side
        out = np.where(crosses, out * fault.transmissibility, out)
    return out


@dataclass
class PressureHalo(_WellCentred):
    """A pore-pressure change around a well (spec section 32).

    ``delta_p`` is signed and in Pa: positive for injection support,
    negative for depletion.  The halo is independent of any saturation
    front around the same well, and normally much larger.
    """

    delta_p: float = 40.0e5

    def apply(self, state: ReservoirState, well: Well, faults: FaultSet | None,
              scale: float = 1.0) -> np.ndarray:
        return self.delta_p * scale * self.shape_field(state.grid, well, faults)


@dataclass
class SaturationFront(_WellCentred):
    """An advancing water front around an injector (spec section 31).

    ``target_sw`` is the water saturation reached at the well; the front
    tapers back to the baseline value outside ``radius``.  The displaced
    volume is taken from oil first and from gas only once oil is exhausted,
    which is the ordering a waterflood actually follows.
    """

    target_sw: float = 0.75
    profile: str = "step"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.target_sw <= 1.0:
            raise ConfigError(f"target_sw must be in [0, 1], got {self.target_sw}")


@dataclass
class GasBreakout(_WellCentred):
    """Free gas appearing in a depleted region (spec section 34).

    ``target_sg`` is the gas saturation at the centre.  Gas comes out of
    the oil, so it is taken from oil first and from water only if the cell
    has no oil left.
    """

    target_sg: float = 0.10
    profile: str = "step"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.target_sg <= 1.0:
            raise ConfigError(f"target_sg must be in [0, 1], got {self.target_sg}")


@dataclass
class ReservoirScenario:
    """A named set of perturbations applied to a baseline state.

    :meth:`apply` returns a new state; the baseline is never mutated, which
    is what lets the same baseline feed the four 4D scenarios of section 48.
    """

    name: str
    pressure: list[PressureHalo] = field(default_factory=list)
    water_fronts: list[SaturationFront] = field(default_factory=list)
    gas: list[GasBreakout] = field(default_factory=list)
    time_state: str = "T4"

    @property
    def scale(self) -> float:
        """Growth multiplier for the chosen pseudo-time state."""
        try:
            return TIME_STATES[self.time_state]
        except KeyError:
            raise ConfigError(
                f"unknown time state {self.time_state!r}; choose from {sorted(TIME_STATES)}"
            ) from None

    def apply(self, baseline: ReservoirState, wells: WellSet,
              faults: FaultSet | None = None,
              include_pressure: bool = True,
              include_saturation: bool = True) -> ReservoirState:
        """Build a monitor state from ``baseline``.

        ``include_pressure`` and ``include_saturation`` are what make the
        pressure-only and saturation-only scenarios of sections 50-51 fall
        out of one code path: they are the *same* perturbations with one
        half switched off, not a rescaling of a combined answer.
        """
        scale = self.scale
        state = baseline.copy(name=self.name)
        res = baseline.reservoir_mask
        notes = [f"time state {self.time_state} (scale {scale:g})"]

        if include_pressure:
            total = np.zeros(baseline.grid.shape)
            for halo in self.pressure:
                well = wells[halo.well]
                total += halo.apply(baseline, well, faults, scale)
                notes.append(
                    f"pressure {halo.delta_p / 1e5:+.0f} bar at {halo.well} "
                    f"over {halo.radius[0]:.0f} m ({halo.profile})"
                )
            state.pressure = baseline.pressure + np.where(res, total, 0.0)

        if include_saturation:
            sw, so, sg = baseline.sw.copy(), baseline.so.copy(), baseline.sg.copy()
            for front in self.water_fronts:
                well = wells[front.well]
                shape = front.shape_field(baseline.grid, well, faults, scale)
                shape = np.where(res, shape, 0.0)
                target = baseline.sw + (front.target_sw - baseline.sw) * shape
                sw = np.maximum(sw, np.clip(target, 0.0, 1.0))
                notes.append(
                    f"water front to Sw={front.target_sw:g} at {front.well} "
                    f"over {front.radius[0]:.0f} m"
                )
            sw, so, sg = _rebalance(baseline, sw, so, sg, grew="sw")

            for breakout in self.gas:
                well = wells[breakout.well]
                shape = breakout.shape_field(baseline.grid, well, faults, scale)
                shape = np.where(res, shape, 0.0)
                target = baseline.sg + (breakout.target_sg - baseline.sg) * shape
                sg = np.maximum(sg, np.clip(target, 0.0, 1.0))
                notes.append(
                    f"gas breakout to Sg={breakout.target_sg:g} at {breakout.well} "
                    f"over {breakout.radius[0]:.0f} m"
                )
            sw, so, sg = _rebalance(baseline, sw, so, sg, grew="sg")
            state.sw, state.so, state.sg = sw, so, sg

        state.provenance = [*baseline.provenance, *notes]
        state.validate()
        return state

    def describe(self) -> str:
        lines = [f"Scenario '{self.name}' at {self.time_state}:"]
        for halo in self.pressure:
            lines.append(f"  {halo.well}: dP = {halo.delta_p / 1e5:+.1f} bar, "
                         f"radii {halo.radius} m, {halo.profile}")
        for front in self.water_fronts:
            lines.append(f"  {front.well}: water front to Sw = {front.target_sw:g}, "
                         f"radii {front.radius} m")
        for breakout in self.gas:
            lines.append(f"  {breakout.well}: gas to Sg = {breakout.target_sg:g}, "
                         f"radii {breakout.radius} m")
        return "\n".join(lines)


def _rebalance(baseline: ReservoirState, sw, so, sg, grew: str):
    """Restore ``Sw + So + Sg = 1`` after one phase has been increased.

    The increase is taken from oil first, then from the remaining phase.
    Doing the bookkeeping explicitly - rather than renormalising all three -
    keeps the displacement physical: injected water displaces oil, and gas
    comes out of oil, not out of the connate water.

    Cells that did not change are left untouched rather than recomputed, so
    a scenario that perturbs nothing returns the baseline bit for bit. That
    exactness is what makes the section 131 null test a real check on the
    pipeline instead of a check on floating-point luck.
    """
    if grew == "sw":
        gain = np.maximum(sw - baseline.sw, 0.0)
        from_oil = np.minimum(gain, so)
        from_gas = np.minimum(gain - from_oil, sg)
        touched = gain > 0.0
        so = np.where(touched, so - from_oil, so)
        sg = np.where(touched, sg - from_gas, sg)
        sw = np.where(touched, 1.0 - so - sg, sw)
    else:
        gain = np.maximum(sg - baseline.sg, 0.0)
        from_oil = np.minimum(gain, so)
        from_water = np.minimum(gain - from_oil, np.maximum(sw, 0.0))
        touched = gain > 0.0
        so = np.where(touched, so - from_oil, so)
        sw = np.where(touched, sw - from_water, sw)
        sg = np.where(touched, 1.0 - sw - so, sg)
    total = sw + so + sg
    if np.any(np.abs(total - 1.0) > 1e-9):
        raise ValidationError(
            f"saturation rebalancing failed to close: worst residual "
            f"{np.max(np.abs(total - 1.0)):.3e}"
        )
    return np.clip(sw, 0.0, 1.0), np.clip(so, 0.0, 1.0), np.clip(sg, 0.0, 1.0)
