r"""Well control modes, schedules and physically-grounded rate defaults.

Requirements sections 9, 10 and 12.

The rule that shapes this module: **the software must not invent a rate
simply to make an anomaly visible.**  A default rate is derived from the
rock the well is actually completed in - net thickness, permeability, fluid
viscosity, spacing and a stated drawdown - through the pseudo-steady-state
radial inflow relation

.. math::
    q = \frac{2\pi k h \,\Delta p}
             {\mu B\left(\ln(r_e/r_w) - \tfrac34 + s\right)}

so a thin, tight or poorly-completed well gets a small number and says why.

Deliverability alone is not a rate, though.  A hundred metres of net
thousand-millidarcy sand will accept several hundred thousand barrels a day
at a normal drawdown, and no one develops a field that way: the rate is
chosen to give the pattern a sensible life, and the inflow relation only
says whether the well can sustain it.  The suggestion is therefore the
**smaller** of the inflow-limited rate and the rate that turns over the
well's own drainage pore volume in a target number of years, and it records
which of the two bound it.

The user can override any of it; what they cannot do is get a plausible
number by accident.

The sanity checks are warnings with reasons, never refusals.  A voidage
replacement ratio of 3 is a legitimate experiment; it is also almost always
a mistake, and the difference is something only the user knows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..core.errors import ConfigError
from ..core.units import DAY, STB, pa_to_psi, si_to_stb_per_day, stb_per_day_to_si

#: Wellbore radius used by the inflow relation and the Peaceman model, m.
WELLBORE_RADIUS = 0.1
#: Default drawdown (producer) or overpressure (injector) if none is given, Pa.
DEFAULT_DRAWDOWN = 500.0 * 6894.757293168      # 500 psi
#: Nominal viscosities, Pa.s.  Batzle-Wang gives moduli and density, not
#: viscosity, so these are stated defaults rather than derived quantities.
OIL_VISCOSITY = 1.0e-3
WATER_VISCOSITY = 0.5e-3
#: Formation volume factors, reservoir m^3 per stock-tank m^3.
OIL_FVF = 1.2
WATER_FVF = 1.02
#: Fracture-gradient limit used to cap and flag injection pressure, Pa/m.
FRACTURE_GRADIENT = 0.7 * 6894.757293168 / 0.3048   # 0.7 psi/ft
#: Years over which a suggested rate turns over the well's drainage volume.
DEFAULT_SWEEP_YEARS = 10.0


class ControlMode(str, Enum):
    """How a well is controlled (section 10)."""

    LIQUID_RATE = "liquid_rate"     #: producer: total liquid at surface
    OIL_RATE = "oil_rate"           #: producer: oil at surface
    WATER_RATE = "water_rate"       #: injector: water at surface
    BHP = "bhp"                     #: either: bottom-hole pressure


PRODUCER_MODES = (ControlMode.LIQUID_RATE, ControlMode.OIL_RATE, ControlMode.BHP)
INJECTOR_MODES = (ControlMode.WATER_RATE, ControlMode.BHP)


@dataclass
class WellControl:
    """A well's control mode, target and active period.

    Rates are stored in SI (m^3/s) and pressures in Pa; the display layer
    converts to STB/day and psi.  ``start_day`` and ``end_day`` give the
    schedule of section 12.
    """

    mode: ControlMode = ControlMode.LIQUID_RATE
    #: Target rate in m^3/s, or target BHP in Pa, depending on ``mode``.
    target: float = 0.0
    #: Pressure limit that overrides the rate target when it would be
    #: violated: a minimum BHP for a producer, a maximum for an injector.
    bhp_limit: float | None = None
    start_day: float = 0.0
    end_day: float | None = None
    #: Free-text note recording where an automatic suggestion came from.
    provenance: str = ""

    def __post_init__(self) -> None:
        self.mode = ControlMode(self.mode)
        if self.end_day is not None and self.end_day <= self.start_day:
            raise ConfigError(
                f"well schedule ends at day {self.end_day:g}, at or before its "
                f"start at day {self.start_day:g}")

    def active(self, day: float) -> bool:
        """Whether the well is open at this simulation time."""
        return day >= self.start_day and (self.end_day is None or day <= self.end_day)

    def valid_for(self, role: str) -> bool:
        allowed = PRODUCER_MODES if role == "producer" else INJECTOR_MODES
        return self.mode in allowed

    def describe(self, role: str) -> str:
        if self.mode is ControlMode.BHP:
            target = f"BHP {pa_to_psi(self.target):,.0f} psi"
        else:
            target = f"{self.mode.value} {si_to_stb_per_day(self.target):,.0f} STB/day"
        window = (f"day {self.start_day:g} to "
                  f"{'end' if self.end_day is None else f'day {self.end_day:g}'}")
        limit = ""
        if self.bhp_limit is not None:
            kind = "min" if role == "producer" else "max"
            limit = f", {kind} BHP {pa_to_psi(self.bhp_limit):,.0f} psi"
        return f"{target}{limit}, {window}"


@dataclass
class InflowProperties:
    """What the inflow relation needs, gathered from the model at one well."""

    net_thickness: float          #: m
    permeability: float           #: m^2, thickness-averaged over completions
    drainage_radius: float        #: m
    reservoir_pressure: float     #: Pa
    viscosity: float              #: Pa.s
    formation_volume_factor: float
    skin: float = 0.0

    @property
    def productivity_index(self) -> float:
        r"""``q / dp`` in m^3/s per Pa - the well's deliverability."""
        ratio = max(self.drainage_radius / WELLBORE_RADIUS, 1.001)
        denominator = (self.viscosity * self.formation_volume_factor
                       * (np.log(ratio) - 0.75 + self.skin))
        if denominator <= 0:
            raise ConfigError("the inflow denominator is non-positive; check skin "
                              "and drainage radius")
        return 2.0 * np.pi * self.permeability * self.net_thickness / denominator


def inflow_properties(well, geology, completions, drainage_radius: float,
                      reservoir_pressure: float, skin: float = 0.0
                      ) -> InflowProperties:
    """Gather the inflow inputs from the model at one well."""
    from .completion import resolve_completions

    intervals = resolve_completions(well, geology, completions)
    if not intervals:
        raise ConfigError(
            f"well {well.name} has no open completion, so it has no deliverability. "
            f"Open at least one unit before assigning a rate.")
    net = sum((base - top) * item.ntg for top, base, item in intervals)
    if net <= 0:
        raise ConfigError(
            f"well {well.name} has zero net completed thickness "
            f"(net-to-gross is zero in every unit it is open to)")
    weighted = sum((base - top) * item.ntg * item.permeability
                   for top, base, item in intervals) / net
    injector = well.role == "injector"
    return InflowProperties(
        net_thickness=net, permeability=weighted, drainage_radius=drainage_radius,
        reservoir_pressure=reservoir_pressure,
        viscosity=WATER_VISCOSITY if injector else OIL_VISCOSITY,
        formation_volume_factor=WATER_FVF if injector else OIL_FVF, skin=skin)


def drainage_pore_volume(well, geology, completions, radius: float) -> float:
    """Pore volume in the well's own drainage cylinder, m^3."""
    from .completion import resolve_completions

    intervals = resolve_completions(well, geology, completions)
    net = sum((base - top) * item.ntg for top, base, item in intervals)
    porosity = (sum((base - top) * item.ntg * item.porosity
                    for top, base, item in intervals) / net) if net else 0.0
    return float(np.pi * radius**2 * net * porosity)


def suggest_control(well, geology, completions, wells, reservoir_pressure: float,
                    drawdown: float = DEFAULT_DRAWDOWN, skin: float = 0.0,
                    start_day: float = 0.0, end_day: float | None = None,
                    sweep_years: float = DEFAULT_SWEEP_YEARS) -> WellControl:
    """A defensible starting rate for a newly created well (section 9).

    Two independent limits, and the smaller wins:

    * **deliverability** - what the completion can flow at ``drawdown``;
    * **pattern scale** - the rate that turns over this well's drainage pore
      volume in ``sweep_years``.

    The drainage radius is half the distance to the nearest other well, so
    both limits already reflect the pattern the user has drawn rather than a
    number chosen in advance. An injector's pressure limit is capped at the
    fracture gradient rather than set from the drawdown.
    """
    others = [w for w in wells if w.name != well.name]
    spacing = (min(well.horizontal_distance(w.x, w.y) for w in others)
               if others else 2.0 * drainage_default(geology))
    radius = max(float(spacing) / 2.0, 10.0)

    inflow = inflow_properties(well, geology, completions, radius,
                               reservoir_pressure, skin)
    deliverability = inflow.productivity_index * drawdown
    volume = drainage_pore_volume(well, geology, completions, radius)
    pattern = volume / max(sweep_years * 365.25 * DAY, 1.0)
    rate = min(deliverability, pattern) if pattern > 0 else deliverability
    bound = "pattern scale" if pattern < deliverability else "deliverability"

    injector = well.role == "injector"
    depth = 0.5 * sum(geology.grid.bounds[2])
    if injector:
        limit = min(reservoir_pressure + 2.0 * drawdown, FRACTURE_GRADIENT * depth)
    else:
        limit = max(reservoir_pressure - 4.0 * drawdown, 0.2 * reservoir_pressure)
    return WellControl(
        mode=ControlMode.WATER_RATE if injector else ControlMode.LIQUID_RATE,
        target=float(rate), bhp_limit=float(limit),
        start_day=start_day, end_day=end_day,
        provenance=(
            f"bound by {bound}: deliverability "
            f"{si_to_stb_per_day(deliverability):,.0f} STB/day "
            f"({inflow.net_thickness:.1f} m net, "
            f"{inflow.permeability / 9.869232667160128e-16:,.0f} mD, "
            f"r_e {radius:,.0f} m, {pa_to_psi(drawdown):,.0f} psi drawdown); "
            f"pattern scale {si_to_stb_per_day(pattern):,.0f} STB/day "
            f"({volume / STB:,.0f} STB drainage pore volume over "
            f"{sweep_years:g} years)"),
    )


def drainage_default(geology) -> float:
    """Half the shorter map dimension, for a well with no neighbours."""
    lx, ly, _ = geology.grid.extent
    return 0.25 * min(lx, ly)


def pore_volume(geology, mask=None) -> float:
    """Hydrocarbon-bearing pore volume in m^3 (net, not gross)."""
    grid = geology.grid
    cell = grid.dx * grid.dy * grid.dz
    selection = geology.reservoir_mask if mask is None else mask
    return float(np.sum(geology.porosity[selection] * geology.ntg[selection]) * cell)


def check_rates(wells, controls: dict, geology, reservoir_pressure: float,
                duration_days: float) -> list[str]:
    """Sanity checks on a rate set (section 9).

    Every finding explains the physics it is worried about.  None of them
    blocks the run: an extreme rate is a legitimate experiment, and only the
    user knows whether this one is.
    """
    warnings: list[str] = []
    volume = pore_volume(geology)
    grid = geology.grid

    injection = production = 0.0
    for well in wells:
        control = controls.get(well.name)
        if control is None:
            continue
        if not control.valid_for(well.role):
            warnings.append(
                f"{well.name} is a {well.role} controlled on {control.mode.value}, "
                f"which is not a {well.role} control mode")
        if control.mode is ControlMode.BHP:
            continue
        rate = control.target
        if rate < 0:
            warnings.append(f"{well.name} has a negative rate target")
        if well.role == "injector":
            injection += rate
        else:
            production += rate

        per_year = rate * 365.25 * DAY
        if volume > 0 and per_year > 0.5 * volume:
            warnings.append(
                f"{well.name} at {si_to_stb_per_day(rate):,.0f} STB/day moves "
                f"{100 * per_year / volume:.0f}% of the field's pore volume per "
                f"year on its own; a mechanistic pattern at this scale usually "
                f"sits well under 50%")

    if production > 0 and injection > 0:
        vrr = injection / production
        if not 0.5 <= vrr <= 2.0:
            warnings.append(
                f"voidage replacement ratio is {vrr:.2f} (injection "
                f"{si_to_stb_per_day(injection):,.0f} against production "
                f"{si_to_stb_per_day(production):,.0f} STB/day). Far from 1 the "
                f"field pressure will drift monotonically, and the 4D response "
                f"will be dominated by that drift rather than by the flood")
    elif injection > 0 and production == 0:
        warnings.append(
            "injection with no production: reservoir pressure will rise without "
            "limit, and the pressure signal will swamp the saturation one")

    total = (production + injection) * duration_days * DAY
    if volume > 0 and total > 3.0 * volume:
        warnings.append(
            f"over {duration_days:,.0f} days the wells move {total / volume:.1f} "
            f"pore volumes in total; the flood will have swept out long before "
            f"the end of the run")

    for well in wells:
        control = controls.get(well.name)
        if control is None or control.bhp_limit is None:
            continue
        depth = 0.5 * sum(grid.bounds[2])
        if well.role == "injector" and control.bhp_limit > FRACTURE_GRADIENT * depth:
            warnings.append(
                f"{well.name} may inject up to "
                f"{pa_to_psi(control.bhp_limit):,.0f} psi, above the "
                f"{pa_to_psi(FRACTURE_GRADIENT * depth):,.0f} psi implied by a "
                f"0.7 psi/ft fracture gradient at this depth")
        if well.role == "producer" and control.bhp_limit < 0.1 * reservoir_pressure:
            warnings.append(
                f"{well.name} may draw down to "
                f"{pa_to_psi(control.bhp_limit):,.0f} psi against a reservoir "
                f"pressure of {pa_to_psi(reservoir_pressure):,.0f} psi, which is "
                f"a drawdown no completion would sustain")
    return warnings
