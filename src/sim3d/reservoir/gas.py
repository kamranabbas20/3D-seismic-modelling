r"""Solution gas liberation without gas flow.

What this is
------------
A local flash that lets gas come *out of solution* when a cell falls below
its bubble point, and go back in when the pressure recovers.  It turns the
gas saturation from something the user imposes into something the
simulation produces, which is the one thing the two-phase solver could
never do however far the pressure fell.

The state it carries is, per cell, the stock-tank oil ``N`` and the total
gas ``G`` (dissolved plus free), both in standard cubic metres.  Both are
conserved scalars, transported between cells on the oil flux.  Given the
pressure, the split is arithmetic rather than iteration:

.. math::
    R_s = \min\!\big(G/N,\; R_s^{sat}(p)\big), \qquad
    S_g = \frac{(G - N R_s)\,B_g(p)}{V_p}

The oil holds what it can hold; whatever is left over is free gas, occupying
the volume that gas occupies at reservoir conditions.  Above the bubble
point :math:`G/N \le R_s^{sat}` and the second term is exactly zero, so no
gas can appear where none should.

Tracking ``N`` rather than deriving it from :math:`S_o` is the load-bearing
choice.  Deriving it means the oil's own expansion as pressure falls has
nowhere to go in a fixed pore volume, and the flash reads that surplus as
gas - which puts free gas *above* the bubble point, where there can be none.

Because the flash is instantaneous rather than time-integrated, it needs no
timestep limit of its own: a step that drops a cell far below its bubble
point liberates exactly the gas that pressure implies, in one go, which is
what thermodynamics says happens.

Volume closure
--------------
:math:`N B_o + V_p S_g` is the room the hydrocarbons want; :math:`V_p(1-S_w)`
is the room they have.  A black-oil simulator makes those equal by solving
for pressure; this one cannot, because the pressure equation it inherits
carries a single lumped compressibility and knows nothing about
:math:`B_o`.  The discrepancy is therefore real, and the simulator measures
and reports it rather than absorbing it quietly - see
``FlowResult.volume_closure_error``.

What this is not
----------------
The liberated gas does not flow.  It appears where the oil was, stays
there, and is not produced.  That is a good approximation while
:math:`S_g` is below the critical gas saturation - which is where gas
genuinely is immobile, and which is the regime a depleting producer spends
its first years in - and it is wrong afterwards, in three ways worth being
explicit about:

* no gas cap forms, because gas cannot segregate upwards;
* the well's gas-oil ratio is capped at :math:`R_s(p_{wf})`, so a
  solution-gas drive produces too little gas and too much oil;
* the pressure solve still carries a single lumped total compressibility,
  so it does not feel gas expansion supporting the reservoir, and the
  decline is steeper than a real solution-gas drive.

The simulator therefore reports the largest :math:`S_g` it reached and
warns when it passes :func:`SolutionGas.critical_saturation`, which is the
point at which those three become the story rather than a detail.  Past
that, a black-oil simulator is the instrument, not this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.errors import ConfigError
from ..core.units import PSI
from ..rockphysics.fluids import bubble_point, gas_fvf, oil_fvf, solution_gor

#: Pressure floor for the PVT correlations, Pa.  A cell at exactly zero
#: pressure has an infinite gas volume factor; this keeps the arithmetic
#: finite while staying far below any pressure a reservoir reaches.
MIN_PRESSURE = 1.0e4

#: Fixed-point sweeps for the undersaturated ``Rs``.  ``Bo`` moves only a
#: few percent across the whole ``Rs`` range, so this converges immediately.
_UNDERSATURATED_ITERATIONS = 3


@dataclass(frozen=True)
class SolutionGas:
    """PVT and the flash for a dissolved-gas model with immobile free gas."""

    api: float = 30.0                 #: stock-tank oil gravity, degrees API
    gas_gravity: float = 0.65         #: gas molar mass relative to air
    initial_gor: float = 15.0         #: solution GOR at initial conditions, m^3/m^3
    temperature: float = 80.0         #: reservoir temperature, degrees Celsius
    #: Gas saturation at which free gas starts to move.  This model cannot
    #: move it, so this is the saturation at which the model stops being
    #: trustworthy rather than a flow parameter.
    critical_saturation: float = 0.02
    oil_compressibility: float = 1.5e-9   #: undersaturated oil, 1/Pa
    #: Gas end point and exponent for the Corey curve used at the well.
    krg_max: float = 0.6
    ng: float = 2.0
    #: Gas viscosity, Pa.s.  0.02 cP is about right for a 0.65-gravity gas at
    #: a few hundred psi and 80 degC; it is a constant rather than a
    #: correlation because it appears in one place, the well inflow, and
    #: varies by less across a depleting reservoir than the relative
    #: permeability it multiplies.
    gas_viscosity: float = 0.02e-3

    def __post_init__(self) -> None:
        if self.initial_gor < 0:
            raise ConfigError(f"initial GOR must be non-negative, got {self.initial_gor}")
        if not 0.0 < self.critical_saturation < 1.0:
            raise ConfigError(
                f"critical gas saturation must lie in (0, 1), "
                f"got {self.critical_saturation}")
        for name in ("gas_gravity", "oil_compressibility", "krg_max", "ng",
                     "gas_viscosity"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive, got {getattr(self, name)}")

    # --------------------------------------------------------- mobility
    def krg(self, sg):
        r"""Gas relative permeability, a Corey curve above the critical saturation.

        Used at the well and nowhere else.  Free gas between cells is held
        in place, which is a fair approximation in the body of the reservoir
        and a poor one in the block a producer is completed in: that is
        where the gas saturation is highest and the potential gradient
        steepest, so it is the one place immobility is indefensible.
        Leaving it there instead traps every bubble the well block ever
        liberates and drives its saturation to the residual-oil endpoint,
        which is a grid artefact rather than a reservoir.
        """
        sg = np.asarray(sg, dtype=float)
        mobile = np.clip((sg - self.critical_saturation)
                         / max(1.0 - self.critical_saturation, 1e-12), 0.0, 1.0)
        return self.krg_max * mobile ** self.ng

    def mobility(self, sg):
        """Gas mobility, 1/(Pa.s)."""
        return self.krg(sg) / self.gas_viscosity

    # ------------------------------------------------------------------ PVT
    @property
    def bubble_point(self) -> float:
        """Bubble-point pressure of the original oil, Pa."""
        return float(bubble_point(self.api, self.gas_gravity,
                                  self.initial_gor, self.temperature))

    def rs_saturated(self, pressure):
        """Gas the oil can hold at ``pressure``, m^3/m^3, capped at the original."""
        rs = solution_gor(np.maximum(pressure, MIN_PRESSURE), self.api,
                          self.gas_gravity, self.temperature)
        return np.minimum(rs, self.initial_gor)

    def bo(self, pressure, rs):
        """Oil formation volume factor, reservoir m^3 per stock-tank m^3."""
        return oil_fvf(np.maximum(pressure, MIN_PRESSURE), rs, self.api,
                       self.gas_gravity, self.temperature, self.oil_compressibility)

    def bg(self, pressure):
        """Gas formation volume factor, reservoir m^3 per standard m^3."""
        return gas_fvf(np.maximum(pressure, MIN_PRESSURE), self.temperature,
                       self.gas_gravity)

    # ---------------------------------------------------------------- flash
    def initial_moles(self, pressure, sw, sg, pore_volume):
        """Stock-tank oil and total gas in each cell initially, standard m^3.

        Oil carries ``initial_gor`` unless the cell starts below its bubble
        point, in which case it carries what it can hold.  Any free gas the
        baseline put there - a gas cap above a gas-oil contact - is counted
        into ``G`` as well, so it is liberated gas from the first step
        rather than a separate species.
        """
        pressure = np.asarray(pressure, dtype=float)
        sw = np.asarray(sw, dtype=float)
        sg = np.asarray(sg, dtype=float)
        so = np.clip(1.0 - sw - sg, 0.0, 1.0)
        rs = self.rs_saturated(pressure)
        oil_std = pore_volume * so / self.bo(pressure, rs)
        gas_std = oil_std * rs + pore_volume * sg / self.bg(pressure)
        return oil_std, gas_std

    def flash(self, pressure, oil_std, gas_std, pore_volume, sw):
        """Split each cell's gas into dissolved and free at ``pressure``.

        Returns ``(sg, rs, over_full)``.  ``over_full`` flags cells where the
        free gas asked for more room than the pore space had left; the
        saturation is clipped there and the caller is expected to notice,
        because a cell in that state has left this model's range entirely.
        """
        pressure = np.maximum(np.asarray(pressure, dtype=float), MIN_PRESSURE)
        oil_std = np.asarray(oil_std, dtype=float)
        gas_std = np.maximum(np.asarray(gas_std, dtype=float), 0.0)
        room = np.clip(1.0 - np.asarray(sw, dtype=float), 0.0, 1.0)

        # A cell with no oil left cannot hold gas in solution, so all of its
        # gas is free.  Dividing by its zero oil would say the same thing
        # much less politely.
        has_oil = oil_std > 0.0
        rs_have = np.divide(gas_std, oil_std, out=np.full_like(gas_std, np.inf),
                            where=has_oil)
        rs = np.minimum(rs_have, self.rs_saturated(pressure))
        rs = np.where(has_oil, rs, 0.0)

        free = np.maximum(gas_std - oil_std * rs, 0.0)
        sg = free * self.bg(pressure) / pore_volume
        over_full = sg > room
        return np.minimum(sg, room), rs, over_full

    def total_compressibility(self, pressure, sw, sg, rs, base):
        r"""Total compressibility of a cell, 1/Pa, with the gas terms included.

        Below the bubble point the compressibility of the hydrocarbons is
        not a constant and is not small.  Two effects, both large:

        .. math::
            c_o = -\frac{1}{B_o}\frac{\partial B_o}{\partial p}
                  + \frac{B_g}{B_o}\frac{\partial R_s}{\partial p},
            \qquad
            c_g = -\frac{1}{B_g}\frac{\partial B_g}{\partial p}

        The first is oil releasing gas that expands as the pressure falls;
        the second is that gas continuing to expand once it is free.  At a
        few hundred psi both are of order 1e-7 /Pa against the 4e-10 /Pa a
        dead-oil reservoir carries - two and a half orders of magnitude -
        and they are the whole reason a solution-gas drive declines slowly
        instead of collapsing to the bottom-hole pressure.

        Leaving them out is not a small error on the gas saturation, it is
        the error: the pressure falls too far, so too much gas comes out.

        The derivatives are taken numerically on this module's own
        correlations rather than from a second set of fitted expressions, so
        the compressibility and the flash cannot disagree about the same
        oil.  ``base`` is used unchanged where the cell is still
        undersaturated, which is where it is the right number.
        """
        p = np.maximum(np.asarray(pressure, dtype=float), MIN_PRESSURE)
        sg = np.asarray(sg, dtype=float)
        so = np.clip(1.0 - np.asarray(sw, dtype=float) - sg, 0.0, 1.0)
        saturated = sg > 0.0
        dp = np.maximum(1.0e-3 * p, 1.0e4)

        def bo_at(q):
            # Saturated cells ride the correlation; undersaturated ones keep
            # the gas they have and merely compress.
            return self.bo(q, np.where(saturated, self.rs_saturated(q), rs))

        bo = bo_at(p)
        dbo = (bo_at(p + dp) - bo_at(p - dp)) / (2.0 * dp)
        bg = self.bg(p)
        dbg = (self.bg(p + dp) - self.bg(p - dp)) / (2.0 * dp)
        drs = (self.rs_saturated(p + dp) - self.rs_saturated(p - dp)) / (2.0 * dp)

        c_oil = -dbo / bo + (bg / bo) * drs
        c_gas = -dbg / bg
        total = base + so * c_oil + sg * c_gas
        return np.where(saturated, np.maximum(total, base), base)

    def volume_error(self, pressure, oil_std, sg, pore_volume, sw, rs):
        """Fraction of pore volume the hydrocarbons over- or under-fill.

        The honest measure of what the lumped-compressibility pressure
        equation is costing: how far ``N B_o + V_p S_g`` is from the room
        the water leaves.
        """
        wanted = oil_std * self.bo(pressure, rs) + pore_volume * sg
        have = pore_volume * np.clip(1.0 - np.asarray(sw, dtype=float), 0.0, 1.0)
        return np.abs(wanted - have) / pore_volume

    def describe(self) -> str:
        return (f"solution gas: {self.initial_gor:g} m3/m3 initial GOR, "
                f"Pb {self.bubble_point / PSI:,.0f} psi, API {self.api:g}, "
                f"gas gravity {self.gas_gravity:g}, T {self.temperature:g} degC, "
                f"critical Sg {self.critical_saturation:g} (free gas immobile)")
