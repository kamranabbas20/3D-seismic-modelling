r"""Two-phase relative permeability and fractional flow.

Corey curves, which is the standard minimum for a waterflood:

.. math::
    S^* = \frac{S_w - S_{wc}}{1 - S_{wc} - S_{or}},\qquad
    k_{rw} = k_{rw}^{max}\,(S^*)^{n_w},\qquad
    k_{ro} = k_{ro}^{max}\,(1-S^*)^{n_o}

The exponents are what set the shape of the flood front: a low water
exponent gives an early, smeared breakthrough, a high one a sharp piston.
They are the most consequential unmeasured parameters in a mechanistic
waterflood study, which is why they are explicit settings rather than
constants inside the solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.errors import ConfigError


@dataclass(frozen=True)
class CoreyRelativePermeability:
    """Corey curves with irreducible water and residual oil."""

    swc: float = 0.20          #: irreducible (connate) water saturation
    sor: float = 0.25          #: residual oil saturation
    krw_max: float = 0.35      #: water end point, at ``1 - sor``
    kro_max: float = 0.90      #: oil end point, at ``swc``
    nw: float = 2.5            #: water exponent
    no: float = 2.0            #: oil exponent
    water_viscosity: float = 0.5e-3   #: Pa.s
    oil_viscosity: float = 1.0e-3     #: Pa.s

    def __post_init__(self) -> None:
        if not 0.0 <= self.swc < 1.0 - self.sor:
            raise ConfigError(
                f"connate water {self.swc} and residual oil {self.sor} leave no "
                f"movable saturation")
        for name in ("krw_max", "kro_max", "nw", "no",
                     "water_viscosity", "oil_viscosity"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive, got {getattr(self, name)}")

    def normalised(self, sw):
        """Movable water saturation, clipped to ``[0, 1]``."""
        span = 1.0 - self.swc - self.sor
        return np.clip((np.asarray(sw, dtype=float) - self.swc) / span, 0.0, 1.0)

    def krw(self, sw):
        return self.krw_max * self.normalised(sw) ** self.nw

    def kro(self, sw):
        return self.kro_max * (1.0 - self.normalised(sw)) ** self.no

    def mobilities(self, sw):
        """``(lambda_w, lambda_o)`` in 1/(Pa.s)."""
        return self.krw(sw) / self.water_viscosity, self.kro(sw) / self.oil_viscosity

    def total_mobility(self, sw):
        water, oil = self.mobilities(sw)
        return water + oil

    def fractional_flow(self, sw):
        """Water cut in the flowing stream, ``lambda_w / lambda_t``."""
        water, oil = self.mobilities(sw)
        total = water + oil
        return np.where(total > 0.0, water / np.where(total > 0.0, total, 1.0), 0.0)

    def describe(self) -> str:
        return (f"Corey: Swc {self.swc:g}, Sor {self.sor:g}, "
                f"krw_max {self.krw_max:g} (nw {self.nw:g}), "
                f"kro_max {self.kro_max:g} (no {self.no:g}), "
                f"mu_w {self.water_viscosity * 1e3:g} cP, "
                f"mu_o {self.oil_viscosity * 1e3:g} cP")
