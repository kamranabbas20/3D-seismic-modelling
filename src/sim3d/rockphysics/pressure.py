r"""Pressure substitution (spec section 46).

Pressure is represented separately from saturation throughout the
platform, and it acts on the rock through the *effective* (differential)
stress

.. math:: P_{eff} = P_{conf} - \alpha P_{pore}

with :math:`\alpha` the Biot coefficient (often taken as 1 for the
velocity-effective-stress law, though it is strictly the poroelastic
coefficient and can be smaller in stiff rocks).  Raising pore pressure
lowers effective stress, softens the frame, and lowers Vp - which is the
physical origin of a pressure-up 4D signal.

Four approaches are supported, matching the four in the spec:

``hertz_mindlin``
    Recompute the dry frame at the new effective stress.  Physically
    motivated, gives the characteristic :math:`P^{1/3}` stiffening, and
    tends to over-predict sensitivity for real sands.
``empirical``
    An exponential fit of the form used by Eberhart-Phillips et al. (1989)
    and MacBeth (2004), applied as a multiplier on the dry moduli::

        M(P) = M_inf * (1 - A exp(-P_eff / P_0))

    A common laboratory-calibrated shape.
``table``
    User-supplied ``(P_eff, multiplier)`` curves from core measurements,
    logs or a previous study, interpolated.  This is the option to use when
    the field has its own calibration.
``none``
    No pressure sensitivity at all.  A control case for isolating the
    saturation response.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.errors import ConfigError, ValidationError

PRESSURE_MODELS = ("hertz_mindlin", "empirical", "table", "none")


def effective_stress(confining_pressure, pore_pressure, biot: float = 1.0):
    """``P_eff = P_conf - biot * P_pore``, all in Pa."""
    p_eff = np.asarray(confining_pressure, dtype=float) \
        - float(biot) * np.asarray(pore_pressure, dtype=float)
    return p_eff


def lithostatic_pressure(depth, overburden_density=2300.0, g: float = 9.81):
    """Confining stress from an overburden column, in Pa.

    ``overburden_density`` may be a depth-dependent array of the *average*
    density above each depth, or a single representative value.
    """
    return np.asarray(depth, dtype=float) * np.asarray(overburden_density, dtype=float) * g


def hydrostatic_pressure(depth, fluid_density=1030.0, g: float = 9.81):
    """Normal pore pressure from a connected water column, in Pa."""
    return np.asarray(depth, dtype=float) * float(fluid_density) * g


@dataclass
class PressureModel:
    """How the dry frame responds to a change in effective stress."""

    model: str = "hertz_mindlin"
    biot: float = 1.0
    #: ``empirical``: asymptotic softening amplitude ``A`` and decay ``P_0`` (Pa).
    amplitude: float = 0.30
    decay_pressure: float = 15.0e6
    #: ``table``: effective stress nodes (Pa) and the modulus multipliers at them.
    table_pressure: np.ndarray | None = None
    table_k_multiplier: np.ndarray | None = None
    table_mu_multiplier: np.ndarray | None = None
    #: Effective stress at which the ``empirical`` and ``table`` frames are
    #: defined; multipliers are normalised to 1 here.
    reference_pressure: float = 20.0e6
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.model not in PRESSURE_MODELS:
            raise ConfigError(
                f"unknown pressure model {self.model!r}; choose from {PRESSURE_MODELS}"
            )
        if self.model == "table":
            if self.table_pressure is None or self.table_k_multiplier is None:
                raise ConfigError(
                    "the 'table' pressure model needs table_pressure and "
                    "table_k_multiplier"
                )
            self.table_pressure = np.asarray(self.table_pressure, dtype=float)
            self.table_k_multiplier = np.asarray(self.table_k_multiplier, dtype=float)
            if self.table_mu_multiplier is None:
                self.table_mu_multiplier = self.table_k_multiplier
            self.table_mu_multiplier = np.asarray(self.table_mu_multiplier, dtype=float)
            if not np.all(np.diff(self.table_pressure) > 0):
                raise ConfigError("table_pressure must increase monotonically")

    def multipliers(self, effective_pressure):
        """Modulus multipliers ``(f_K, f_mu)`` at the given effective stress.

        Returns ``(1, 1)`` for models that recompute the frame from scratch
        (``hertz_mindlin``) or that have no pressure response (``none``).
        """
        p = np.asarray(effective_pressure, dtype=float)
        if self.model in ("hertz_mindlin", "none"):
            ones = np.ones_like(p)
            return ones, ones
        if self.model == "empirical":
            def curve(x):
                return 1.0 - self.amplitude * np.exp(-x / self.decay_pressure)
            reference = curve(self.reference_pressure)
            value = curve(np.maximum(p, 0.0)) / reference
            return value, value
        # table
        outside = (np.min(p) < self.table_pressure[0]) or (np.max(p) > self.table_pressure[-1])
        if outside:
            raise ValidationError(
                f"effective stress spans [{np.min(p) / 1e6:.2f}, {np.max(p) / 1e6:.2f}] MPa, "
                f"outside the calibrated table "
                f"[{self.table_pressure[0] / 1e6:.2f}, {self.table_pressure[-1] / 1e6:.2f}] MPa. "
                f"Extend the table rather than extrapolating a laboratory calibration."
            )
        ref_k = np.interp(self.reference_pressure, self.table_pressure, self.table_k_multiplier)
        ref_mu = np.interp(self.reference_pressure, self.table_pressure, self.table_mu_multiplier)
        return (np.interp(p, self.table_pressure, self.table_k_multiplier) / ref_k,
                np.interp(p, self.table_pressure, self.table_mu_multiplier) / ref_mu)

    def frame_pressure(self, effective_pressure):
        """Effective stress to hand the dry-frame model.

        ``hertz_mindlin`` passes the real stress through so the frame is
        rebuilt at it.  The other models build the frame once at the
        reference stress and apply a multiplier afterwards, so they get the
        reference stress here.
        """
        if self.model == "hertz_mindlin":
            return np.asarray(effective_pressure, dtype=float)
        return np.full_like(np.asarray(effective_pressure, dtype=float),
                            self.reference_pressure)

    def describe(self) -> str:
        if self.model == "hertz_mindlin":
            detail = "dry frame recomputed at the local effective stress (P^1/3)"
        elif self.model == "empirical":
            detail = (f"M(P) = M_inf (1 - {self.amplitude:g} exp(-P_eff / "
                      f"{self.decay_pressure / 1e6:g} MPa)), normalised at "
                      f"{self.reference_pressure / 1e6:g} MPa")
        elif self.model == "table":
            detail = (f"user table over "
                      f"[{self.table_pressure[0] / 1e6:g}, "
                      f"{self.table_pressure[-1] / 1e6:g}] MPa")
        else:
            detail = "no pressure sensitivity (control case)"
        return f"pressure model '{self.model}': {detail}, Biot alpha = {self.biot:g}"
