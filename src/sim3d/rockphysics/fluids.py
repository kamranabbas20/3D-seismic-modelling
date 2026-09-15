"""Fluid properties and fluid mixing (spec sections 43-44).

Brine, oil and gas properties follow Batzle, M. & Wang, Z. (1992),
*Seismic properties of pore fluids*, Geophysics 57(11), 1396-1408.  The
correlations are empirical fits to laboratory data and are stated by their
authors in these units:

* pressure ``P`` in MPa
* temperature ``T`` in degrees Celsius
* density in g/cm^3
* velocity in m/s

so the implementations below convert into those units, evaluate the
published expression verbatim, and convert the result back to SI.  Doing
the arithmetic in the authors' units is the only way to keep the printed
coefficients checkable against the paper.

Validity limits
---------------
Batzle-Wang was fitted over roughly 5-100 MPa and 20-350 degC, with
salinity up to about 320,000 ppm.  Outside those ranges the expressions
still evaluate but the answers are extrapolations, and
:func:`check_validity` says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.errors import ConfigError, ValidationError

MPA = 1.0e6
GCC = 1.0e3
R_GAS = 8.31441  # J / (mol K), as used by Batzle & Wang

#: Fitted range of the Batzle-Wang correlations (P in Pa, T in degC, S in ppm).
VALIDITY = {
    "pressure": (5.0e6, 1.0e8),
    "temperature": (10.0, 350.0),
    "salinity": (0.0, 320000.0),
}


@dataclass(frozen=True)
class FluidState:
    """Properties of a single fluid phase, in SI."""

    name: str
    k: float | np.ndarray       #: bulk modulus, Pa
    rho: float | np.ndarray     #: density, kg/m^3

    @property
    def velocity(self):
        """Acoustic velocity ``sqrt(K/rho)`` in m/s."""
        return np.sqrt(self.k / self.rho)


def check_validity(pressure, temperature, salinity=0.0) -> list[str]:
    """Return warnings for inputs outside the fitted range of Batzle-Wang."""
    notes = []
    for label, value in (("pressure", pressure), ("temperature", temperature),
                         ("salinity", salinity)):
        lo, hi = VALIDITY[label]
        v = np.asarray(value, dtype=float)
        if np.any(v < lo) or np.any(v > hi):
            unit = {"pressure": "Pa", "temperature": "degC", "salinity": "ppm"}[label]
            notes.append(
                f"{label} spans [{np.min(v):g}, {np.max(v):g}] {unit}, outside the "
                f"Batzle-Wang fitted range [{lo:g}, {hi:g}] {unit}; results are "
                f"extrapolations"
            )
    return notes


# --- brine ----------------------------------------------------------------
#: Batzle & Wang (1992) Table 1: coefficients w_ij of the pure-water velocity.
_W = np.array([
    [1402.85,    1.524,     3.437e-3,  -1.197e-5],
    [4.871,     -0.0111,    1.739e-4,  -1.628e-6],
    [-0.04783,   2.747e-4, -2.135e-6,   1.237e-8],
    [1.487e-4,  -6.503e-7, -1.455e-8,   1.327e-10],
    [-2.197e-7,  7.987e-10, 5.230e-11, -4.614e-13],
])


def brine_properties(pressure, temperature, salinity=35000.0) -> FluidState:
    """Brine density and bulk modulus (Batzle & Wang eqs 27a, 27b, 28, 29).

    Parameters
    ----------
    pressure:
        Pore pressure in Pa.
    temperature:
        Temperature in degrees Celsius.
    salinity:
        Salinity in ppm (parts per million by weight); 35,000 ppm is
        seawater.
    """
    p = np.asarray(pressure, dtype=float) / MPA
    t = np.asarray(temperature, dtype=float)
    s = np.asarray(salinity, dtype=float) * 1.0e-6  # ppm -> weight fraction
    if np.any(s < 0) or np.any(s > 1):
        raise ValidationError(f"salinity must be within 0-1e6 ppm, got {salinity}")

    # Pure water density, eq 27a (g/cm^3).
    rho_w = 1.0 + 1e-6 * (
        -80.0 * t - 3.3 * t**2 + 0.00175 * t**3 + 489.0 * p - 2.0 * t * p
        + 0.016 * t**2 * p - 1.3e-5 * t**3 * p - 0.333 * p**2 - 0.002 * t * p**2
    )
    # Brine density, eq 27b.
    rho_b = rho_w + s * (
        0.668 + 0.44 * s
        + 1e-6 * (300.0 * p - 2400.0 * p * s
                  + t * (80.0 + 3.0 * t - 3300.0 * s - 13.0 * p + 47.0 * p * s))
    )

    # Pure water velocity, eq 28 (m/s).
    v_w = np.zeros_like(np.asarray(t + p, dtype=float))
    for i in range(_W.shape[0]):
        for j in range(_W.shape[1]):
            v_w = v_w + _W[i, j] * t**i * p**j
    # Brine velocity, eq 29.
    v_b = v_w + s * (
        1170.0 - 9.6 * t + 0.055 * t**2 - 8.5e-5 * t**3
        + 2.6 * p - 0.0029 * t * p - 0.0476 * p**2
    ) + s**1.5 * (780.0 - 10.0 * p + 0.16 * p**2) - 1820.0 * s**2

    rho_si = rho_b * GCC
    return FluidState("brine", k=rho_si * v_b**2, rho=rho_si)


# --- gas ------------------------------------------------------------------
def gas_properties(pressure, temperature, gravity=0.65) -> FluidState:
    """Hydrocarbon-gas density and adiabatic bulk modulus (Batzle & Wang eqs 9-11).

    ``gravity`` is the gas gravity (molar mass relative to air); 0.56 is
    close to pure methane, 0.65-0.8 covers most reservoir gases.
    """
    p = np.asarray(pressure, dtype=float) / MPA
    t = np.asarray(temperature, dtype=float)
    g = np.asarray(gravity, dtype=float)
    if np.any(g <= 0):
        raise ValidationError(f"gas gravity must be positive, got {gravity}")

    t_abs = t + 273.15
    p_pr = p / (4.892 - 0.4048 * g)              # pseudo-reduced pressure
    t_pr = t_abs / (94.72 + 170.75 * g)          # pseudo-reduced temperature

    e_exp = -(0.45 + 8.0 * (0.56 - 1.0 / t_pr) ** 2) * p_pr**1.2 / t_pr
    e = 0.109 * (3.85 - t_pr) ** 2 * np.exp(e_exp)
    z = (0.03 + 0.00527 * (3.5 - t_pr) ** 3) * p_pr + 0.642 * t_pr \
        - 0.007 * t_pr**4 - 0.52 + e

    rho = 28.8 * g * p * MPA / (z * R_GAS * t_abs * 1000.0)  # kg/m^3

    gamma0 = (0.85 + 5.6 / (p_pr + 2.0) + 27.1 / (p_pr + 3.5) ** 2
              - 8.7 * np.exp(-0.65 * (p_pr + 1.0)))
    # dE/dP_pr, needed for the adiabatic modulus.
    de = e * 1.2 * p_pr**0.2 * (-(0.45 + 8.0 * (0.56 - 1.0 / t_pr) ** 2) / t_pr)
    dz = (0.03 + 0.00527 * (3.5 - t_pr) ** 3) + de
    k = p * MPA * gamma0 / (1.0 - p_pr / z * dz)
    return FluidState("gas", k=k, rho=rho)


# --- oil ------------------------------------------------------------------
def oil_properties(pressure, temperature, api=30.0, gas_gravity=0.65,
                   gor=0.0) -> FluidState:
    """Oil density and bulk modulus (Batzle & Wang eqs 18-21).

    Parameters
    ----------
    api:
        API gravity in degrees; ``rho_0 = 141.5 / (API + 131.5)`` g/cm^3 at
        15.6 degC and atmospheric pressure.
    gas_gravity:
        Gravity of the dissolved gas.
    gor:
        Gas-oil ratio in litres of gas per litre of oil (equivalently
        m^3/m^3).  ``gor = 0`` gives dead oil.
    """
    p = np.asarray(pressure, dtype=float) / MPA
    t = np.asarray(temperature, dtype=float)
    api = np.asarray(api, dtype=float)
    rg = np.asarray(gor, dtype=float)
    g = np.asarray(gas_gravity, dtype=float)
    if np.any(api <= -131.5):
        raise ValidationError(f"API gravity out of range, got {api}")
    if np.any(rg < 0):
        raise ValidationError(f"GOR must be non-negative, got {gor}")

    rho0 = 141.5 / (api + 131.5)  # reference oil density, g/cm^3

    # Volume factor for the dissolved gas, eq 23.
    b0 = 0.972 + 0.00038 * (2.4 * rg * np.sqrt(g / rho0) + t + 17.8) ** 1.175
    # Saturated (live) oil density at reference conditions, eq 24.
    rho_g = (rho0 + 0.0012 * g * rg) / b0
    # Pseudo-density used by the velocity expression, eq 22.
    rho_pseudo = rho0 / (b0 * (1.0 + 0.001 * rg))

    # Pressure and temperature correction, eqs 18-19, applied to the live density.
    rho_p = (rho_g + (0.00277 * p - 1.71e-7 * p**3) * (rho_g - 1.15) ** 2
             + 3.49e-4 * p)
    rho = rho_p / (0.972 + 3.81e-4 * (t + 17.78) ** 1.175)

    # Velocity, eq 20b, evaluated at the pseudo-density.
    v = (2096.0 * np.sqrt(rho_pseudo / (2.6 - rho_pseudo)) - 3.7 * t + 4.64 * p
         + 0.0115 * (4.12 * np.sqrt(np.maximum(1.08 / rho_pseudo - 1.0, 0.0)) - 1.0) * t * p)

    rho_si = rho * GCC
    return FluidState("oil", k=rho_si * v**2, rho=rho_si)


# --- mixing ---------------------------------------------------------------
def mix_fluids(phases: dict[str, FluidState], saturations: dict[str, np.ndarray],
               model: str = "wood", brie_exponent: float = 3.0,
               tolerance: float = 1e-6) -> FluidState:
    """Effective properties of a multiphase pore fluid.

    Parameters
    ----------
    model:
        ``"wood"`` (equivalently Reuss) assumes the phases are mixed at a
        scale far below the seismic wavelength, so pore pressure equalises
        and the moduli combine harmonically::

            1/K_fl = sum_i S_i / K_i

        This is the standard choice and is what makes a few percent of gas
        collapse the fluid modulus.

        ``"brie"`` is the empirical patchy-saturation interpolation of
        Brie et al. (1995)::

            K_fl = (K_liquid - K_gas)(1 - S_g)^e + K_gas

        with ``e = 1`` giving a linear (fully patchy) mix and large ``e``
        approaching Wood.  Patchiness raises the modulus at low gas
        saturation, so choosing between them materially changes the
        predicted 4D gas response - which is why it is an explicit setting
        and not a default buried in the code.

    Density always mixes linearly by saturation, in every model.
    """
    if set(phases) != set(saturations):
        raise ConfigError(
            f"phases {sorted(phases)} and saturations {sorted(saturations)} must match"
        )
    sats = {n: np.asarray(s, dtype=float) for n, s in saturations.items()}
    total = sum(sats.values())
    if np.any(np.abs(total - 1.0) > tolerance):
        raise ValidationError(
            f"saturations must sum to 1 within {tolerance}; they range over "
            f"[{np.min(total):.8f}, {np.max(total):.8f}]"
        )

    rho = sum(sats[n] * phases[n].rho for n in phases)

    if model == "wood":
        k = 1.0 / sum(sats[n] / phases[n].k for n in phases)
    elif model == "brie":
        if "gas" not in phases:
            raise ConfigError("the Brie mixing law needs a phase named 'gas'")
        liquid = {n: s for n, s in sats.items() if n != "gas"}
        s_liquid = sum(liquid.values())
        with np.errstate(divide="ignore", invalid="ignore"):
            k_liquid = np.where(
                s_liquid > 0.0,
                1.0 / sum(np.where(s_liquid > 0.0, liquid[n] / s_liquid, 0.0)
                          / phases[n].k for n in liquid),
                phases["gas"].k,
            )
        k = (k_liquid - phases["gas"].k) * (1.0 - sats["gas"]) ** brie_exponent \
            + phases["gas"].k
    else:
        raise ConfigError(
            f"unknown fluid mixing model {model!r}; choose 'wood' or 'brie'"
        )
    return FluidState("mixed", k=k, rho=rho)
