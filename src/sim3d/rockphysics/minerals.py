"""Mineral properties and elastic mixing laws (spec section 41).

Moduli are in Pa and densities in kg/m^3, following the SI-internal rule.
The tabulated values are the standard laboratory numbers compiled in
Mavko, Mukerji & Dvorkin, *The Rock Physics Handbook* (2nd ed., 2009),
Section 9.  Clay is the least well-constrained of them by a wide margin -
published estimates for its moduli span roughly a factor of two - so a
study whose conclusions depend on the clay endpoint should override it
rather than accept the default.

Mixing laws
-----------
Voigt (iso-strain) and Reuss (iso-stress) are the rigorous upper and lower
bounds for an isotropic mixture of unknown geometry; their arithmetic mean
(Voigt-Reuss-Hill) is a common estimate with no rigorous status.  The
Hashin-Shtrikman bounds are the narrowest bounds obtainable without
specifying geometry, and are the ones worth quoting when the separation
matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.errors import ConfigError, ValidationError

GPA = 1.0e9


@dataclass(frozen=True)
class Mineral:
    """A mineral endpoint: bulk modulus, shear modulus, density (SI)."""

    name: str
    k: float      #: bulk modulus, Pa
    mu: float     #: shear modulus, Pa
    rho: float    #: density, kg/m^3

    def __post_init__(self) -> None:
        if self.k <= 0 or self.mu < 0 or self.rho <= 0:
            raise ValidationError(f"mineral {self.name} has unphysical properties")

    @property
    def poisson(self) -> float:
        """Poisson's ratio of the pure mineral."""
        return (3.0 * self.k - 2.0 * self.mu) / (2.0 * (3.0 * self.k + self.mu))


#: Standard mineral endpoints (Mavko et al. 2009, Section 9).
MINERALS: dict[str, Mineral] = {
    "quartz":    Mineral("quartz",    37.0 * GPA, 44.0 * GPA, 2650.0),
    "clay":      Mineral("clay",      21.0 * GPA,  7.0 * GPA, 2580.0),
    "calcite":   Mineral("calcite",   76.8 * GPA, 32.0 * GPA, 2710.0),
    "dolomite":  Mineral("dolomite",  94.9 * GPA, 45.0 * GPA, 2870.0),
    "feldspar":  Mineral("feldspar",  75.6 * GPA, 25.6 * GPA, 2630.0),
    "halite":    Mineral("halite",    24.8 * GPA, 14.9 * GPA, 2160.0),
    "anhydrite": Mineral("anhydrite", 56.1 * GPA, 29.1 * GPA, 2980.0),
    "siderite":  Mineral("siderite", 123.7 * GPA, 51.0 * GPA, 3960.0),
    "pyrite":    Mineral("pyrite",   147.4 * GPA, 132.5 * GPA, 4930.0),
}


def _as_arrays(moduli, fractions):
    m = np.asarray(moduli, dtype=float)
    f = np.asarray(fractions, dtype=float)
    if m.shape[0] != f.shape[0]:
        raise ConfigError(f"{m.shape[0]} moduli but {f.shape[0]} fractions")
    total = np.sum(f, axis=0)
    if np.any(np.abs(total - 1.0) > 1e-6):
        raise ValidationError(
            f"volume fractions must sum to 1; they range over "
            f"[{np.min(total):.6f}, {np.max(total):.6f}]"
        )
    return m, f


def voigt(moduli, fractions):
    """Voigt (iso-strain) average - the rigorous upper bound.

    ``M_V = sum_i f_i M_i``.  ``moduli`` may be scalars or arrays, with the
    mixture index first.
    """
    m, f = _as_arrays(moduli, fractions)
    return np.sum(f * m, axis=0)


def reuss(moduli, fractions):
    """Reuss (iso-stress) average - the rigorous lower bound.

    ``1 / M_R = sum_i f_i / M_i``.  A zero-modulus phase (a gas-filled pore
    space treated as a mixture component, or a zero-shear fluid) drives the
    average to zero, which is the physically correct answer for a
    suspension.
    """
    m, f = _as_arrays(moduli, fractions)
    if np.any(m <= 0):
        return np.zeros_like(np.sum(f * m, axis=0))
    return 1.0 / np.sum(f / m, axis=0)


def vrh(moduli, fractions):
    """Voigt-Reuss-Hill average: the arithmetic mean of the two bounds.

    Widely used as a mineral-mixture estimate.  It is a convenience, not a
    bound, and has no rigorous justification.
    """
    return 0.5 * (voigt(moduli, fractions) + reuss(moduli, fractions))


def hashin_shtrikman(k_moduli, mu_moduli, fractions, bound: str = "upper"):
    """Hashin-Shtrikman bounds for an isotropic n-phase mixture.

    Uses the Berryman (1995) form::

        K_HS = Lambda(mu*) ,  Lambda(z) = <1/(K + 4z/3)>^-1 - 4z/3
        mu_HS = Gamma(zeta*),  Gamma(z) = <1/(mu + z)>^-1 - z
        zeta(K, mu) = (mu/6)(9K + 8mu)/(K + 2mu)

    with ``mu*`` the largest (upper bound) or smallest (lower bound) shear
    modulus in the mixture, and ``zeta*`` evaluated at the corresponding
    phase.  These are the narrowest bounds achievable without specifying
    the geometry of the mixture.

    Returns
    -------
    tuple
        ``(K_HS, mu_HS)`` in Pa.
    """
    if bound not in ("upper", "lower"):
        raise ConfigError(f"bound must be 'upper' or 'lower', got {bound!r}")
    k, f = _as_arrays(k_moduli, fractions)
    mu, _ = _as_arrays(mu_moduli, fractions)

    pick = np.argmax if bound == "upper" else np.argmin
    idx = int(pick(np.mean(np.atleast_1d(mu).reshape(len(mu), -1), axis=1)))
    mu_star = mu[idx]
    k_star = k[idx]

    k_hs = 1.0 / np.sum(f / (k + 4.0 * mu_star / 3.0), axis=0) - 4.0 * mu_star / 3.0
    zeta = (mu_star / 6.0) * (9.0 * k_star + 8.0 * mu_star) / (k_star + 2.0 * mu_star)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_hs = np.where(
            mu_star > 0.0,
            1.0 / np.sum(f / (mu + zeta), axis=0) - zeta,
            0.0,
        )
    return k_hs, mu_hs


def mixed_mineral(composition: dict[str, float] | dict[str, np.ndarray],
                  model: str = "vrh", minerals: dict[str, Mineral] | None = None):
    """Effective solid-phase properties for a mineral composition.

    Parameters
    ----------
    composition:
        Mineral name -> volume fraction of the *solid* phase (excluding
        pore space).  Fractions must sum to 1.  Values may be arrays, so a
        whole 3D volume can be mixed at once.
    model:
        ``"voigt"``, ``"reuss"``, ``"vrh"``, ``"hs_upper"`` or ``"hs_lower"``.

    Returns
    -------
    tuple
        ``(K_mineral, mu_mineral, rho_mineral)`` in SI.
    """
    table = minerals or MINERALS
    unknown = set(composition) - set(table)
    if unknown:
        raise ConfigError(
            f"unknown mineral(s) {sorted(unknown)}; known: {sorted(table)}. "
            f"Add a Mineral entry rather than approximating with another phase."
        )
    names = list(composition)
    fractions = [np.asarray(composition[n], dtype=float) for n in names]
    k = [np.full_like(fractions[0], table[n].k, dtype=float) for n in names]
    mu = [np.full_like(fractions[0], table[n].mu, dtype=float) for n in names]
    rho = [np.full_like(fractions[0], table[n].rho, dtype=float) for n in names]

    density = voigt(rho, fractions)  # density always mixes linearly by volume
    if model == "voigt":
        return voigt(k, fractions), voigt(mu, fractions), density
    if model == "reuss":
        return reuss(k, fractions), reuss(mu, fractions), density
    if model == "vrh":
        return vrh(k, fractions), vrh(mu, fractions), density
    if model in ("hs_upper", "hs_lower"):
        k_hs, mu_hs = hashin_shtrikman(k, mu, fractions,
                                       "upper" if model == "hs_upper" else "lower")
        return k_hs, mu_hs, density
    raise ConfigError(
        f"unknown mineral mixing model {model!r}; choose from "
        f"voigt, reuss, vrh, hs_upper, hs_lower"
    )
