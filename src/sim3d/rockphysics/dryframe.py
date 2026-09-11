r"""Dry-rock frame models (spec section 42).

All of these predict the *dry* frame moduli, which Gassmann then saturates.
Keeping the frame separate from the fluid is what allows pressure and
saturation to be varied independently, which is the whole point of the
4D decomposition in spec sections 48-55.

Hertz-Mindlin
-------------
Mindlin (1949); see Mavko et al. (2009) Section 5.2.  For a random dense
pack of identical spheres at critical porosity :math:`\phi_c` with
coordination number :math:`n` under effective pressure :math:`P`:

.. math::
    K_{HM} = \left[\frac{n^2(1-\phi_c)^2\mu^2}{18\pi^2(1-\nu)^2}P\right]^{1/3}

.. math::
    \mu_{HM} = \frac{5-4\nu}{5(2-\nu)}
        \left[\frac{3n^2(1-\phi_c)^2\mu^2}{2\pi^2(1-\nu)^2}P\right]^{1/3}

The :math:`P^{1/3}` dependence is the mechanism behind the pressure
sensitivity of the frame, and therefore behind the pressure half of a 4D
signal.  It assumes perfect adhesion at the grain contacts (no slip); real
sands are often softer and less pressure-sensitive than it predicts, which
is what the user-calibrated option in :mod:`sim3d.rockphysics.pressure`
exists for.

Soft- and stiff-sand
--------------------
Dvorkin & Nur (1996).  Both interpolate between the Hertz-Mindlin point at
:math:`\phi_c` and the mineral point at :math:`\phi = 0`: the soft-sand
model along the modified Hashin-Shtrikman *lower* bound (uncemented,
compliant), the stiff-sand model along the *upper* bound (cemented, stiff).
"""

from __future__ import annotations

import numpy as np

from ..core.errors import ConfigError, ValidationError

DEFAULT_COORDINATION = 9.0
DEFAULT_CRITICAL_POROSITY = 0.40


def _poisson(k, mu):
    return (3.0 * k - 2.0 * mu) / (2.0 * (3.0 * k + mu))


def hertz_mindlin(k_mineral, mu_mineral, effective_pressure,
                  critical_porosity: float = DEFAULT_CRITICAL_POROSITY,
                  coordination: float = DEFAULT_COORDINATION):
    """Dry moduli of a sphere pack at critical porosity.

    Parameters
    ----------
    k_mineral, mu_mineral:
        Solid-phase moduli in Pa.
    effective_pressure:
        Effective (differential) pressure in Pa.  Must be positive: a
        sphere pack under zero confining stress has no contact stiffness at
        all, so the model degenerates rather than approaching a small
        number.
    """
    p = np.asarray(effective_pressure, dtype=float)
    if np.any(p <= 0):
        raise ValidationError(
            "Hertz-Mindlin needs a strictly positive effective pressure; "
            f"the minimum here is {np.min(p):g} Pa. A non-positive effective "
            f"stress means the frame has no grain-contact stiffness, which is "
            f"outside the model's validity, not a small number."
        )
    if not 0 < critical_porosity < 1:
        raise ConfigError(f"critical porosity must be in (0, 1), got {critical_porosity}")

    nu = _poisson(k_mineral, mu_mineral)
    c = coordination
    phi_c = critical_porosity
    k_hm = np.cbrt(c**2 * (1.0 - phi_c) ** 2 * mu_mineral**2 * p
                   / (18.0 * np.pi**2 * (1.0 - nu) ** 2))
    mu_hm = ((5.0 - 4.0 * nu) / (5.0 * (2.0 - nu))) * np.cbrt(
        3.0 * c**2 * (1.0 - phi_c) ** 2 * mu_mineral**2 * p
        / (2.0 * np.pi**2 * (1.0 - nu) ** 2)
    )
    return k_hm, mu_hm


def _modified_hs(k_end, mu_end, k_mineral, mu_mineral, porosity, phi_c, mu_star):
    """Modified Hashin-Shtrikman interpolation between an endpoint and the mineral."""
    f = np.clip(np.asarray(porosity, dtype=float) / phi_c, 0.0, 1.0)
    z_k = 4.0 * mu_star / 3.0
    k = 1.0 / (f / (k_end + z_k) + (1.0 - f) / (k_mineral + z_k)) - z_k
    z_mu = (mu_star / 6.0) * (9.0 * k_end + 8.0 * mu_star) / (k_end + 2.0 * mu_star) \
        if np.all(mu_star == mu_end) else \
        (mu_star / 6.0) * (9.0 * k_mineral + 8.0 * mu_star) / (k_mineral + 2.0 * mu_star)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(
            mu_star > 0.0,
            1.0 / (f / (mu_end + z_mu) + (1.0 - f) / (mu_mineral + z_mu)) - z_mu,
            0.0,
        )
    return k, mu


def soft_sand(k_mineral, mu_mineral, porosity, effective_pressure,
              critical_porosity: float = DEFAULT_CRITICAL_POROSITY,
              coordination: float = DEFAULT_COORDINATION):
    """Friable / uncemented sand: modified HS lower bound (Dvorkin & Nur 1996).

    Appropriate for unconsolidated to moderately consolidated sands, where
    porosity reduction is by sorting rather than by cement.  Gives the
    stronger pressure sensitivity of the two Dvorkin-Nur models.
    """
    k_hm, mu_hm = hertz_mindlin(k_mineral, mu_mineral, effective_pressure,
                                critical_porosity, coordination)
    z_k = 4.0 * mu_hm / 3.0
    f = np.clip(np.asarray(porosity, dtype=float) / critical_porosity, 0.0, 1.0)
    k = 1.0 / (f / (k_hm + z_k) + (1.0 - f) / (k_mineral + z_k)) - z_k
    z_mu = (mu_hm / 6.0) * (9.0 * k_hm + 8.0 * mu_hm) / (k_hm + 2.0 * mu_hm)
    mu = 1.0 / (f / (mu_hm + z_mu) + (1.0 - f) / (mu_mineral + z_mu)) - z_mu
    return k, mu


def stiff_sand(k_mineral, mu_mineral, porosity, effective_pressure,
               critical_porosity: float = DEFAULT_CRITICAL_POROSITY,
               coordination: float = DEFAULT_COORDINATION):
    """Cemented / stiff sand: modified HS upper bound (Dvorkin & Nur 1996).

    Appropriate for consolidated sands whose porosity loss is by cementation.
    Stiffer and markedly less pressure-sensitive than the soft-sand model,
    which is why the choice between the two changes the pressure half of a
    predicted 4D response.
    """
    k_hm, mu_hm = hertz_mindlin(k_mineral, mu_mineral, effective_pressure,
                                critical_porosity, coordination)
    z_k = 4.0 * mu_mineral / 3.0
    f = np.clip(np.asarray(porosity, dtype=float) / critical_porosity, 0.0, 1.0)
    k = 1.0 / (f / (k_hm + z_k) + (1.0 - f) / (k_mineral + z_k)) - z_k
    z_mu = (mu_mineral / 6.0) * (9.0 * k_mineral + 8.0 * mu_mineral) \
        / (k_mineral + 2.0 * mu_mineral)
    mu = 1.0 / (f / (mu_hm + z_mu) + (1.0 - f) / (mu_mineral + z_mu)) - z_mu
    return k, mu


def critical_porosity_model(k_mineral, mu_mineral, porosity,
                            critical_porosity: float = DEFAULT_CRITICAL_POROSITY):
    """Nur's linear critical-porosity frame: ``M_dry = M_mineral (1 - phi/phi_c)``.

    Has no pressure dependence at all, which makes it a useful control case:
    a 4D experiment run with this frame isolates the saturation response
    from any frame stiffening.
    """
    if not 0 < critical_porosity < 1:
        raise ConfigError(f"critical porosity must be in (0, 1), got {critical_porosity}")
    f = 1.0 - np.clip(np.asarray(porosity, dtype=float) / critical_porosity, 0.0, 1.0)
    return k_mineral * f, mu_mineral * f


#: Name -> callable, for configuration files.
DRY_FRAME_MODELS = {
    "soft_sand": soft_sand,
    "stiff_sand": stiff_sand,
    "critical_porosity": critical_porosity_model,
    "hertz_mindlin": hertz_mindlin,
}
