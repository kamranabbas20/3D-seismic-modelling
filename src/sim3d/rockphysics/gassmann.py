r"""Gassmann fluid substitution (spec section 45).

Gassmann, F. (1951), *Uber die Elastizitat poroser Medien*.  See also
Mavko et al. (2009) Section 6.3 and Smith, Sondergeld & Rai (2003) for the
practical recipe.

.. math::
    K_{sat} = K_{dry} +
        \frac{\left(1 - K_{dry}/K_{min}\right)^2}
             {\dfrac{\phi}{K_{fl}} + \dfrac{1-\phi}{K_{min}}
              - \dfrac{K_{dry}}{K_{min}^2}},
    \qquad \mu_{sat} = \mu_{dry}

Assumptions, all of which matter for 4D interpretation:

* the rock is monomineralic and isotropic, or well described by an
  effective mineral;
* the pore space is fully connected and pore pressure equilibrates over a
  seismic period - the low-frequency limit.  At log or ultrasonic
  frequencies this is violated and Gassmann under-predicts the saturated
  modulus;
* the fluid does not chemically alter the frame;
* the shear modulus is unaffected by the pore fluid.

The last point is what makes the pressure/saturation decomposition
tractable: saturation changes act only on ``K``, while an effective-stress
change acts on both ``K`` and ``mu``.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import ValidationError


def gassmann_saturated_modulus(k_dry, mu_dry, k_mineral, k_fluid, porosity,
                               tolerance: float = 1e-6):
    """Saturated bulk modulus from the dry frame and a pore fluid.

    All moduli in Pa, porosity as a fraction.  Raises
    :class:`~sim3d.core.errors.ValidationError` when the inputs are outside
    the model's validity rather than returning a negative modulus that
    would later surface as a NaN velocity.
    """
    k_dry = np.asarray(k_dry, dtype=float)
    k_min = np.asarray(k_mineral, dtype=float)
    k_fl = np.asarray(k_fluid, dtype=float)
    phi = np.asarray(porosity, dtype=float)

    if np.any(phi <= 0) or np.any(phi >= 1):
        raise ValidationError(
            f"porosity must be strictly between 0 and 1 for fluid substitution; "
            f"it spans [{np.min(phi):g}, {np.max(phi):g}]"
        )
    if np.any(k_dry > k_min * (1.0 + tolerance)):
        raise ValidationError(
            f"the dry frame is stiffer than its own mineral "
            f"({np.max(k_dry) / 1e9:.3f} > {np.max(k_min) / 1e9:.3f} GPa), which is "
            f"unphysical; check the frame model and porosity"
        )
    if np.any(k_fl <= 0):
        raise ValidationError("fluid bulk modulus must be positive")

    denominator = phi / k_fl + (1.0 - phi) / k_min - k_dry / k_min**2
    if np.any(denominator <= 0):
        raise ValidationError(
            "the Gassmann denominator is non-positive somewhere, so the "
            "substitution is undefined there; this normally means the dry "
            "frame modulus is too close to the mineral modulus for the "
            "porosity given"
        )
    k_sat = k_dry + (1.0 - k_dry / k_min) ** 2 / denominator
    if np.any(k_sat > k_min * (1.0 + tolerance)):
        raise ValidationError(
            "Gassmann produced a saturated modulus above the mineral modulus"
        )
    return k_sat


def gassmann_substitute(k_sat_initial, mu, k_mineral, k_fluid_initial, k_fluid_new,
                        porosity):
    """Substitute one fluid for another in an already-saturated rock.

    Inverts Gassmann to recover the dry frame from the initial saturated
    state, then re-saturates with the new fluid.  Use this when the
    measured input is a saturated log or volume rather than a dry frame.
    """
    k_sat = np.asarray(k_sat_initial, dtype=float)
    k_min = np.asarray(k_mineral, dtype=float)
    k_fl0 = np.asarray(k_fluid_initial, dtype=float)
    phi = np.asarray(porosity, dtype=float)

    ratio = k_sat / (k_min - k_sat) - k_fl0 / (phi * (k_min - k_fl0))
    k_dry = k_min * ratio / (1.0 + ratio)
    if np.any(k_dry < 0):
        raise ValidationError(
            "inverting Gassmann gave a negative dry-frame modulus; the input "
            "saturated modulus, mineral modulus and porosity are mutually "
            "inconsistent"
        )
    return gassmann_saturated_modulus(k_dry, mu, k_min, k_fluid_new, phi), k_dry


def velocities(k, mu, rho):
    """``(Vp, Vs)`` in m/s from saturated moduli (Pa) and density (kg/m^3)."""
    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if np.any(rho <= 0):
        raise ValidationError("density must be positive")
    vp = np.sqrt((k + 4.0 * mu / 3.0) / rho)
    vs = np.sqrt(mu / rho)
    return vp, vs
