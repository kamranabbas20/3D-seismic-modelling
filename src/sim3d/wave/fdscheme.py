r"""Finite-difference scheme properties: coefficients, stability, dispersion.

Spec sections 11 and 12 forbid hard-coding an FD grid size or time step.
Everything here is derived from the scheme actually in use.

Scheme
------
sim3d propagates the first-order velocity-pressure acoustic system on a
standard staggered grid (Virieux 1986; Levander 1988; Graves 1996)::

    dv/dt  = -(1/rho) grad p
    dp/dt  = -kappa div v + s,        kappa = rho * Vp^2

Spatial derivatives use a centred staggered operator of order ``2N``::

    (df/dx)_{i+1/2} = (1/dx) * sum_{k=1..N} c_k * ( f_{i+k} - f_{i-k+1} )

Time stepping is 2nd-order leapfrog.

Discrete dispersion
-------------------
Substituting a plane wave gives the effective wavenumber of the spatial
operator,

.. math:: \tilde k(k) = \frac{2}{dx}\sum_k c_k \sin\!\big((k-\tfrac12) k\,dx\big)

and the fully discrete dispersion relation of the leapfrog system,

.. math:: \sin(\omega\,dt/2) = \tfrac12 c\,dt\,|\tilde{\mathbf k}|.

Both the stability limit and the points-per-wavelength requirement below
are read off these two expressions rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..core.errors import ConfigError, StabilityError

#: Staggered-grid first-derivative coefficients ``c_1..c_N`` by spatial order.
STAGGERED_COEFFS: dict[int, tuple[Fraction, ...]] = {
    2: (Fraction(1),),
    4: (Fraction(9, 8), Fraction(-1, 24)),
    6: (Fraction(75, 64), Fraction(-25, 384), Fraction(3, 640)),
    8: (Fraction(1225, 1024), Fraction(-245, 3072), Fraction(49, 5120), Fraction(-5, 7168)),
    10: (
        Fraction(19845, 16384), Fraction(-735, 8192), Fraction(567, 40960),
        Fraction(-405, 229376), Fraction(35, 294912),
    ),
}

#: Default safety factor applied to the analytic CFL limit.
DEFAULT_COURANT_SAFETY = 0.90

#: Default tolerated phase-velocity error used to pick the grid spacing.
DEFAULT_DISPERSION_TOLERANCE = 0.01


def coefficients(order: int) -> np.ndarray:
    """Staggered first-derivative coefficients for the given spatial order."""
    try:
        return np.array([float(c) for c in STAGGERED_COEFFS[order]], dtype=float)
    except KeyError:
        raise ConfigError(
            f"spatial order {order} not available; choose one of "
            f"{sorted(STAGGERED_COEFFS)}"
        ) from None


def half_stencil(order: int) -> int:
    """Number of nodes the operator reaches on each side (``order // 2``)."""
    return order // 2


def effective_wavenumber(k_dx: np.ndarray | float, order: int) -> np.ndarray:
    """Effective wavenumber ``k~ * dx`` of the discrete operator.

    ``k_dx`` is the true wavenumber times the grid spacing.  The exact
    operator would return ``k_dx`` itself; the deviation is the spatial
    dispersion error.
    """
    c = coefficients(order)
    idx = np.arange(1, len(c) + 1)
    k_dx = np.asarray(k_dx, dtype=float)
    return 2.0 * np.sum(
        c * np.sin(np.multiply.outer(k_dx, idx - 0.5)), axis=-1
    )


def max_courant(order: int, ndim: int = 3) -> float:
    r"""Analytic stability limit as ``dt * Vmax * sqrt(sum 1/dx_i^2) <= limit``.

    From :math:`|c\,dt\,\tilde k| \le 2` with
    :math:`\max|\tilde k_i| = (2/dx_i)\sum_k |c_k|`, the limit is
    :math:`1/\sum_k|c_k|` and is independent of ``ndim`` once the axes are
    combined through :math:`\sqrt{\sum 1/dx_i^2}`.
    """
    if ndim not in (1, 2, 3):
        raise ConfigError(f"ndim must be 1, 2 or 3, got {ndim}")
    return 1.0 / float(np.sum(np.abs(coefficients(order))))


def max_stable_dt(spacing, vmax: float, order: int, safety: float = DEFAULT_COURANT_SAFETY) -> float:
    """Largest stable time step, in seconds, including the safety factor.

    Parameters
    ----------
    spacing:
        ``(dx, dy, dz)`` in metres (a scalar is broadcast to all axes).
    vmax:
        Maximum P velocity anywhere in the propagation domain, m/s.
    order:
        Spatial order of the FD operator.
    safety:
        Fraction of the analytic limit to use (``1.0`` sits exactly on it).
    """
    d = np.atleast_1d(np.asarray(spacing, dtype=float))
    if np.any(d <= 0):
        raise ConfigError(f"grid spacing must be positive, got {spacing}")
    if vmax <= 0:
        raise ConfigError(f"vmax must be positive, got {vmax}")
    inv = float(np.sqrt(np.sum(1.0 / d**2)))
    return safety * max_courant(order, ndim=d.size) / (vmax * inv)


def check_dt(dt: float, spacing, vmax: float, order: int,
             safety: float = DEFAULT_COURANT_SAFETY) -> "CFLReport":
    """Validate a time step and return the numbers the user must be shown.

    Raises :class:`StabilityError` when the step is unstable.  The solver
    never silently reduces ``dt`` (spec section 12); the caller decides.
    """
    dt_max = max_stable_dt(spacing, vmax, order, safety=safety)
    limit = max_stable_dt(spacing, vmax, order, safety=1.0)
    report = CFLReport(
        requested_dt=float(dt), max_stable_dt=float(dt_max),
        analytic_limit_dt=float(limit), vmax=float(vmax), order=int(order),
        safety=float(safety),
    )
    if dt > limit:
        raise StabilityError(
            f"CFL violated: dt={dt * 1e3:.4f} ms exceeds the analytic stability "
            f"limit {limit * 1e3:.4f} ms for Vmax={vmax:g} m/s, order {order}, "
            f"spacing {tuple(float(v) for v in np.atleast_1d(spacing))} m. "
            f"Reduce dt to at most {dt_max * 1e3:.4f} ms, increase the grid "
            f"spacing, or lower Vmax by changing the model - sim3d will not "
            f"adjust dt for you."
        )
    return report


@dataclass(frozen=True)
class CFLReport:
    """Everything spec section 12 requires to be displayed about the time step."""

    requested_dt: float
    max_stable_dt: float
    analytic_limit_dt: float
    vmax: float
    order: int
    safety: float

    @property
    def cfl_ratio(self) -> float:
        """Requested dt as a fraction of the analytic stability limit."""
        return self.requested_dt / self.analytic_limit_dt

    def describe(self) -> str:
        return (
            f"CFL: requested dt = {self.requested_dt * 1e3:.4f} ms, "
            f"max stable (safety {self.safety:g}) = {self.max_stable_dt * 1e3:.4f} ms, "
            f"analytic limit = {self.analytic_limit_dt * 1e3:.4f} ms, "
            f"ratio = {self.cfl_ratio:.3f} (Vmax {self.vmax:g} m/s, order {self.order})"
        )


def phase_velocity_error(ppw: float, order: int, courant: float,
                         n_directions: int = 24) -> float:
    """Worst-case relative phase-velocity error of the fully discrete scheme.

    Parameters
    ----------
    ppw:
        Grid points per wavelength, ``lambda / dx``.
    order:
        Spatial order.
    courant:
        ``V * dt * sqrt(sum 1/dx_i^2)`` - the same dimensionless number
        :func:`max_courant` bounds.  Time-stepping error partly cancels
        spatial error, so the answer genuinely depends on it.
    n_directions:
        Sampling of propagation directions over the irreducible octant.

    Returns
    -------
    float
        ``max |v_phase / v_exact - 1|`` over direction, isotropic grid assumed.
    """
    if ppw <= 0:
        raise ConfigError(f"points per wavelength must be positive, got {ppw}")
    k_dx = 2.0 * np.pi / float(ppw)
    theta = np.linspace(0.0, np.pi / 2, n_directions)
    phi = np.linspace(0.0, np.pi / 2, n_directions)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    comps = np.stack(
        [
            np.sin(th) * np.cos(ph),
            np.sin(th) * np.sin(ph),
            np.cos(th),
        ]
    )
    k_eff = np.sqrt(
        np.sum(effective_wavenumber(k_dx * comps, order) ** 2, axis=0)
    )
    # sin(omega dt / 2) = (courant / 2) * (|k~| dx) / sqrt(3)  for an isotropic
    # grid, where courant = V dt sqrt(3)/dx.  Solve for omega, then v = omega/k.
    arg = 0.5 * courant * k_eff / np.sqrt(3.0)
    if np.any(arg >= 1.0):
        return np.inf  # beyond the stability limit: no real phase velocity
    omega_scaled = 2.0 * np.arcsin(arg)  # = omega * dt
    v_ratio = omega_scaled * np.sqrt(3.0) / (courant * k_dx)
    return float(np.max(np.abs(v_ratio - 1.0)))


def required_ppw(order: int, tolerance: float = DEFAULT_DISPERSION_TOLERANCE,
                 courant: float | None = None, safety: float = DEFAULT_COURANT_SAFETY) -> float:
    """Minimum points per wavelength meeting a phase-velocity error tolerance.

    Solved from :func:`phase_velocity_error` by bisection, so the answer
    tracks the scheme in use instead of the folklore "8 cells per
    wavelength".  ``courant`` defaults to the operating point implied by
    ``safety``.
    """
    if not 0 < tolerance < 1:
        raise ConfigError(f"tolerance must be in (0, 1), got {tolerance}")
    if courant is None:
        courant = safety * max_courant(order)
    lo, hi = 1.5, 200.0
    if phase_velocity_error(hi, order, courant) > tolerance:
        raise ConfigError(
            f"cannot reach a {tolerance:.3%} dispersion tolerance with order "
            f"{order} at Courant number {courant:g}"
        )
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if phase_velocity_error(mid, order, courant) > tolerance:
            lo = mid
        else:
            hi = mid
    return float(hi)


@dataclass(frozen=True)
class SamplingRecommendation:
    """The frequency-dependent sampling report of spec section 11."""

    vmin: float
    fmax: float
    order: int
    tolerance: float
    points_per_wavelength: float
    min_wavelength: float
    max_spacing: float

    def describe(self) -> str:
        return (
            f"Sampling: Vmin = {self.vmin:g} m/s, Fmax = {self.fmax:g} Hz, "
            f"lambda_min = {self.min_wavelength:.2f} m; order {self.order} needs "
            f"{self.points_per_wavelength:.2f} cells/wavelength for "
            f"{self.tolerance:.2%} phase error, so dx <= {self.max_spacing:.3f} m"
        )


def recommend_spacing(vmin: float, fmax: float, order: int = 8,
                      tolerance: float = DEFAULT_DISPERSION_TOLERANCE,
                      safety: float = DEFAULT_COURANT_SAFETY) -> SamplingRecommendation:
    """Largest grid spacing that respects the dispersion tolerance.

    ``lambda_min = vmin / fmax`` and ``dx <= lambda_min / ppw``.
    """
    if vmin <= 0 or fmax <= 0:
        raise ConfigError(f"vmin and fmax must be positive, got {vmin}, {fmax}")
    ppw = required_ppw(order, tolerance=tolerance, safety=safety)
    lam = vmin / fmax
    return SamplingRecommendation(
        vmin=float(vmin), fmax=float(fmax), order=int(order), tolerance=float(tolerance),
        points_per_wavelength=ppw, min_wavelength=float(lam),
        max_spacing=float(lam / ppw),
    )


def cells_per_wavelength(spacing, vmin: float, fmax: float) -> float:
    """Actual cells per minimum wavelength for a grid, using the coarsest axis."""
    d = float(np.max(np.atleast_1d(np.asarray(spacing, dtype=float))))
    return (vmin / fmax) / d
