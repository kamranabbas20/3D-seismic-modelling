r"""Convolutional PML absorbing boundaries (spec section 63).

Implements the unsplit CPML of Roden & Gedney (2000) in the form given by
Komatitsch & Martin (2007), *An unsplit convolutional perfectly matched
layer improved at grazing incidence for the seismic wave equation*,
Geophysics 72(5), SM155-SM167.

Each spatial derivative inside a boundary layer is replaced by

.. math:: \partial_x \to \frac{1}{\kappa_x}\partial_x + \psi_x,
          \qquad \psi_x^{n} = b_x\,\psi_x^{n-1} + a_x\,\partial_x

with the layer profiles, over a normalised depth :math:`u \in [0, 1]`,

.. math::
    d(u) = d_0 u^{m},\quad
    d_0 = -\frac{(m+1)\,V_{max}\,\ln R_0}{2 L},\quad
    \alpha(u) = \alpha_{max}(1-u),\quad
    \kappa(u) = 1 + (\kappa_{max}-1)u^{m}

and the recursion coefficients

.. math::
    b = e^{-(d/\kappa + \alpha)\,\Delta t},\qquad
    a = \frac{d\,(b-1)}{\kappa\,(d + \kappa\alpha)}.

``alpha`` (the complex frequency shift) is what makes this a *C*-PML: it
suppresses the late-time drift and grazing-incidence leakage that plague
the classical split PML in reservoir-scale surveys with long offsets.

Memory variables are stored only inside the layers, not over the whole
grid, so a 12-node PML on a 200^3 model costs a few tens of MB rather than
several hundred.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.errors import ConfigError

#: Theoretical normal-incidence reflection coefficient targeted by the layer.
DEFAULT_R0 = 1.0e-5
#: Polynomial grading exponent of the damping profile.
DEFAULT_POLY_ORDER = 2.0
#: Default layer thickness in grid nodes.
DEFAULT_NPML = 12


@dataclass(frozen=True)
class PMLSettings:
    """User-facing absorbing-boundary configuration."""

    n_nodes: int = DEFAULT_NPML
    r0: float = DEFAULT_R0
    poly_order: float = DEFAULT_POLY_ORDER
    kappa_max: float = 1.0
    alpha_max_factor: float = np.pi  #: ``alpha_max = factor * f0``
    #: Per-face switch, order ``(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)``.
    #: Set a face ``False`` for a free surface handled elsewhere.
    active: tuple[bool, bool, bool, bool, bool, bool] = (True,) * 6

    def __post_init__(self) -> None:
        if self.n_nodes < 1:
            raise ConfigError(f"PML thickness must be >= 1 node, got {self.n_nodes}")
        if not 0 < self.r0 < 1:
            raise ConfigError(f"PML target reflection must be in (0, 1), got {self.r0}")
        if self.kappa_max < 1.0:
            raise ConfigError(f"kappa_max must be >= 1, got {self.kappa_max}")

    def thickness_m(self, spacing) -> tuple[float, float, float]:
        """Physical layer thickness per axis in metres."""
        return tuple(self.n_nodes * float(d) for d in np.atleast_1d(spacing))


def normalised_depth(positions: np.ndarray, n_nodes_axis: int, npml: int,
                     low: bool = True, high: bool = True) -> np.ndarray:
    """Depth into the layer, 0 at the inner edge and 1 at the model boundary.

    ``positions`` are node coordinates along the axis in node units; use
    ``i`` for integer-node fields and ``i + 0.5`` for half-node (velocity)
    fields, which is why the profiles must be built twice per axis.
    """
    pos = np.asarray(positions, dtype=float)
    u = np.zeros_like(pos)
    if low:
        u = np.maximum(u, (npml - pos) / npml)
    if high:
        u = np.maximum(u, (pos - (n_nodes_axis - 1 - npml)) / npml)
    return np.clip(u, 0.0, 1.0)


@dataclass(frozen=True)
class PMLProfile:
    """Recursion coefficients ``a``, ``b`` and stretch ``kappa`` along one axis."""

    a: np.ndarray
    b: np.ndarray
    kappa: np.ndarray

    @property
    def size(self) -> int:
        return int(self.a.size)


def build_profile(positions: np.ndarray, n_nodes_axis: int, spacing: float, dt: float,
                  vmax: float, f0: float, settings: PMLSettings,
                  low: bool = True, high: bool = True,
                  dtype=np.float64) -> PMLProfile:
    """Build ``a``, ``b``, ``kappa`` for one axis at the given staggered positions."""
    npml = settings.n_nodes
    m = settings.poly_order
    length = npml * float(spacing)
    d0 = -(m + 1.0) * float(vmax) * np.log(settings.r0) / (2.0 * length)

    u = normalised_depth(positions, n_nodes_axis, npml, low=low, high=high)
    d = d0 * u**m
    alpha = settings.alpha_max_factor * float(f0) * (1.0 - u)
    kappa = 1.0 + (settings.kappa_max - 1.0) * u**m

    b = np.exp(-(d / kappa + alpha) * float(dt))
    denom = kappa * (d + kappa * alpha)
    a = np.where(np.abs(denom) > 0.0, d * (b - 1.0) / np.where(denom == 0.0, 1.0, denom), 0.0)
    # Outside the layer d == 0, so a == 0 and the memory variable stays zero.
    a = np.where(u > 0.0, a, 0.0)
    return PMLProfile(a=a.astype(dtype), b=b.astype(dtype), kappa=kappa.astype(dtype))


class AxisCPML:
    """CPML state for one derivative of one field along one axis.

    Holds the two slab-shaped memory variables and applies the correction
    in place.  ``deriv`` is the raw finite-difference derivative array; the
    call returns it corrected for the stretched coordinate.
    """

    def __init__(self, deriv_shape: tuple[int, ...], axis: int, profile: PMLProfile,
                 npml: int, low: bool = True, high: bool = True, dtype=np.float64):
        n_axis = deriv_shape[axis]
        if npml * 2 >= n_axis:
            raise ConfigError(
                f"PML layers ({npml} nodes each side) do not fit along axis "
                f"{'xyz'[axis]} of length {n_axis}; enlarge the propagation "
                f"domain or thin the boundary"
            )
        self.axis = axis
        self.npml = int(npml)
        self.profile = profile
        self.low = bool(low)
        self.high = bool(high)
        self._n_axis = int(n_axis)

        slab = list(deriv_shape)
        slab[axis] = npml
        self.psi_lo = np.zeros(slab, dtype=dtype) if low else None
        self.psi_hi = np.zeros(slab, dtype=dtype) if high else None
        self._shape_for_bcast = [1] * len(deriv_shape)
        self._shape_for_bcast[axis] = npml

    def _slice(self, which: str) -> tuple[slice, ...]:
        sl = [slice(None)] * self.psi_lo.ndim if self.psi_lo is not None else [slice(None)] * self.psi_hi.ndim
        sl[self.axis] = slice(0, self.npml) if which == "lo" else slice(self._n_axis - self.npml, self._n_axis)
        return tuple(sl)

    def apply(self, deriv: np.ndarray) -> np.ndarray:
        """Apply the CPML stretch and memory update to ``deriv`` in place."""
        bshape = tuple(self._shape_for_bcast)
        if self.low:
            sl = self._slice("lo")
            a = self.profile.a[: self.npml].reshape(bshape)
            b = self.profile.b[: self.npml].reshape(bshape)
            kap = self.profile.kappa[: self.npml].reshape(bshape)
            chunk = deriv[sl]
            self.psi_lo *= b
            self.psi_lo += a * chunk
            deriv[sl] = chunk / kap + self.psi_lo
        if self.high:
            sl = self._slice("hi")
            a = self.profile.a[-self.npml:].reshape(bshape)
            b = self.profile.b[-self.npml:].reshape(bshape)
            kap = self.profile.kappa[-self.npml:].reshape(bshape)
            chunk = deriv[sl]
            self.psi_hi *= b
            self.psi_hi += a * chunk
            deriv[sl] = chunk / kap + self.psi_hi
        return deriv

    def reset(self) -> None:
        """Zero the memory variables (needed between forward and adjoint runs)."""
        if self.psi_lo is not None:
            self.psi_lo.fill(0.0)
        if self.psi_hi is not None:
            self.psi_hi.fill(0.0)

    @property
    def nbytes(self) -> int:
        total = 0
        if self.psi_lo is not None:
            total += self.psi_lo.nbytes
        if self.psi_hi is not None:
            total += self.psi_hi.nbytes
        return total
