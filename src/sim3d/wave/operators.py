r"""Staggered-grid finite-difference derivative operators.

Two operators are needed by the velocity-pressure system, and they differ
in where their result lands:

``forward_diff``
    integer nodes -> half nodes, used for :math:`\nabla p`;
``backward_diff``
    half nodes -> integer nodes, used for :math:`\nabla\cdot\mathbf v`.

Both evaluate

.. math:: (\partial_x f)_{i+1/2} = \frac{1}{\Delta x}
          \sum_{k=1}^{N} c_k\,\big(f_{i+k} - f_{i-k+1}\big)

with the coefficients in :mod:`sim3d.wave.fdscheme`.  The outermost
``N`` nodes of each axis cannot host the full stencil; their derivative is
left at zero, which is why :func:`sim3d.wave.acoustic.AcousticSolver`
requires the absorbing layer to be thicker than the stencil half-width, so
those nodes always sit deep inside the PML.
"""

from __future__ import annotations

import numpy as np

from .fdscheme import coefficients, half_stencil


def _axis_slice(ndim: int, axis: int, sl: slice) -> tuple[slice, ...]:
    out = [slice(None)] * ndim
    out[axis] = sl
    return tuple(out)


def forward_diff(f: np.ndarray, axis: int, spacing: float, order: int,
                 out: np.ndarray | None = None) -> np.ndarray:
    """Derivative of a node-centred field, evaluated on the half grid.

    Input length ``n`` along ``axis`` gives output length ``n - 1``, where
    element ``i`` is the derivative at ``i + 1/2``.
    """
    c = coefficients(order)
    n = f.shape[axis]
    shape = list(f.shape)
    shape[axis] = n - 1
    if out is None:
        out = np.zeros(shape, dtype=f.dtype)
    else:
        out.fill(0.0)
    nd = f.ndim
    for k, ck in enumerate(c, start=1):
        if n - 2 * k + 1 <= 0:
            break
        dst = _axis_slice(nd, axis, slice(k - 1, n - k))
        hi = _axis_slice(nd, axis, slice(2 * k - 1, n))
        lo = _axis_slice(nd, axis, slice(0, n - 2 * k + 1))
        out[dst] += ck * (f[hi] - f[lo])
    out /= spacing
    return out


def backward_diff(f: np.ndarray, axis: int, spacing: float, order: int,
                  n_nodes: int | None = None, out: np.ndarray | None = None) -> np.ndarray:
    """Derivative of a half-grid field, evaluated on the integer nodes.

    Input length ``n - 1`` along ``axis`` (element ``j`` sitting at
    ``j + 1/2``) gives output length ``n``.
    """
    c = coefficients(order)
    n = (f.shape[axis] + 1) if n_nodes is None else int(n_nodes)
    shape = list(f.shape)
    shape[axis] = n
    if out is None:
        out = np.zeros(shape, dtype=f.dtype)
    else:
        out.fill(0.0)
    nd = f.ndim
    for k, ck in enumerate(c, start=1):
        if n - 2 * k <= 0:
            break
        dst = _axis_slice(nd, axis, slice(k, n - k))
        hi = _axis_slice(nd, axis, slice(2 * k - 1, n - 1))
        lo = _axis_slice(nd, axis, slice(0, n - 2 * k))
        out[dst] += ck * (f[hi] - f[lo])
    out /= spacing
    return out


def min_boundary_nodes(order: int) -> int:
    """Nodes at each end of an axis that the full stencil cannot reach."""
    return half_stencil(order)


def average_to_half(f: np.ndarray, axis: int) -> np.ndarray:
    """Arithmetic average of a node-centred field onto the half grid.

    Applied to buoyancy ``1/rho`` rather than to density itself: the
    velocity update needs ``1/rho`` at the staggered location, and
    averaging the reciprocal (equivalently, taking the harmonic mean of
    ``rho``) is the choice that keeps the scheme's effective impedance
    correct across a sharp interface.
    """
    n = f.shape[axis]
    nd = f.ndim
    lo = f[_axis_slice(nd, axis, slice(0, n - 1))]
    hi = f[_axis_slice(nd, axis, slice(1, n))]
    return 0.5 * (lo + hi)
