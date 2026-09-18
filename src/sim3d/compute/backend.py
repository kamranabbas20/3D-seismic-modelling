"""Pluggable compute backends for the finite-difference kernels.

The wave solver calls a backend for its two hot operations - the staggered
forward and backward derivatives - and does everything else (boundary
memory variables, source injection, receiver gathers) in NumPy, where the
work is confined to thin slabs and a handful of points.

``NumpyBackend``
    Reference implementation.  Correct, dependency-free, memory-bound: the
    8th-order operator makes four separate passes over the grid per axis.
``NumbaBackend``
    The same arithmetic fused into one pass with ``prange`` over the outer
    axis.  Selected automatically when Numba is importable.
``GPUBackend``
    Deliberately unimplemented.  It exists so that the interface a GPU
    kernel must satisfy is fixed now (spec section 16), and it raises
    rather than silently falling back to the CPU.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.errors import BackendError, ConfigError
from ..wave import operators as _ops
from ..wave.fdscheme import coefficients


class ComputeBackend(ABC):
    """Interface every backend must satisfy."""

    name: str = "abstract"

    @property
    @abstractmethod
    def available(self) -> bool:
        """True when this backend can actually run here."""

    @abstractmethod
    def forward_diff(self, f: np.ndarray, axis: int, spacing: float, order: int,
                     out: np.ndarray) -> np.ndarray:
        """Node-centred field -> half-grid derivative, written into ``out``."""

    @abstractmethod
    def backward_diff(self, f: np.ndarray, axis: int, spacing: float, order: int,
                      n_nodes: int, out: np.ndarray) -> np.ndarray:
        """Half-grid field -> node-centred derivative, written into ``out``."""

    def describe(self) -> str:
        return f"{self.name} backend ({'available' if self.available else 'unavailable'})"


class NumpyBackend(ComputeBackend):
    """Reference NumPy implementation."""

    name = "numpy"

    @property
    def available(self) -> bool:
        return True

    def forward_diff(self, f, axis, spacing, order, out):
        return _ops.forward_diff(f, axis, spacing, order, out=out)

    def backward_diff(self, f, axis, spacing, order, n_nodes, out):
        return _ops.backward_diff(f, axis, spacing, order, n_nodes=n_nodes, out=out)


# --------------------------------------------------------------------------
# Numba kernels.  Six explicit kernels rather than one generic one: the
# stencil axis must be the innermost loop dimension it can be for each case,
# and specialising is what makes the single-pass fusion worth having.
# --------------------------------------------------------------------------
try:  # pragma: no cover - exercised by whichever path the machine supports
    from numba import njit, prange

    _NUMBA_OK = True
except Exception:  # pragma: no cover
    _NUMBA_OK = False


if _NUMBA_OK:
    _JIT = dict(parallel=True, fastmath=True, cache=True, nogil=True)

    @njit(**_JIT)
    def _fwd0(f, out, c, inv_d):
        nx, ny, nz = f.shape
        nc = c.shape[0]
        for i in prange(nx - 1):
            for j in range(ny):
                for k in range(nz):
                    acc = 0.0
                    for m in range(nc):
                        a = i + m + 1
                        b = i - m
                        if a < nx and b >= 0:
                            acc += c[m] * (f[a, j, k] - f[b, j, k])
                    out[i, j, k] = acc * inv_d

    @njit(**_JIT)
    def _fwd1(f, out, c, inv_d):
        nx, ny, nz = f.shape
        nc = c.shape[0]
        for i in prange(nx):
            for j in range(ny - 1):
                for k in range(nz):
                    acc = 0.0
                    for m in range(nc):
                        a = j + m + 1
                        b = j - m
                        if a < ny and b >= 0:
                            acc += c[m] * (f[i, a, k] - f[i, b, k])
                    out[i, j, k] = acc * inv_d

    @njit(**_JIT)
    def _fwd2(f, out, c, inv_d):
        nx, ny, nz = f.shape
        nc = c.shape[0]
        for i in prange(nx):
            for j in range(ny):
                for k in range(nz - 1):
                    acc = 0.0
                    for m in range(nc):
                        a = k + m + 1
                        b = k - m
                        if a < nz and b >= 0:
                            acc += c[m] * (f[i, j, a] - f[i, j, b])
                    out[i, j, k] = acc * inv_d

    @njit(**_JIT)
    def _bwd0(f, out, c, inv_d):
        nx, ny, nz = out.shape
        nf = f.shape[0]
        nc = c.shape[0]
        for i in prange(nx):
            for j in range(ny):
                for k in range(nz):
                    acc = 0.0
                    for m in range(nc):
                        a = i + m
                        b = i - m - 1
                        if a < nf and b >= 0:
                            acc += c[m] * (f[a, j, k] - f[b, j, k])
                    out[i, j, k] = acc * inv_d

    @njit(**_JIT)
    def _bwd1(f, out, c, inv_d):
        nx, ny, nz = out.shape
        nf = f.shape[1]
        nc = c.shape[0]
        for i in prange(nx):
            for j in range(ny):
                for k in range(nz):
                    acc = 0.0
                    for m in range(nc):
                        a = j + m
                        b = j - m - 1
                        if a < nf and b >= 0:
                            acc += c[m] * (f[i, a, k] - f[i, b, k])
                    out[i, j, k] = acc * inv_d

    @njit(**_JIT)
    def _bwd2(f, out, c, inv_d):
        nx, ny, nz = out.shape
        nf = f.shape[2]
        nc = c.shape[0]
        for i in prange(nx):
            for j in range(ny):
                for k in range(nz):
                    acc = 0.0
                    for m in range(nc):
                        a = k + m
                        b = k - m - 1
                        if a < nf and b >= 0:
                            acc += c[m] * (f[i, j, a] - f[i, j, b])
                    out[i, j, k] = acc * inv_d

    _FWD = (_fwd0, _fwd1, _fwd2)
    _BWD = (_bwd0, _bwd1, _bwd2)


class NumbaBackend(ComputeBackend):
    """Single-pass JIT kernels; identical arithmetic to :class:`NumpyBackend`."""

    name = "numba"

    def __init__(self):
        self._coeff_cache: dict[tuple[int, str], np.ndarray] = {}

    @property
    def available(self) -> bool:
        return _NUMBA_OK

    def _coeffs(self, order: int, dtype) -> np.ndarray:
        key = (order, np.dtype(dtype).name)
        if key not in self._coeff_cache:
            self._coeff_cache[key] = coefficients(order).astype(dtype)
        return self._coeff_cache[key]

    def forward_diff(self, f, axis, spacing, order, out):
        if not _NUMBA_OK:
            raise BackendError("Numba is not installed; install sim3d[accel]")
        _FWD[axis](f, out, self._coeffs(order, f.dtype), f.dtype.type(1.0 / spacing))
        return out

    def backward_diff(self, f, axis, spacing, order, n_nodes, out):
        if not _NUMBA_OK:
            raise BackendError("Numba is not installed; install sim3d[accel]")
        _BWD[axis](f, out, self._coeffs(order, f.dtype), f.dtype.type(1.0 / spacing))
        return out


class GPUBackend(ComputeBackend):
    """Placeholder for the future CuPy / JAX / CUDA path (spec section 16).

    It raises instead of quietly running on the CPU, because a run that
    silently used different hardware than requested is a run whose
    performance numbers and provenance are wrong.
    """

    name = "gpu"

    @property
    def available(self) -> bool:
        return False

    def forward_diff(self, f, axis, spacing, order, out):
        raise BackendError(
            "the GPU backend is not implemented yet; select 'numba' or 'numpy'"
        )

    def backward_diff(self, f, axis, spacing, order, n_nodes, out):
        raise BackendError(
            "the GPU backend is not implemented yet; select 'numba' or 'numpy'"
        )


_REGISTRY: dict[str, type[ComputeBackend]] = {
    "numpy": NumpyBackend,
    "numba": NumbaBackend,
    "gpu": GPUBackend,
}


def available_backends() -> list[str]:
    """Names of backends that can run on this machine."""
    return [name for name, cls in _REGISTRY.items() if cls().available]


def get_backend(name: str | ComputeBackend | None = None) -> ComputeBackend:
    """Resolve a backend by name.

    ``None`` or ``"auto"`` picks the fastest available (Numba if present),
    and says so rather than pretending the choice did not happen.
    """
    if isinstance(name, ComputeBackend):
        return name
    if name in (None, "auto"):
        nb = NumbaBackend()
        return nb if nb.available else NumpyBackend()
    try:
        backend = _REGISTRY[name]()
    except KeyError:
        raise ConfigError(
            f"unknown compute backend {name!r}; choose from {sorted(_REGISTRY)}"
        ) from None
    if not backend.available:
        raise BackendError(
            f"the {name!r} backend is not available on this machine; "
            f"available: {available_backends()}"
        )
    return backend
