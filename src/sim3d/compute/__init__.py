"""Compute backends (spec sections 14-16).

The scientific configuration of an experiment never names a backend, so
the same YAML runs on any of them.  Only the kernels differ.
"""

from .backend import (
    ComputeBackend,
    NumpyBackend,
    NumbaBackend,
    GPUBackend,
    available_backends,
    get_backend,
)

__all__ = [
    "ComputeBackend", "NumpyBackend", "NumbaBackend", "GPUBackend",
    "available_backends", "get_backend",
]
