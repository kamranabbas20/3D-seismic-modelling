import numpy as np
import pytest

from sim3d.compute import NumpyBackend, available_backends, get_backend
from sim3d.core.errors import BackendError, ConfigError


def test_numpy_is_always_available():
    assert "numpy" in available_backends()


def test_auto_selects_an_available_backend():
    assert get_backend("auto").available


def test_gpu_backend_refuses_rather_than_falling_back():
    gpu = get_backend.__globals__["GPUBackend"]()
    assert not gpu.available
    with pytest.raises(BackendError, match="not implemented"):
        gpu.forward_diff(np.zeros((3, 3, 3)), 0, 1.0, 2, np.zeros((2, 3, 3)))
    with pytest.raises(BackendError):
        get_backend("gpu")


def test_unknown_backend_is_rejected():
    with pytest.raises(ConfigError, match="unknown compute backend"):
        get_backend("quantum")


@pytest.mark.skipif("numba" not in available_backends(), reason="numba not installed")
@pytest.mark.parametrize("order", [2, 4, 8])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_numba_and_numpy_kernels_agree_to_round_off(order, axis):
    rng = np.random.default_rng(0)
    ref, jit = NumpyBackend(), get_backend("numba")
    f = rng.standard_normal((17, 19, 23))

    half = list(f.shape)
    half[axis] -= 1
    a, b = np.zeros(half), np.zeros(half)
    ref.forward_diff(f, axis, 3.0, order, a)
    jit.forward_diff(f, axis, 3.0, order, b)
    assert np.allclose(a, b, rtol=0, atol=1e-13)

    g = rng.standard_normal(half)
    c, d = np.zeros(f.shape), np.zeros(f.shape)
    ref.backward_diff(g, axis, 3.0, order, f.shape[axis], c)
    jit.backward_diff(g, axis, 3.0, order, f.shape[axis], d)
    assert np.allclose(c, d, rtol=0, atol=1e-13)
