import numpy as np
import pytest

from sim3d.core.grid import Grid3D
from sim3d.wave.acoustic import AcousticModel, SolverSettings
from sim3d.wave.cpml import PMLSettings


@pytest.fixture(scope="session")
def homogeneous_settings():
    return SolverSettings(pml=PMLSettings(n_nodes=12), dtype=np.float64)


def make_homogeneous(shape=(81, 81, 81), spacing=8.0, vp=2000.0, rho=2200.0):
    grid = Grid3D((0.0, 0.0, 0.0), (spacing,) * 3, shape)
    return AcousticModel(grid, np.full(shape, vp), np.full(shape, rho))


def ricker_derivative_peak(f0):
    """(max |d/dt ricker|, lag of that extremum) for the analytic reference.

    The Ricker's derivative extremum sits at ``u = 0.5246609`` in units of
    ``pi f0 tau``, from ``d/du[(4u^3 - 6u) exp(-u^2)] = 0``.
    """
    u = 0.5246609
    amp = np.pi * f0 * abs(4 * u**3 - 6 * u) * np.exp(-(u**2))
    return amp, -u / (np.pi * f0)
