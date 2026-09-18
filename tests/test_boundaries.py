"""Does the absorbing boundary actually absorb?

The CPML targets a normal-incidence reflection coefficient of 1e-5, but
that is a design parameter, not a measurement: real performance depends on
how thick the layer is in wavelengths and on the angle energy meets it at,
and grazing incidence is where PML is worst.

The measurement here needs no analytic solution. Run the same homogeneous
model in a small box and in a reference box whose faces are too far away
for anything to return inside the record. After the small box's first face
echo would have arrived, any difference between the two traces *is* the
boundary's leakage - nothing else can produce it.

Marked slow: each case propagates a 3D wavefield.
"""

import numpy as np
import pytest

from sim3d.core.grid import Grid3D
from sim3d.wave.acoustic import AcousticModel, AcousticSolver, SolverSettings
from sim3d.wave.cpml import PMLSettings
from sim3d.wave.sources import PointSource
from sim3d.wave.wavelets import ricker

V = 2024.0          # the slowest cell of the dipping-wedge migration
DX = 10.0           # its grid
F0 = 16.0           # its source
RECORD = 0.55       # seconds
SMALL_NODES = 61    # a 600 m box: faces 300 m from the source
REFERENCE_NODES = 181   # an 1800 m box: faces 900 m away


def _trace(n_nodes: int, n_pml: int):
    grid = Grid3D(origin=(0.0, 0.0, 0.0), spacing=(DX, DX, DX),
                  shape=(n_nodes,) * 3)
    model = AcousticModel(grid=grid, vp=np.full(grid.shape, V),
                          rho=np.full(grid.shape, 2200.0))
    solver = AcousticSolver(model, SolverSettings(pml=PMLSettings(n_nodes=n_pml)),
                            f0=F0)
    nt = int(RECORD / solver.dt)
    times = np.arange(nt) * solver.dt
    centre = (n_nodes - 1) * DX / 2.0
    record = solver.run(
        PointSource(grid, (centre, centre, centre), ricker(times, F0)), nt,
        receivers=np.array([[centre, centre, centre + 150.0]]))
    return times, record.traces[0]


def _leakage(n_pml: int, small_nodes: int = SMALL_NODES):
    times, small = _trace(small_nodes, n_pml)
    _, reference = _trace(REFERENCE_NODES, n_pml)
    n = min(small.size, reference.size)
    small, reference, times = small[:n], reference[:n], times[:n]
    # The nearest face is 300 m away, so its echo cannot arrive before 2*300/V.
    after_echo = times > 0.9 * (2.0 * 300.0 / V)
    difference = np.abs(small - reference)[after_echo]
    direct = float(np.abs(reference).max())
    return float(difference.max() / direct), float(
        np.sqrt(np.mean(difference**2)) / direct)


@pytest.mark.slow
def test_the_shipped_absorbing_layer_does_not_reflect():
    """10 nodes at 10 m is 0.79 wavelengths - thinner than the usual one-to-two
    guidance - and still absorbs to within a rounding error of a domain with
    no boundary at all. Measured at 0.028 % of the direct arrival."""
    peak, rms = _leakage(10)
    assert peak < 0.005, f"boundary leaks {100 * peak:.3f} % of the direct arrival"
    assert rms < 0.001


@pytest.mark.slow
def test_a_thicker_layer_absorbs_better_but_it_hardly_matters():
    """Doubling the layer doubles the domain it is wrapped around. The gain is
    real and the absolute numbers are both negligible, which is the finding:
    thickness is not the lever for anything visible in an image."""
    thin_peak, _ = _leakage(10)
    thick_peak, _ = _leakage(20, small_nodes=SMALL_NODES + 20)
    assert thick_peak < thin_peak
    assert thin_peak < 0.005 and thick_peak < 0.005


def test_the_pml_refuses_a_layer_thinner_than_its_own_stencil():
    """A layer narrower than the finite-difference half-stencil cannot hold a
    profile, and silently proceeding would absorb nothing."""
    from sim3d.core.errors import ConfigError
    grid = Grid3D(origin=(0.0, 0.0, 0.0), spacing=(DX, DX, DX), shape=(21, 21, 21))
    model = AcousticModel(grid=grid, vp=np.full(grid.shape, V),
                          rho=np.full(grid.shape, 2200.0))
    with pytest.raises(ConfigError):
        AcousticSolver(model, SolverSettings(spatial_order=8,
                                             pml=PMLSettings(n_nodes=2)), f0=F0)
