"""Numerical benchmarks against closed-form solutions (spec section 129).

Every test here compares the solver to an analytic answer, not to a
previously recorded output of itself.
"""

import numpy as np
import pytest

from conftest import make_homogeneous, ricker_derivative_peak

from sim3d.core.errors import ValidationError
from sim3d.core.grid import Grid3D
from sim3d.wave.acoustic import (
    AcousticModel, AcousticSolver, SolverSettings, common_dt, steps_for_duration,
)
from sim3d.wave.cpml import PMLSettings
from sim3d.wave.sources import NoSource, PointSource
from sim3d.wave.wavelets import ricker

F0 = 12.0
SETTINGS = SolverSettings(pml=PMLSettings(n_nodes=10), dtype=np.float64)


def test_model_rejects_unphysical_properties():
    grid = Grid3D((0.0, 0.0, 0.0), (10.0,) * 3, (5, 5, 5))
    with pytest.raises(ValidationError, match="positive"):
        AcousticModel(grid, np.zeros(grid.shape), np.ones(grid.shape))
    with pytest.raises(ValidationError, match="non-finite"):
        AcousticModel(grid, np.full(grid.shape, np.nan), np.ones(grid.shape))


def test_homogeneous_medium_matches_the_analytic_greens_function():
    r"""Whole-space acoustic Green's function.

    Injecting ``s(t)`` into the pressure equation gives
    :math:`\ddot p - V^2\nabla^2 p = \kappa\,\dot s\,\delta^3(\mathbf x)`,
    whose solution is :math:`p = \rho\,\dot s(t - r/V)/(4\pi r)`.
    Both the arrival time and the absolute amplitude are checked.
    """
    vp, rho = 2000.0, 2200.0
    model = make_homogeneous((61, 61, 61), spacing=10.0, vp=vp, rho=rho)
    solver = AcousticSolver(model, SETTINGS, f0=F0)
    nt = steps_for_duration(0.25, solver.dt)
    t = np.arange(nt) * solver.dt
    centre = (300.0, 300.0, 300.0)
    offsets = [100.0, 150.0]
    receivers = np.array([[300.0 + o, 300.0, 300.0] for o in offsets])
    record = solver.run(PointSource(model.grid, centre, ricker(t, F0)), nt,
                        receivers=receivers)

    amp, lag = ricker_derivative_peak(F0)
    for offset, trace in zip(offsets, record.traces):
        ipeak = int(np.argmax(np.abs(trace)))
        t_peak = ipeak * solver.dt - 1.0 / F0  # remove the wavelet delay
        assert t_peak == pytest.approx(offset / vp + lag, abs=1.5 * solver.dt)
        expected = rho * amp / (4.0 * np.pi * offset)
        assert abs(trace[ipeak]) == pytest.approx(expected, rel=0.05)


def test_geometric_spreading_follows_one_over_r():
    vp, rho = 2000.0, 2200.0
    model = make_homogeneous((61, 61, 61), spacing=10.0, vp=vp, rho=rho)
    solver = AcousticSolver(model, SETTINGS, f0=F0)
    nt = steps_for_duration(0.25, solver.dt)
    t = np.arange(nt) * solver.dt
    offsets = np.array([100.0, 150.0, 200.0])
    receivers = np.array([[300.0 + o, 300.0, 300.0] for o in offsets])
    record = solver.run(PointSource(model.grid, (300.0,) * 3, ricker(t, F0)), nt,
                        receivers=receivers)
    peaks = np.max(np.abs(record.traces), axis=1)
    products = peaks * offsets
    assert np.ptp(products) / products.mean() < 0.05


@pytest.mark.parametrize(
    "v2,rho2", [(2600.0, 2400.0), (1700.0, 2000.0)], ids=["hard_kick", "soft_kick"]
)
def test_two_layer_reflection_traveltime_and_coefficient(v2, rho2):
    """Flat interface: ``t = sqrt(x^2 + (2h)^2)/V1`` and ``R = (Z2-Z1)/(Z2+Z1)``.

    Both a positive and a negative impedance contrast are checked, so the
    test constrains the polarity as well as the magnitude.

    The recorded event is the *derivative* of the Ricker (see the
    Green's-function test), which is antisymmetric about its arrival, so
    the arrival is picked as the midpoint of its two main lobes rather
    than as the peak of either.  The interface position on a staggered
    grid is only defined to within about one cell, which sets the
    traveltime tolerance.
    """
    v1, rho1 = 2000.0, 2200.0
    grid = Grid3D((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (61, 61, 101))
    z_interface, z_shot, offset = 600.0, 200.0, 60.0
    deep = grid.axis(2)[None, None, :] >= z_interface
    model = AcousticModel(grid,
                          np.where(deep, v2, v1) * np.ones(grid.shape),
                          np.where(deep, rho2, rho1) * np.ones(grid.shape))

    solver = AcousticSolver(model, SETTINGS, f0=F0)
    nt = steps_for_duration(0.55, solver.dt)
    t = np.arange(nt) * solver.dt
    record = solver.run(PointSource(grid, (300.0, 300.0, z_shot), ricker(t, F0)), nt,
                        receivers=np.array([[300.0 + offset, 300.0, z_shot]]))
    trace = record.traces[0]

    h = z_interface - z_shot
    path = np.hypot(offset, 2.0 * h)
    t_reflect = path / v1
    window = (t - 1.0 / F0 > t_reflect - 0.07) & (t - 1.0 / F0 < t_reflect + 0.07)
    windowed = np.where(window, trace, 0.0)
    i_pos, i_neg = int(np.argmax(windowed)), int(np.argmin(windowed))
    arrival = 0.5 * (i_pos + i_neg) * solver.dt - 1.0 / F0

    one_cell_two_way = 2.0 * grid.dz / v1
    assert arrival == pytest.approx(t_reflect, abs=one_cell_two_way + 2 * solver.dt)

    r_coeff = (rho2 * v2 - rho1 * v1) / (rho2 * v2 + rho1 * v1)
    amp, _ = ricker_derivative_peak(F0)
    expected = abs(r_coeff) * rho1 * amp / (4.0 * np.pi * path)
    measured = 0.5 * (abs(windowed[i_pos]) + abs(windowed[i_neg]))
    assert measured == pytest.approx(expected, rel=0.15)

    # Polarity: a hard kick leads with the same sign as the direct wave,
    # a soft kick with the opposite sign.
    direct = trace[int(np.argmax(np.abs(trace)))]
    leading = windowed[min(i_pos, i_neg)]
    assert np.sign(leading) == np.sign(r_coeff) * np.sign(direct)


def test_absorbing_boundary_leaves_little_energy_behind():
    """After the wavefront has crossed every face, the box should be quiet."""
    model = make_homogeneous((45, 45, 45), spacing=10.0, vp=2000.0)
    solver = AcousticSolver(model, SETTINGS, f0=F0)
    grid = model.grid
    nt = steps_for_duration(0.5, solver.dt)  # 1 s of two-way travel across 440 m
    t = np.arange(nt) * solver.dt
    peak = {"value": 0.0}

    def watch(istep, p):
        peak["value"] = max(peak["value"], float(np.max(np.abs(p))))

    solver.run(PointSource(grid, (220.0,) * 3, ricker(t, F0)), nt, on_step=watch)
    interior = solver.p[12:-12, 12:-12, 12:-12]
    assert np.max(np.abs(interior)) / peak["value"] < 5e-3


def test_energy_decays_monotonically_once_the_source_is_silent():
    model = make_homogeneous((45, 45, 45), spacing=10.0, vp=2000.0)
    solver = AcousticSolver(model, SETTINGS, f0=F0)
    nt = steps_for_duration(0.4, solver.dt)
    t = np.arange(nt) * solver.dt
    energies = []

    def watch(istep, p):
        if istep % 20 == 0:
            energies.append(float(np.sum(p.astype(np.float64) ** 2)))

    solver.run(PointSource(model.grid, (220.0,) * 3, ricker(t, F0)), nt, on_step=watch)
    tail = np.array(energies[3:])
    assert np.all(np.diff(tail) <= 1e-12 * tail[0])
    assert tail[-1] / max(energies) < 1e-4


def test_a_silent_run_stays_exactly_zero():
    model = make_homogeneous((30, 30, 30), spacing=10.0)
    solver = AcousticSolver(model, SETTINGS, f0=F0)
    solver.run(NoSource(), 20)
    assert np.all(solver.p == 0.0)


def test_common_dt_is_set_by_the_fastest_model():
    slow = make_homogeneous((30, 30, 30), vp=2000.0)
    fast = make_homogeneous((30, 30, 30), vp=3000.0)
    dt = common_dt([slow, fast], spatial_order=8)
    for model in (slow, fast):
        settings = SolverSettings(pml=PMLSettings(n_nodes=10), dt=dt)
        assert AcousticSolver(model, settings, f0=F0).dt == dt
    assert dt < AcousticSolver(slow, SETTINGS, f0=F0).dt


def test_the_pml_must_be_thicker_than_the_stencil():
    from sim3d.core.errors import ConfigError

    model = make_homogeneous((30, 30, 30))
    with pytest.raises(ConfigError, match="thicker than the stencil"):
        AcousticSolver(model, SolverSettings(spatial_order=8,
                                             pml=PMLSettings(n_nodes=4)), f0=F0)
