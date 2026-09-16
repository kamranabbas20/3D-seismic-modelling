"""Solution-gas liberation benchmarks.

As with the rest of the flow tests, every assertion here is against a
conservation law, a thermodynamic identity or a stated physical claim -
never against a stored output of this code.
"""

import numpy as np
import pytest

from sim3d.core.errors import ConfigError
from sim3d.core.units import PSI
from sim3d.reservoir.flow import FlowSettings, FlowSimulator
from sim3d.reservoir.gas import SolutionGas
from sim3d.rockphysics.fluids import bubble_point, gas_fvf, solution_gor
from sim3d.wells import Well
from sim3d.wells.completion import Completion
from sim3d.wells.controls import ControlMode, WellControl

from test_flow import slab

GAS = SolutionGas(initial_gor=15.0, critical_saturation=0.02)


# ------------------------------------------------------------------- PVT
def test_bubble_point_and_solution_gor_are_inverses():
    """The flow model and the QC check must agree on where the bubble point is.

    They can only disagree if the GOR-at-pressure curve is a second fit
    rather than the first one solved the other way round, so this is the
    property that keeps them honest.
    """
    p = np.array([50.0, 200.0, 600.0, 1500.0, 4000.0]) * PSI
    rs = solution_gor(p, api=32.0, gas_gravity=0.7, temperature=75.0)
    back = bubble_point(32.0, 0.7, rs, 75.0)
    assert np.allclose(back, p, rtol=1e-9)


def test_gas_volume_factor_follows_the_ideal_gas_law_loosely():
    """Bg is close to p_sc T / (p T_sc) - it is Z that makes it interesting."""
    p = np.array([500.0, 2000.0]) * PSI
    bg = gas_fvf(p, 80.0, 0.65)
    ideal = 101325.0 / p * (353.15 / 288.706)
    assert np.all(bg < ideal)                  # Z < 1 for a real gas here
    assert np.all(bg > 0.8 * ideal)
    assert bg[0] > bg[1]                       # gas shrinks as pressure rises


def test_refuses_impossible_gas_settings():
    with pytest.raises(ConfigError):
        SolutionGas(initial_gor=-1.0)
    with pytest.raises(ConfigError):
        SolutionGas(critical_saturation=0.0)
    with pytest.raises(ConfigError):
        SolutionGas(gas_viscosity=0.0)


# ----------------------------------------------------------------- flash
def _cell(pressure, sw=0.3, sg=0.0, volume=1000.0, gas=GAS):
    vp = np.array([volume])
    p = np.array([pressure])
    oil, total = gas.initial_moles(p, np.array([sw]), np.array([sg]), vp)
    return vp, oil, total


def test_no_free_gas_above_the_bubble_point():
    """The property that the first formulation of this got wrong.

    Deriving stock-tank oil from the oil saturation rather than tracking it
    leaves the oil's own expansion nowhere to go in a fixed pore volume, and
    the flash reads that surplus as gas - putting free gas hundreds of psi
    *above* the bubble point, where there can be none.
    """
    vp, oil, total = _cell(3000.0 * PSI)
    above = np.array([3000.0, 2000.0, 1200.0, 700.0]) * PSI
    assert np.all(above > GAS.bubble_point)
    for p in above:
        sg, rs, _ = GAS.flash(np.array([p]), oil, total, vp, np.array([0.3]))
        assert sg[0] == 0.0
        assert rs[0] == pytest.approx(GAS.initial_gor, rel=1e-9)


def test_flash_conserves_gas_exactly():
    """Dissolved plus free must equal what went in, at any pressure."""
    vp, oil, total = _cell(2000.0 * PSI)
    for psi in (2000.0, 600.0, 400.0, 300.0):
        p = np.array([psi * PSI])
        sg, rs, over = GAS.flash(p, oil, total, vp, np.array([0.3]))
        assert not over[0]
        accounted = oil[0] * rs[0] + vp[0] * sg[0] / GAS.bg(p)[0]
        assert accounted == pytest.approx(total[0], rel=1e-10)


def test_a_cell_with_no_room_left_is_flagged_rather_than_fudged():
    """Below some pressure a sealed cell simply cannot hold its own gas.

    The saturation is clipped there because there is nowhere else for it to
    go, and the flag is how the simulator knows to say so instead of
    reporting a number it silently invented.
    """
    vp, oil, total = _cell(2000.0 * PSI)
    p = np.array([80.0 * PSI])
    sg, _, over = GAS.flash(p, oil, total, vp, np.array([0.3]))
    assert over[0]
    assert sg[0] == pytest.approx(0.7)          # everything the water left


def test_liberation_grows_as_pressure_falls_below_the_bubble_point():
    vp, oil, total = _cell(2000.0 * PSI)
    pressures = np.array([600.0, 500.0, 400.0, 300.0, 200.0]) * PSI
    sg, rs, _ = GAS.flash(pressures, np.repeat(oil, 5), np.repeat(total, 5),
                          np.repeat(vp, 5), np.full(5, 0.3))
    assert np.all(np.diff(sg) > 0)             # more gas out, lower pressure
    assert np.all(np.diff(rs) < 0)             # less left in solution


def test_gas_goes_back_into_solution_when_pressure_recovers():
    """Liberation is a reversible flash, not a one-way valve."""
    vp, oil, total = _cell(2000.0 * PSI)
    low = np.array([300.0 * PSI])
    sg_low, rs_low, _ = GAS.flash(low, oil, total, vp, np.array([0.3]))
    assert sg_low[0] > 0.01
    back = np.array([2000.0 * PSI])
    sg_back, rs_back, _ = GAS.flash(back, oil, total, vp, np.array([0.3]))
    assert sg_back[0] == 0.0
    assert rs_back[0] == pytest.approx(GAS.initial_gor, rel=1e-9)
    assert rs_low[0] < rs_back[0]


# ------------------------------------------------------- compressibility
def test_compressibility_is_the_base_above_the_bubble_point():
    base = 4.0e-10
    ct = GAS.total_compressibility(np.array([2000.0 * PSI]), np.array([0.3]),
                                   np.array([0.0]), np.array([15.0]), base)
    assert ct[0] == base


def test_gas_expansion_stiffens_the_reservoir_by_orders_of_magnitude():
    """The claim the pressure equation rests on, as a number.

    Without this term the reservoir has only the 4e-10 /Pa a dead oil
    carries, snaps to the bottom-hole pressure within a year and liberates
    far too much gas.
    """
    base = 4.0e-10
    p = np.array([400.0 * PSI])
    ct = GAS.total_compressibility(p, np.array([0.3]), np.array([0.05]),
                                   np.array([9.0]), base)
    assert ct[0] > 100 * base


# --------------------------------------------------------------- solver
def _slab_producer(gas, bhp_psi=400.0, initial_psi=1500.0, days=1825.0):
    geology = slab(shape=(21, 21, 4), spacing=(50.0, 50.0, 10.0), phi=0.25,
                   k_md=200.0)
    settings = FlowSettings(gravity=False, max_timestep_days=30.0,
                            solution_gas=gas)
    simulator = FlowSimulator(
        geology, [Well(name="P1", role="producer", x=500.0, y=500.0)],
        {"P1": [Completion("slab")]},
        {"P1": WellControl(mode=ControlMode.BHP, target=bhp_psi * PSI)},
        np.full(geology.grid.shape, initial_psi * PSI),
        np.full(geology.grid.shape, settings.relperm.swc), settings)
    return simulator, simulator.run(days, report_every_days=days / 5.0)


def test_dead_oil_run_is_untouched_by_the_gas_machinery():
    _, result = _slab_producer(None)
    assert result.gas_saturation == []
    assert result.solution_gor == []
    assert result.volume_closure_error == 0.0
    assert result.material_balance_error < 1e-6


def test_depletion_below_the_bubble_point_liberates_gas():
    """The question the two-phase model could not answer.

    Dead oil gives exactly zero gas however far the pressure falls; the
    same reservoir with the same drive gives a gas saturation that grows
    with time, which is the whole point.
    """
    _, dead = _slab_producer(None)
    _, live = _slab_producer(GAS)
    assert live.gas_saturation
    peaks = [float(sg.max()) for sg in live.gas_saturation]
    assert peaks[0] == 0.0                     # starts above the bubble point
    assert peaks[-1] > 0.02
    assert all(b >= a - 1e-12 for a, b in zip(peaks, peaks[1:]))
    assert not dead.gas_saturation


def test_liberated_gas_holds_the_pressure_up():
    """A solution-gas drive declines slowly; a dead-oil one collapses.

    This is the single physical claim the gas-aware compressibility makes,
    and it is checked on the same reservoir with the same well.
    """
    _, dead = _slab_producer(None)
    _, live = _slab_producer(GAS)
    mask = np.ones_like(dead.pressure[-1], dtype=bool)
    assert live.pressure[-1][mask].mean() > dead.pressure[-1][mask].mean() + 50 * PSI


def test_material_balance_still_closes_with_gas():
    _, live = _slab_producer(GAS)
    assert live.material_balance_error < 1e-6


def test_gas_is_conserved_by_transport_and_flash():
    """No wells: every standard cubic metre of gas stays in the model.

    Transport moves it between cells at the upstream solution GOR and the
    flash only moves it between the dissolved and free terms, so the total
    is invariant to both.
    """
    geology = slab(shape=(21, 3, 3), spacing=(50.0, 25.0, 10.0))
    settings = FlowSettings(gravity=False, solution_gas=GAS)
    pressure = np.zeros(geology.grid.shape)
    pressure[:] = np.linspace(1500.0, 500.0, geology.grid.shape[0])[:, None, None] * PSI
    simulator = FlowSimulator(geology, [], {}, {}, pressure,
                              np.full(geology.grid.shape, settings.relperm.swc),
                              settings)
    before = float(simulator.gas_std.sum())
    assert before > 0
    for _ in range(20):
        simulator._solve_pressure(5.0 * 86400.0, {})
        simulator._advance_saturation(5.0 * 86400.0, {})
        simulator._advance_solution_gas(5.0 * 86400.0, {})
    assert float(simulator.gas_std.sum()) == pytest.approx(before, rel=1e-9)
    assert float(simulator.sg.max()) > 0.0     # and some of it did come out


def test_producer_reports_a_gas_rate_only_once_gas_is_in_play():
    _, above = _slab_producer(GAS, bhp_psi=1200.0, initial_psi=2000.0, days=365.0)
    _, below = _slab_producer(GAS, bhp_psi=300.0, initial_psi=1500.0, days=1825.0)
    gor_above = above.wells["P1"].producing_gor
    gor_below = below.wells["P1"].producing_gor
    # Above the bubble point the produced gas is exactly what was dissolved.
    assert np.allclose(gor_above[gor_above != 0], GAS.initial_gor, rtol=1e-6)
    assert float(np.max(gor_below)) > 0.0
    assert float(np.min(gor_below[gor_below != 0])) < GAS.initial_gor


def test_state_at_carries_the_simulated_gas_into_the_rock_physics():
    from sim3d.reservoir.state import initial_state
    simulator, result = _slab_producer(GAS)
    baseline = initial_state(simulator.geology, sw=simulator.settings.relperm.swc)
    state = result.state_at(result.days[-1], baseline)
    mask = np.asarray(baseline.reservoir_mask, dtype=bool)
    assert float(state.sg[mask].max()) > 0.0
    assert np.allclose(state.sw + state.so + state.sg, 1.0, atol=1e-9)
    state.validate()
