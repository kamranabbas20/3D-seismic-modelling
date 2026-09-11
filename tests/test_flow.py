"""Flow-simulation benchmarks.

Every test compares against an analytic solution or a conservation law, not
against a stored output of this code.
"""

import numpy as np
import pytest

from sim3d.core.errors import ConfigError, ValidationError
from sim3d.core.grid import Grid3D
from sim3d.core.units import DAY, MILLIDARCY, si_to_stb_per_day
from sim3d.geology.builder import GeologyModel, Layer
from sim3d.geology.faults import Fault, FaultSet
from sim3d.geology.surfaces import Flat
from sim3d.reservoir.flow import FlowSettings, FlowSimulator
from sim3d.reservoir.relperm import CoreyRelativePermeability
from sim3d.wells import Well, WellSet
from sim3d.wells.completion import Completion
from sim3d.wells.controls import ControlMode, WellControl


def slab(shape=(41, 5, 5), spacing=(25.0, 25.0, 10.0), phi=0.25, k_md=500.0,
         faults=None):
    """A homogeneous reservoir block, built directly so the numerics are isolated."""
    grid = Grid3D((0.0, 0.0, 1000.0), spacing, shape)
    ones = np.ones(grid.shape)
    return GeologyModel(
        grid=grid, layer_index=np.zeros(grid.shape, np.int16),
        facies_code=np.ones(grid.shape, np.int16), porosity=phi * ones,
        vsh=0.1 * ones, ntg=ones, permeability=k_md * MILLIDARCY * ones,
        reservoir_mask=np.ones(grid.shape, bool),
        layers=[Layer("slab", Flat(0.0), "clean_sandstone")],
        faults=faults or FaultSet([]), horizons={}, catalogue={})


def build(geology, wells, controls, settings, pressure=2.0e7, sw=None):
    completions = {w.name: [Completion("slab")] for w in wells}
    swc = settings.relperm.swc
    initial = np.full(geology.grid.shape, swc if sw is None else sw)
    return FlowSimulator(geology, wells, completions, controls,
                         np.full(geology.grid.shape, pressure), initial, settings)


# ------------------------------------------------------------- relative perm
def test_corey_endpoints_and_monotonicity():
    rp = CoreyRelativePermeability()
    assert rp.krw(rp.swc) == pytest.approx(0.0)
    assert rp.kro(rp.swc) == pytest.approx(rp.kro_max)
    assert rp.krw(1 - rp.sor) == pytest.approx(rp.krw_max)
    assert rp.kro(1 - rp.sor) == pytest.approx(0.0)
    sw = np.linspace(rp.swc, 1 - rp.sor, 50)
    assert np.all(np.diff(rp.krw(sw)) >= 0)
    assert np.all(np.diff(rp.kro(sw)) <= 0)
    fw = rp.fractional_flow(sw)
    assert fw[0] == pytest.approx(0.0) and fw[-1] == pytest.approx(1.0)
    assert np.all(np.diff(fw) >= -1e-12)


def test_impossible_corey_settings_are_refused():
    with pytest.raises(ConfigError, match="no movable saturation"):
        CoreyRelativePermeability(swc=0.6, sor=0.5)
    with pytest.raises(ConfigError, match="must be positive"):
        CoreyRelativePermeability(nw=0.0)


# ---------------------------------------------------------------- transport
@pytest.mark.slow
def test_buckley_leverett_front_matches_the_analytic_solution():
    r"""The classic check on two-phase transport.

    Welge's tangent construction gives the shock saturation and its speed;
    after injecting ``Q`` pore volumes the front sits at
    ``x/L = Q * dfw/dSw`` evaluated there. First-order upstream weighting
    smears the shock slightly forward, so the numerical front is expected to
    run a little ahead - a few percent, not tens.
    """
    rp = CoreyRelativePermeability(swc=0.2, sor=0.2, krw_max=0.4, kro_max=0.9,
                                   nw=2.0, no=2.0, water_viscosity=1e-3,
                                   oil_viscosity=2e-3)
    sw = np.linspace(rp.swc + 1e-6, 1 - rp.sor - 1e-6, 200_001)
    fw = rp.fractional_flow(sw)
    dfw = np.gradient(fw, sw)
    tangent = int(np.argmin(np.abs(dfw - fw / (sw - rp.swc))[10:-10])) + 10
    slope = dfw[tangent]

    geology = slab(shape=(201, 2, 2), spacing=(5.0, 20.0, 10.0))
    grid = geology.grid
    length = grid.extent[0]
    rate = 1.0e-4
    wells = WellSet([Well("INJ", "injector", 0.0, 10.0, (1000.0, 1010.0)),
                     Well("PRO", "producer", length, 10.0, (1000.0, 1010.0))])
    controls = {"INJ": WellControl(ControlMode.WATER_RATE, rate),
                "PRO": WellControl(ControlMode.LIQUID_RATE, rate)}
    settings = FlowSettings(relperm=rp, total_compressibility=1e-12, gravity=False,
                            max_saturation_change=0.01, max_timestep_days=2.0)
    simulator = build(geology, wells, controls, settings)

    pore_volume = float(simulator.pore_volume.sum())
    days = 0.35 * pore_volume / rate / DAY
    result = simulator.run(days, report_every_days=days)

    _, saturation = result.at(days)
    profile = saturation[:, 0, 0]
    injected = rate * days * DAY / pore_volume
    analytic = injected * slope
    moved = np.flatnonzero(profile > rp.swc + 0.02)
    numeric = (grid.axis(0)[moved[-1]] + 0.5 * grid.dx) / length

    assert numeric == pytest.approx(analytic, rel=0.06)
    assert numeric >= analytic - 1e-9      # smearing runs forward, never back
    assert profile[2] > sw[tangent]        # plateau behind the shock
    assert result.material_balance_error < 1e-8


# ------------------------------------------------------------ conservation
def test_material_balance_closes():
    geology = slab()
    wells = WellSet([Well("I1", "injector", 100.0, 50.0, (1000.0, 1040.0)),
                     Well("P1", "producer", 900.0, 50.0, (1000.0, 1040.0))])
    controls = {"I1": WellControl(ControlMode.WATER_RATE, 2.0e-3),
                "P1": WellControl(ControlMode.LIQUID_RATE, 2.0e-3)}
    result = build(geology, wells, controls, FlowSettings()).run(
        180.0, report_every_days=60.0)
    assert result.material_balance_error < 1e-8


def test_a_closed_tank_depletes_at_the_rate_compressibility_demands():
    r"""With no injection, ``dp/dt = -q / (c_t V_p)`` for the field average.

    This is the check that the storage term is right; get it wrong and every
    pressure-driven 4D signal is wrong by the same factor.
    """
    geology = slab()
    rate = 1.0e-3
    settings = FlowSettings(total_compressibility=5.0e-10, gravity=False)
    wells = WellSet([Well("P1", "producer", 500.0, 50.0, (1000.0, 1040.0))])
    controls = {"P1": WellControl(ControlMode.LIQUID_RATE, rate)}
    simulator = build(geology, wells, controls, settings)
    pore_volume = float(simulator.pore_volume.sum())

    days = 100.0
    result = simulator.run(days, report_every_days=days)
    start, _ = result.at(0.0)
    end, _ = result.at(days)
    mask = geology.reservoir_mask
    measured = float(end[mask].mean() - start[mask].mean())
    expected = -rate * days * DAY / (settings.total_compressibility * pore_volume)
    assert measured == pytest.approx(expected, rel=0.02)


def test_a_reservoir_in_gravity_equilibrium_stays_there():
    """Nothing moves without wells - provided the initial state is in
    equilibrium.

    At connate water the mobile phase is oil alone, so equilibrium is the
    oil gradient. Starting from a uniform pressure instead, the solver
    correctly relaxes towards that gradient, which is a check on the gravity
    term rather than a failure of it.
    """
    geology = slab()
    grid = geology.grid
    settings = FlowSettings()
    z = grid.axis(2)[None, None, :] * np.ones(grid.shape)
    hydrostatic = 2.0e7 + settings.oil_density * 9.81 * (z - grid.axis(2)[0])

    simulator = build(geology, WellSet([]), {}, settings)
    simulator.p = hydrostatic[geology.reservoir_mask].copy()
    result = simulator.run(90.0, report_every_days=45.0)
    start_p, start_s = result.at(0.0)
    end_p, end_s = result.at(90.0)
    assert np.allclose(end_p, start_p, atol=1.0)      # within a pascal
    assert np.allclose(end_s, start_s, atol=1e-12)


def test_a_uniform_pressure_relaxes_to_the_gravity_gradient():
    geology = slab()
    settings = FlowSettings()
    result = build(geology, WellSet([]), {}, settings).run(
        365.0, report_every_days=365.0)
    end, _ = result.at(365.0)
    column = end[20, 2, :]
    gradient = np.diff(column) / geology.grid.dz
    assert np.allclose(gradient, settings.oil_density * 9.81, rtol=0.02)


# ------------------------------------------------------------------- wells
def test_injection_raises_pressure_and_production_lowers_it():
    geology = slab()
    wells = WellSet([Well("I1", "injector", 100.0, 50.0, (1000.0, 1040.0)),
                     Well("P1", "producer", 900.0, 50.0, (1000.0, 1040.0))])
    controls = {"I1": WellControl(ControlMode.WATER_RATE, 2.0e-3),
                "P1": WellControl(ControlMode.LIQUID_RATE, 2.0e-3)}
    result = build(geology, wells, controls, FlowSettings(gravity=False)).run(
        120.0, report_every_days=120.0)
    start, _ = result.at(0.0)
    end, _ = result.at(120.0)
    change = end - start
    assert change[2, 2, 2] > 0     # near the injector
    assert change[-3, 2, 2] < 0    # near the producer


def test_a_bhp_limit_overrides_the_rate_target():
    """A rate a well cannot deliver must become the pressure it can hold."""
    geology = slab(k_md=5.0)
    wells = WellSet([Well("P1", "producer", 500.0, 50.0, (1000.0, 1040.0))])
    limit = 1.5e7
    controls = {"P1": WellControl(ControlMode.LIQUID_RATE, 1.0, bhp_limit=limit)}
    result = build(geology, wells, controls, FlowSettings()).run(
        30.0, report_every_days=30.0)
    history = result.wells["P1"]
    assert "bhp" in history.control
    assert np.nanmax(history.arrays()["bhp"]) <= limit * 1.001
    assert abs(history.liquid_rate[-1]) < 1.0


def test_material_balance_survives_wells_switching_control_mode():
    """The case that broke it: a rate target the well cannot hold.

    When a well switches to pressure control mid-run, the rate the pressure
    solve honours and the rate the saturation update applies must be the
    same one. Getting that wrong leaks a few percent of throughput - and
    pinning only the *mode* without the pressure is worse still, because the
    well then reads its rate target as a bottom-hole pressure.
    """
    geology = slab(k_md=20.0)
    wells = WellSet([Well("I1", "injector", 100.0, 50.0, (1000.0, 1040.0)),
                     Well("P1", "producer", 900.0, 50.0, (1000.0, 1040.0))])
    controls = {
        "I1": WellControl(ControlMode.WATER_RATE, 5.0e-3, bhp_limit=2.6e7),
        "P1": WellControl(ControlMode.LIQUID_RATE, 5.0e-3, bhp_limit=1.4e7),
    }
    result = build(geology, wells, controls, FlowSettings()).run(
        400.0, report_every_days=100.0)

    modes = {mode for history in result.wells.values() for mode in history.control}
    assert "bhp" in modes, "the test needs a well to actually hit its limit"
    assert result.material_balance_error < 1e-8


def test_a_schedule_keeps_a_well_shut_until_its_start_day():
    geology = slab()
    wells = WellSet([Well("P1", "producer", 500.0, 50.0, (1000.0, 1040.0))])
    controls = {"P1": WellControl(ControlMode.LIQUID_RATE, 1.0e-3, start_day=60.0)}
    result = build(geology, wells, controls, FlowSettings(gravity=False)).run(
        120.0, report_every_days=30.0)
    history = result.wells["P1"]
    days = np.asarray(history.days)
    rate = np.abs(history.liquid_rate)
    assert np.all(rate[days < 60.0] == 0.0)
    assert np.any(rate[days > 60.0] > 0.0)


def test_water_cut_rises_after_breakthrough():
    geology = slab(shape=(41, 3, 3), spacing=(25.0, 25.0, 10.0))
    wells = WellSet([Well("I1", "injector", 0.0, 25.0, (1000.0, 1020.0)),
                     Well("P1", "producer", 1000.0, 25.0, (1000.0, 1020.0))])
    rate = 3.0e-3
    controls = {"I1": WellControl(ControlMode.WATER_RATE, rate),
                "P1": WellControl(ControlMode.LIQUID_RATE, rate)}
    simulator = build(geology, wells, controls,
                      FlowSettings(gravity=False, max_timestep_days=10.0))
    # Long enough to inject well over a pore volume, so breakthrough is certain.
    days = 1.4 * float(simulator.pore_volume.sum()) / rate / DAY
    result = simulator.run(days, report_every_days=days / 12)
    cut = result.wells["P1"].water_cut
    assert cut[0] < 0.02
    assert cut[-1] > 0.5
    assert np.all(np.diff(cut) > -0.05)      # rises, and does not oscillate


def test_cumulative_volumes_are_consistent_with_the_rates():
    geology = slab()
    rate = 1.0e-3
    wells = WellSet([Well("P1", "producer", 500.0, 50.0, (1000.0, 1040.0))])
    controls = {"P1": WellControl(ControlMode.LIQUID_RATE, rate)}
    result = build(geology, wells, controls, FlowSettings()).run(
        100.0, report_every_days=50.0)
    history = result.wells["P1"]
    produced = history.cumulative("oil")[-1] + history.cumulative("water")[-1]
    assert produced == pytest.approx(rate * 100.0 * DAY, rel=0.02)


def test_a_well_with_no_open_completion_is_refused():
    from sim3d.wells.controls import inflow_properties

    geology = slab()
    well = Well("P1", "producer", 500.0, 50.0, (1000.0, 1040.0))
    with pytest.raises(ConfigError, match="no open completion"):
        inflow_properties(well, geology, [], 300.0, 2.0e7)


# ------------------------------------------------------------------ faults
def test_a_sealing_fault_stops_the_pressure_reaching_the_far_block():
    geology_open = slab(shape=(41, 5, 5))
    fault = Fault(name="F", origin=(500.0, 50.0, 1020.0), strike=0.0, dip=90.0,
                  throw=0.0, transmissibility=0.0, zone_width=20.0)
    geology_sealed = slab(shape=(41, 5, 5), faults=FaultSet([fault]))

    wells = WellSet([Well("I1", "injector", 200.0, 50.0, (1000.0, 1040.0))])
    controls = {"I1": WellControl(ControlMode.WATER_RATE, 1.0e-3)}
    changes = []
    for geology in (geology_sealed, geology_open):
        result = build(geology, wells, controls, FlowSettings(gravity=False)).run(
            60.0, report_every_days=60.0)
        start, _ = result.at(0.0)
        end, _ = result.at(60.0)
        changes.append(float((end - start)[-3, 2, 2]))    # far side of the fault
    assert changes[0] < 0.02 * changes[1]


def test_a_partially_sealing_fault_lets_some_pressure_through():
    wells = WellSet([Well("I1", "injector", 200.0, 50.0, (1000.0, 1040.0))])
    controls = {"I1": WellControl(ControlMode.WATER_RATE, 1.0e-3)}
    far = []
    for transmissibility in (0.0, 0.05, 1.0):
        fault = Fault(name="F", origin=(500.0, 50.0, 1020.0), strike=0.0, dip=90.0,
                      throw=0.0, transmissibility=transmissibility, zone_width=20.0)
        result = build(slab(faults=FaultSet([fault])), wells, controls,
                       FlowSettings(gravity=False)).run(60.0, report_every_days=60.0)
        start, _ = result.at(0.0)
        end, _ = result.at(60.0)
        far.append(float((end - start)[-3, 2, 2]))
    assert far[0] < far[1] < far[2]


# ----------------------------------------------------------------- gravity
def test_gravity_segregates_the_flood_downwards():
    """Water is denser than oil, so an unconfined flood underruns."""
    geology = slab(shape=(21, 5, 9), spacing=(25.0, 25.0, 5.0))
    wells = WellSet([Well("I1", "injector", 100.0, 50.0, (1000.0, 1040.0)),
                     Well("P1", "producer", 400.0, 50.0, (1000.0, 1040.0))])
    rate = 1.5e-3
    controls = {"I1": WellControl(ControlMode.WATER_RATE, rate),
                "P1": WellControl(ControlMode.LIQUID_RATE, rate)}
    z = geology.grid.axis(2)
    centroids = []
    for gravity in (False, True):
        simulator = build(geology, wells, controls,
                          FlowSettings(gravity=gravity, max_timestep_days=5.0))
        days = 0.3 * float(simulator.pore_volume.sum()) / rate / DAY
        result = simulator.run(days, report_every_days=days)
        start, _ = result.at(0.0)
        _, saturation = result.at(days)
        _, initial = result.at(0.0)
        moved = saturation - initial
        assert moved.max() > 0.1, "the flood did not develop"
        # Depth centroid of the injected water.
        weight = moved.sum(axis=(0, 1))
        centroids.append(float(np.sum(weight * z) / np.sum(weight)))
    flat, segregated = centroids
    middle = 0.5 * (z[0] + z[-1])
    assert flat == pytest.approx(middle, abs=0.1 * (z[-1] - z[0]))
    assert segregated > flat + 0.05 * (z[-1] - z[0])   # water sits deeper


def test_state_at_returns_a_valid_reservoir_state():
    from sim3d.reservoir.state import ReservoirState

    geology = slab()
    grid = geology.grid
    baseline = ReservoirState(
        grid=grid, porosity=geology.porosity, ntg=geology.ntg, vsh=geology.vsh,
        pressure=np.full(grid.shape, 2.0e7), sw=np.full(grid.shape, 0.25),
        so=np.full(grid.shape, 0.70), sg=np.full(grid.shape, 0.05),
        reservoir_mask=geology.reservoir_mask, name="baseline")
    wells = WellSet([Well("I1", "injector", 100.0, 50.0, (1000.0, 1040.0))])
    controls = {"I1": WellControl(ControlMode.WATER_RATE, 1.0e-3)}
    simulator = build(geology, wells, controls, FlowSettings(), sw=0.25)
    result = simulator.run(60.0, report_every_days=30.0)

    state = result.state_at(60.0, baseline)
    state.validate()
    # The flow model has no gas phase, so gas must be left exactly alone.
    assert np.array_equal(state.sg, baseline.sg)
    assert np.any(state.sw > baseline.sw)
