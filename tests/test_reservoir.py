"""Wells, mechanistic reservoir state and the 4D scenario construction."""

import numpy as np
import pytest

from sim3d.core.errors import ConfigError, ValidationError
from sim3d.core.grid import Grid3D
from sim3d.geology import build_geology, template
from sim3d.geology.faults import Fault, FaultSet
from sim3d.reservoir import (
    GasBreakout, PressureHalo, ReservoirScenario, SaturationFront, TIME_STATES,
    initial_state,
)
from sim3d.wells import Well, WellSet, pattern
from sim3d.wells.well import PATTERNS


@pytest.fixture(scope="module")
def setup():
    grid = Grid3D.from_bounds(((0, 3000), (0, 3000), (0, 2200)), (100, 100, 20))
    layers, faults = template("fault_compartment", throw=45.0, transmissibility=0.0)
    geology = build_geology(grid, layers, faults)
    return grid, geology, faults, initial_state(geology, sw=0.30), pattern("demonstration")


# ----------------------------------------------------------------------- wells
def test_well_roles_are_validated():
    with pytest.raises(ConfigError, match="role"):
        Well("W1", "magic", 0.0, 0.0)


def test_perforation_must_increase_downwards():
    with pytest.raises(ConfigError, match="increase downwards"):
        Well("W1", "injector", 0.0, 0.0, perforation=(1300.0, 1200.0))


def test_well_position_is_the_middle_of_the_perforation():
    w = Well("I1", "injector", 100.0, 200.0, perforation=(1200.0, 1400.0))
    assert w.position == (100.0, 200.0, 1300.0)


@pytest.mark.parametrize("name", sorted(PATTERNS))
def test_every_pattern_builds_wells_with_valid_roles(name):
    ws = pattern(name)
    assert len(ws) >= 2
    assert all(w.role in ("injector", "producer", "observation") for w in ws)
    assert "wells" in ws.summary()


def test_the_demonstration_pattern_meets_the_spec_section_29_requirements():
    ws = pattern("demonstration")
    assert len(ws.injectors) == 2 and len(ws.producers) == 3
    assert ws.check_spacing() == []
    d = ws.distance_matrix()
    assert d[~np.eye(len(d), dtype=bool)].min() >= 500.0


def test_close_wells_warn_but_are_not_forbidden():
    ws = WellSet([Well("I1", "injector", 0.0, 0.0), Well("P1", "producer", 200.0, 0.0)])
    warnings = ws.check_spacing()
    assert len(warnings) == 1 and "allowed" in warnings[0]
    assert ws.spacing_violations()[0][2] == pytest.approx(200.0)


# ----------------------------------------------------------------------- state
def test_the_baseline_state_is_valid_and_hydrostatic(setup):
    grid, geology, _, state, _ = setup
    state.validate()
    column = state.pressure[grid.nx // 2, grid.ny // 2, :]
    gradient = np.diff(column) / grid.dz
    assert np.allclose(gradient, 10500.0)


def test_saturations_must_close(setup):
    _, _, _, state, _ = setup
    broken = state.copy()
    broken.sg = broken.sg + 0.2
    with pytest.raises(ValidationError, match="sum to 1"):
        broken.validate()
    assert broken.renormalise().sw.shape == state.sw.shape


def test_out_of_range_saturations_are_refused(setup):
    _, _, _, state, _ = setup
    broken = state.copy()
    broken.sw = broken.sw - 2.0
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        broken.validate()


def test_copying_a_state_does_not_alias_its_arrays(setup):
    _, _, _, state, _ = setup
    other = state.copy()
    other.pressure += 1.0
    assert not np.shares_memory(other.pressure, state.pressure)
    assert np.max(state.pressure - other.pressure) == pytest.approx(-1.0)


# --------------------------------------------------------------- perturbations
def _scenario():
    return ReservoirScenario(
        "monitor",
        pressure=[PressureHalo("I1", delta_p=+40e5, radius=(600.0, 600.0, 150.0)),
                  PressureHalo("P1", delta_p=-50e5, radius=(500.0, 500.0, 150.0))],
        water_fronts=[SaturationFront("I1", target_sw=0.75, radius=(300.0, 300.0, 90.0))],
        gas=[GasBreakout("P3", target_sg=0.10, radius=(250.0, 250.0, 80.0))],
    )


def test_a_monitor_state_stays_physical(setup):
    _, _, faults, base, wells = setup
    monitor = _scenario().apply(base, wells, faults)
    monitor.validate()
    d = monitor.difference(base)
    assert np.allclose(d["dSw"] + d["dSo"] + d["dSg"], 0.0, atol=1e-9)


def test_injection_raises_water_saturation_and_removes_oil(setup):
    _, _, faults, base, wells = setup
    monitor = _scenario().apply(base, wells, faults)
    d = monitor.difference(base)
    assert d["dSw"].max() > 0.4
    assert d["dSo"].min() < -0.4
    assert d["dSw"].min() >= -1e-12  # water never decreases in a waterflood


def test_gas_breakout_appears_only_where_it_was_asked_for(setup):
    grid, _, faults, base, wells = setup
    monitor = _scenario().apply(base, wells, faults)
    d = monitor.difference(base)
    assert d["dSg"].max() == pytest.approx(0.10, abs=0.01)
    x, y, _ = np.meshgrid(*[grid.axis(a) for a in range(3)], indexing="ij")
    far = np.hypot(x - wells["P3"].x, y - wells["P3"].y) > 700.0
    assert np.all(d["dSg"][far] < 1e-6)


def test_the_pressure_halo_reaches_further_than_the_saturation_front(setup):
    """Spec section 32: pressure and saturation are independent, and the
    pressure response normally extends well beyond the flood front."""
    _, _, faults, base, wells = setup
    monitor = _scenario().apply(base, wells, faults)
    d = monitor.difference(base)
    res = base.reservoir_mask
    pressure_cells = np.sum(np.abs(d["dP"][res]) > 1e5)   # 1 bar
    water_cells = np.sum(d["dSw"][res] > 0.01)
    assert pressure_cells > 3 * water_cells


def test_pressure_only_and_saturation_only_are_genuinely_isolated(setup):
    _, _, faults, base, wells = setup
    scenario = _scenario()
    pressure_only = scenario.apply(base, wells, faults, include_saturation=False)
    saturation_only = scenario.apply(base, wells, faults, include_pressure=False)
    assert np.max(np.abs(pressure_only.difference(base)["dSw"])) == 0.0
    assert np.max(np.abs(pressure_only.difference(base)["dSg"])) == 0.0
    assert np.max(np.abs(saturation_only.difference(base)["dP"])) == 0.0


def test_a_sealing_fault_truncates_the_pressure_halo():
    grid = Grid3D.from_bounds(((0, 3000), (0, 3000), (900, 1600)), (50, 50, 20))
    fault = Fault(name="F", origin=(1500.0, 1500.0, 1250.0), strike=0.0, dip=90.0,
                  throw=0.0, transmissibility=0.0, zone_width=30.0)
    layers, _ = template("flat")
    geology = build_geology(grid, layers, FaultSet([fault]))
    base = initial_state(geology, sw=0.30)
    wells = WellSet([Well("I1", "injector", 1100.0, 1500.0, (1200.0, 1350.0))])
    scenario = ReservoirScenario(
        "sealed", pressure=[PressureHalo("I1", delta_p=40e5, radius=(900.0, 900.0, 200.0))])

    sealed = scenario.apply(base, wells, FaultSet([fault]))
    open_case = scenario.apply(base, wells, FaultSet([]))
    x, _, _ = np.meshgrid(*[grid.axis(a) for a in range(3)], indexing="ij")
    across = (x > 1600.0) & base.reservoir_mask

    assert np.max(np.abs(sealed.difference(base)["dP"][across])) < 1e-9
    assert np.max(np.abs(open_case.difference(base)["dP"][across])) > 5e5


def test_a_partially_sealing_fault_attenuates_rather_than_blocks():
    grid = Grid3D.from_bounds(((0, 3000), (0, 3000), (900, 1600)), (50, 50, 20))
    layers, _ = template("flat")
    wells = WellSet([Well("I1", "injector", 1100.0, 1500.0, (1200.0, 1350.0))])
    scenario = ReservoirScenario(
        "leaky", pressure=[PressureHalo("I1", delta_p=40e5, radius=(900.0, 900.0, 200.0))])
    x, _, _ = np.meshgrid(*[grid.axis(a) for a in range(3)], indexing="ij")

    peaks = []
    for transmissibility in (0.0, 0.3, 1.0):
        fault = Fault(name="F", origin=(1500.0, 1500.0, 1250.0), strike=0.0, dip=90.0,
                      throw=0.0, transmissibility=transmissibility, zone_width=30.0)
        faults = FaultSet([fault])
        geology = build_geology(grid, layers, faults)
        base = initial_state(geology, sw=0.30)
        across = (x > 1600.0) & base.reservoir_mask
        peaks.append(np.max(np.abs(scenario.apply(base, wells, faults)
                                   .difference(base)["dP"][across])))
    assert peaks[0] < peaks[1] < peaks[2]
    assert peaks[1] == pytest.approx(0.3 * peaks[2], rel=1e-9)


@pytest.mark.parametrize("state", sorted(TIME_STATES))
def test_pseudo_time_states_grow_monotonically(setup, state):
    _, _, faults, base, wells = setup
    scenario = _scenario()
    scenario.time_state = state
    monitor = scenario.apply(base, wells, faults)
    peak = np.max(np.abs(monitor.difference(base)["dP"]))
    assert peak == pytest.approx(TIME_STATES[state] * 50e5, rel=0.05) or state == "T0"


def test_the_earliest_time_state_produces_no_change(setup):
    _, _, faults, base, wells = setup
    scenario = _scenario()
    scenario.time_state = "T0"
    monitor = scenario.apply(base, wells, faults)
    assert np.max(np.abs(monitor.difference(base)["dP"])) == 0.0


def test_anisotropic_fronts_spread_along_their_azimuth(setup):
    grid, _, faults, base, wells = setup
    scenario = ReservoirScenario(
        "aniso",
        water_fronts=[SaturationFront("I1", target_sw=0.75, azimuth=90.0,
                                      radius=(600.0, 150.0, 90.0))])
    monitor = scenario.apply(base, wells, None)
    d = monitor.difference(base)["dSw"]
    well = wells["I1"]
    x, y, z = np.meshgrid(*[grid.axis(a) for a in range(3)], indexing="ij")
    perforated = well.perforation_mask(z) & base.reservoir_mask
    reach_x = np.abs(x - well.x)[(d > 0.01) & perforated].max()
    reach_y = np.abs(y - well.y)[(d > 0.01) & perforated].max()
    assert reach_x > 2.0 * reach_y


def test_unknown_profiles_and_bad_radii_are_refused():
    with pytest.raises(ConfigError, match="profile"):
        PressureHalo("I1", profile="magic")
    with pytest.raises(ConfigError, match="radii"):
        PressureHalo("I1", radius=(0.0, 100.0, 100.0))
    with pytest.raises(ConfigError, match="target_sw"):
        SaturationFront("I1", target_sw=1.5)
