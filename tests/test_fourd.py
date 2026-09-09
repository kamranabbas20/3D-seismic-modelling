"""4D scenario construction, decomposition and metrics."""

import numpy as np
import pytest

from sim3d.core.errors import ConfigError
from sim3d.core.grid import Grid3D
from sim3d.fourd import (
    SCENARIO_NAMES, build_earth_models, build_states, confining_pressure_from_density,
    decompose, interaction, nrms, rms,
)
from sim3d.fourd.decomposition import describe, nonlinearity_ratio
from sim3d.fourd.metrics import (
    affected_volume, cross_correlation, local_time_shift, overlap_volume,
    predictability, radial_profile, time_shift_volume, well_centred_statistics,
)
from sim3d.geology import build_geology, template
from sim3d.reservoir import (
    GasBreakout, PressureHalo, ReservoirScenario, SaturationFront, initial_state,
)
from sim3d.rockphysics import RockPhysicsConfig
from sim3d.wells import pattern


@pytest.fixture(scope="module")
def experiment():
    grid = Grid3D.from_bounds(((0, 3000), (0, 3000), (0, 2200)), (100, 100, 25))
    layers, faults = template("fault_compartment", throw=45.0, transmissibility=0.0)
    geology = build_geology(grid, layers, faults)
    baseline = initial_state(geology, sw=0.30, temperature=80.0)
    wells = pattern("demonstration")
    scenario = ReservoirScenario(
        "monitor",
        pressure=[PressureHalo("I1", delta_p=+40e5, radius=(600.0, 600.0, 150.0)),
                  PressureHalo("P1", delta_p=-50e5, radius=(500.0, 500.0, 150.0)),
                  PressureHalo("P2", delta_p=-30e5, radius=(450.0, 450.0, 150.0))],
        water_fronts=[SaturationFront("I1", target_sw=0.75, radius=(300.0, 300.0, 90.0)),
                      SaturationFront("I2", target_sw=0.75, radius=(200.0, 200.0, 90.0))],
        gas=[GasBreakout("P3", target_sg=0.10, radius=(250.0, 250.0, 80.0))],
    )
    states = build_states(baseline, scenario, wells, faults)
    earth = build_earth_models(states, geology, RockPhysicsConfig(temperature=80.0))
    return grid, geology, wells, states, earth


# ------------------------------------------------------------------- scenarios
def test_four_distinct_earth_models_are_produced(experiment):
    _, _, _, _, earth = experiment
    assert set(earth.models) == set(SCENARIO_NAMES)
    velocities = [earth.models[name].vp for name in SCENARIO_NAMES]
    for i in range(1, len(velocities)):
        assert not np.array_equal(velocities[0], velocities[i])


def test_scenario_isolation_is_enforced_at_construction(experiment):
    _, _, _, states, _ = experiment
    states.check_isolation()
    broken = build_states.__wrapped__ if hasattr(build_states, "__wrapped__") else None
    assert broken is None  # the check runs inside build_states, not as a decorator


def test_the_pressure_only_model_leaves_density_almost_untouched(experiment):
    """Density changes with pressure only through the pore fluid, so the
    pressure-only case must move rho far less than the saturation-only case,
    where brine replaces oil."""
    _, _, _, _, earth = experiment
    base = earth.rock_physics["baseline"].rho
    d_pressure = np.max(np.abs(earth.rock_physics["pressure_only"].rho - base))
    d_saturation = np.max(np.abs(earth.rock_physics["saturation_only"].rho - base))
    assert d_pressure < 0.1 * d_saturation


def test_water_injection_raises_impedance_and_gas_lowers_it(experiment):
    grid, _, wells, states, earth = experiment
    d_ai = earth.rock_physics["combined"].ai - earth.rock_physics["baseline"].ai
    res = states.baseline.reservoir_mask
    near_injector = well_centred_statistics(grid, wells["I1"], d_ai, radii=(200.0,), mask=res)
    near_gas = well_centred_statistics(grid, wells["P3"], d_ai, radii=(200.0,), mask=res)
    assert near_injector[200.0] > 0
    assert near_gas[200.0] < 0


def test_depletion_stiffens_the_frame(experiment):
    """Producing lowers pore pressure, raising effective stress and Vp."""
    grid, _, wells, states, earth = experiment
    d_vp = earth.rock_physics["pressure_only"].vp - earth.rock_physics["baseline"].vp
    res = states.baseline.reservoir_mask
    assert well_centred_statistics(grid, wells["P1"], d_vp, radii=(200.0,), mask=res)[200.0] > 0
    assert well_centred_statistics(grid, wells["I1"], d_vp, radii=(200.0,), mask=res)[200.0] < 0


def test_confining_pressure_integrates_the_density_column():
    grid = Grid3D((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (3, 3, 11))
    density = np.full(grid.shape, 2000.0)
    p = confining_pressure_from_density(grid, density, surface_pressure=0.0)
    assert p[0, 0, 10] == pytest.approx(2000 * 9.81 * 100.0, rel=1e-9)
    assert p[0, 0, 0] == pytest.approx(0.0, abs=1e-9)
    # A varying column integrates trapezoidally, not by a block sum.
    varying = np.broadcast_to(np.linspace(1800.0, 2600.0, 11), grid.shape).copy()
    q = confining_pressure_from_density(grid, varying, surface_pressure=0.0)
    expected = 9.81 * np.trapezoid(varying[0, 0], dx=grid.dz)
    assert q[0, 0, -1] == pytest.approx(expected, rel=1e-12)


def test_mismatched_density_shape_is_refused():
    grid = Grid3D((0.0,) * 3, (10.0,) * 3, (3, 3, 3))
    with pytest.raises(ConfigError, match="does not match grid"):
        confining_pressure_from_density(grid, np.zeros((4, 4, 4)))


# --------------------------------------------------------------- decomposition
def test_decomposition_is_exact_algebra():
    rng = np.random.default_rng(0)
    base, pres, sat, comb = (rng.standard_normal((4, 5)) for _ in range(4))
    parts = decompose(base, pres, sat, comb)
    assert np.allclose(parts["d_pressure"], pres - base)
    assert np.allclose(parts["d_sum"], parts["d_pressure"] + parts["d_saturation"])
    assert np.allclose(parts["d_interaction"], parts["d_combined"] - parts["d_sum"])
    assert np.allclose(interaction(base, pres, sat, comb), parts["d_interaction"])


def test_a_perfectly_additive_response_has_no_interaction():
    base = np.zeros((3, 3))
    pres, sat = np.full((3, 3), 2.0), np.full((3, 3), 5.0)
    parts = decompose(base, pres, sat, pres + sat)
    assert np.allclose(parts["d_interaction"], 0.0)
    assert nonlinearity_ratio(parts) == pytest.approx(0.0)


def test_mismatched_shapes_are_refused():
    with pytest.raises(ConfigError, match="same shape"):
        decompose(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(4))


def test_the_rock_physics_interaction_is_small_but_not_zero(experiment):
    """Pressure and saturation do not superpose exactly: Gassmann's response
    to a fluid change depends on the frame, which depends on stress."""
    _, _, _, states, earth = experiment
    res = states.baseline.reservoir_mask
    parts = decompose(*[earth.rock_physics[n].ai for n in SCENARIO_NAMES])
    ratio = nonlinearity_ratio(parts, res)
    assert 0.001 < ratio < 0.30
    assert "interaction" in describe(parts, "AI", 1e6, mask=res)


def test_interaction_grows_with_the_size_of_the_perturbation(experiment):
    grid, geology, wells, states, _ = experiment
    ratios = []
    for amplitude in (10e5, 60e5):
        scenario = ReservoirScenario(
            "sweep",
            pressure=[PressureHalo("I1", delta_p=amplitude, radius=(600.0, 600.0, 150.0))],
            water_fronts=[SaturationFront("I1", target_sw=0.80,
                                          radius=(400.0, 400.0, 120.0))])
        s = build_states(states.baseline, scenario, wells)
        e = build_earth_models(s, geology, RockPhysicsConfig(temperature=80.0))
        parts = decompose(*[e.rock_physics[n].ai for n in SCENARIO_NAMES])
        ratios.append(nonlinearity_ratio(parts, states.baseline.reservoir_mask))
    assert ratios[1] > ratios[0]


# --------------------------------------------------------------------- metrics
def test_rms_and_nrms_have_their_definitions():
    a = np.array([3.0, 4.0])
    assert rms(a) == pytest.approx(np.sqrt(12.5))
    assert nrms(a, a) == pytest.approx(0.0)
    assert nrms(a, -a) == pytest.approx(200.0)
    assert nrms(np.zeros(3), np.zeros(3)) == 0.0


def test_cross_correlation_and_predictability():
    a = np.array([1.0, -2.0, 3.0])
    assert cross_correlation(a, a) == pytest.approx(1.0)
    assert cross_correlation(a, -a) == pytest.approx(-1.0)
    assert predictability(a, -a) == pytest.approx(1.0)


def test_time_shift_recovers_a_known_lag():
    dt = 0.002
    t = np.arange(400) * dt
    wavelet = np.exp(-((t - 0.3) ** 2) / 0.0006) * np.sin(2 * np.pi * 25 * (t - 0.3))
    for true_shift in (0.006, -0.010):
        shifted = np.interp(t - true_shift, t, wavelet, left=0.0, right=0.0)
        assert local_time_shift(wavelet, shifted, dt) == pytest.approx(true_shift, abs=0.6 * dt)


def test_time_shift_is_zero_for_identical_traces():
    t = np.arange(200) * 0.002
    w = np.sin(2 * np.pi * 20 * t) * np.exp(-((t - 0.2) ** 2) / 0.002)
    assert local_time_shift(w, w, 0.002) == pytest.approx(0.0, abs=1e-9)
    assert local_time_shift(np.zeros(50), np.zeros(50), 0.002) == 0.0


def test_time_shift_volume_maps_over_traces():
    dt = 0.002
    t = np.arange(300) * dt
    w = np.exp(-((t - 0.25) ** 2) / 0.0006) * np.sin(2 * np.pi * 25 * (t - 0.25))
    base = np.broadcast_to(w, (2, 3, w.size)).copy()
    monitor = np.broadcast_to(np.interp(t - 0.004, t, w, left=0.0, right=0.0),
                              (2, 3, w.size)).copy()
    shifts = time_shift_volume(base, monitor, dt)
    assert shifts.shape == (2, 3)
    assert np.allclose(shifts, 0.004, atol=0.6 * dt)


def test_well_centred_statistics_average_over_growing_cylinders(experiment):
    grid, _, wells, states, earth = experiment
    d_ai = earth.rock_physics["combined"].ai - earth.rock_physics["baseline"].ai
    stats = well_centred_statistics(grid, wells["I1"], d_ai, mask=states.baseline.reservoir_mask)
    assert sorted(stats) == [100.0, 250.0, 500.0]
    assert abs(stats[100.0]) > abs(stats[500.0])  # the anomaly is well-centred


def test_unknown_statistic_is_refused(experiment):
    grid, _, wells, _, earth = experiment
    with pytest.raises(ConfigError, match="unknown statistic"):
        well_centred_statistics(grid, wells["I1"], earth.rock_physics["baseline"].ai,
                                statistic="median")


def test_radial_profiles_decay_away_from_the_well(experiment):
    grid, _, wells, states, earth = experiment
    d_ai = earth.rock_physics["combined"].ai - earth.rock_physics["baseline"].ai
    radii, means, counts = radial_profile(grid, wells["I1"], np.abs(d_ai),
                                          bin_width=100.0, max_radius=900.0,
                                          mask=states.baseline.reservoir_mask)
    valid = counts > 0
    assert np.nanmean(means[valid][:2]) > np.nanmean(means[valid][-2:])


def test_affected_and_overlap_volumes(experiment):
    grid, _, _, states, earth = experiment
    res = states.baseline.reservoir_mask
    base = earth.rock_physics["baseline"].ai
    d_pressure = earth.rock_physics["pressure_only"].ai - base
    d_saturation = earth.rock_physics["saturation_only"].ai - base
    v_pressure = affected_volume(grid, d_pressure, 2e4, res)
    v_saturation = affected_volume(grid, d_saturation, 2e4, res)
    assert v_pressure > v_saturation          # the halo is the bigger body
    overlap = overlap_volume(grid, d_pressure, 2e4, d_saturation, 2e4, res)
    assert 0 < overlap <= min(v_pressure, v_saturation)
