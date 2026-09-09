"""Rock-physics benchmarks (spec section 130).

Every formula is checked against a closed-form limit, a published bound,
or a literature value - not against a stored output of this code.
"""

import numpy as np
import pytest

from sim3d.core.errors import ConfigError, ValidationError
from sim3d.rockphysics.dryframe import (
    critical_porosity_model, hertz_mindlin, soft_sand, stiff_sand,
)
from sim3d.rockphysics.fluids import (
    brine_properties, check_validity, gas_properties, mix_fluids, oil_properties,
)
from sim3d.rockphysics.gassmann import (
    gassmann_saturated_modulus, gassmann_substitute, velocities,
)
from sim3d.rockphysics.minerals import (
    MINERALS, hashin_shtrikman, mixed_mineral, reuss, voigt, vrh,
)
from sim3d.rockphysics.model import RockPhysicsConfig, elastic_from_state
from sim3d.rockphysics.pressure import PressureModel, effective_stress

GPA = 1e9
QUARTZ, CLAY = MINERALS["quartz"], MINERALS["clay"]


def as_scalar(value):
    """Unwrap a scalar or single-element array to a Python float."""
    return float(np.asarray(value).reshape(-1)[0])


# --------------------------------------------------------------------- mixing
def test_a_pure_phase_reproduces_itself_under_every_law():
    for law in (voigt, reuss, vrh):
        assert law([QUARTZ.k, CLAY.k], [1.0, 0.0]) == pytest.approx(QUARTZ.k)


def test_voigt_and_reuss_have_their_closed_forms():
    f = [0.7, 0.3]
    assert voigt([37.0, 21.0], f) == pytest.approx(0.7 * 37 + 0.3 * 21)
    assert reuss([37.0, 21.0], f) == pytest.approx(1.0 / (0.7 / 37 + 0.3 / 21))


@pytest.mark.parametrize("f_clay", [0.1, 0.3, 0.5, 0.9])
def test_bounds_are_correctly_ordered(f_clay):
    """Reuss <= HS- <= HS+ <= Voigt, and Reuss <= VRH <= Voigt, for K and mu."""
    frac = [1.0 - f_clay, f_clay]
    k, mu = [QUARTZ.k, CLAY.k], [QUARTZ.mu, CLAY.mu]
    k_hsl, mu_hsl = hashin_shtrikman(k, mu, frac, "lower")
    k_hsu, mu_hsu = hashin_shtrikman(k, mu, frac, "upper")
    for moduli, hsl, hsu in ((k, k_hsl, k_hsu), (mu, mu_hsl, mu_hsu)):
        assert reuss(moduli, frac) <= hsl <= hsu <= voigt(moduli, frac)
        assert reuss(moduli, frac) <= vrh(moduli, frac) <= voigt(moduli, frac)


def test_vrh_is_an_estimate_and_can_fall_outside_the_hashin_shtrikman_bounds():
    """VRH is the mean of two wide bounds, so it carries no bound status.

    At 90% clay the shear VRH sits just above the HS upper bound. Anything
    that relies on VRH being bracketed by HS would be relying on luck.
    """
    frac = [0.1, 0.9]
    mu = [QUARTZ.mu, CLAY.mu]
    _, mu_hsu = hashin_shtrikman([QUARTZ.k, CLAY.k], mu, frac, "upper")
    assert vrh(mu, frac) > mu_hsu


def test_hashin_shtrikman_bounds_are_tighter_than_voigt_reuss():
    frac = [0.5, 0.5]
    k, mu = [QUARTZ.k, CLAY.k], [QUARTZ.mu, CLAY.mu]
    k_hsl, _ = hashin_shtrikman(k, mu, frac, "lower")
    k_hsu, _ = hashin_shtrikman(k, mu, frac, "upper")
    assert (k_hsu - k_hsl) < 0.6 * (voigt(k, frac) - reuss(k, frac))


def test_fractions_that_do_not_sum_to_one_are_rejected():
    with pytest.raises(ValidationError, match="sum to 1"):
        voigt([37.0, 21.0], [0.7, 0.5])


def test_unknown_minerals_are_named_not_approximated():
    with pytest.raises(ConfigError, match="unknown mineral"):
        mixed_mineral({"unobtainium": 1.0})


def test_density_always_mixes_linearly():
    _, _, rho = mixed_mineral({"quartz": 0.8, "clay": 0.2}, "hs_upper")
    assert rho == pytest.approx(0.8 * QUARTZ.rho + 0.2 * CLAY.rho)


# --------------------------------------------------------------------- fluids
def test_pure_water_matches_the_measured_sound_speed():
    """Batzle-Wang eq 28 at 20 degC, atmospheric: 1482 m/s, 998 kg/m^3."""
    w = brine_properties(0.1e6, 20.0, salinity=0.0)
    assert as_scalar(w.velocity) == pytest.approx(1482.0, abs=3.0)
    assert as_scalar(w.rho) == pytest.approx(998.0, abs=3.0)


def test_seawater_matches_its_measured_sound_speed():
    sea = brine_properties(0.1e6, 20.0, salinity=35000.0)
    assert as_scalar(sea.velocity) == pytest.approx(1521.0, abs=5.0)
    assert as_scalar(sea.rho) == pytest.approx(1024.0, abs=5.0)


def test_brine_stiffens_with_pressure_and_softens_with_temperature():
    base = brine_properties(20e6, 60.0, 35000.0)
    assert float(brine_properties(40e6, 60.0, 35000.0).k) > as_scalar(base.k)
    assert float(brine_properties(20e6, 120.0, 35000.0).k) < as_scalar(base.k)


def test_gas_is_far_softer_and_lighter_than_brine():
    gas = gas_properties(30e6, 80.0, 0.65)
    brine = brine_properties(30e6, 80.0, 35000.0)
    assert as_scalar(gas.k) < 0.05 * as_scalar(brine.k)
    assert as_scalar(gas.rho) < 0.3 * as_scalar(brine.rho)
    assert 0 < as_scalar(gas.k) and 0 < as_scalar(gas.rho)


def test_gas_density_and_modulus_rise_with_pressure():
    low, high = gas_properties(10e6, 80.0), gas_properties(40e6, 80.0)
    assert as_scalar(high.rho) > as_scalar(low.rho)
    assert as_scalar(high.k) > as_scalar(low.k)


def test_oil_gets_lighter_and_softer_as_gas_dissolves_into_it():
    dead = oil_properties(30e6, 80.0, api=30.0, gor=0.0)
    live = oil_properties(30e6, 80.0, api=30.0, gor=150.0)
    assert as_scalar(live.rho) < as_scalar(dead.rho)
    assert as_scalar(live.k) < as_scalar(dead.k)


def test_heavier_oil_has_a_higher_density_and_modulus():
    heavy = oil_properties(30e6, 80.0, api=15.0, gor=0.0)
    light = oil_properties(30e6, 80.0, api=45.0, gor=0.0)
    assert as_scalar(heavy.rho) > as_scalar(light.rho)
    assert as_scalar(heavy.k) > as_scalar(light.k)


def test_validity_warnings_are_raised_outside_the_fitted_range():
    assert check_validity(30e6, 80.0, 35000.0) == []
    assert any("temperature" in w for w in check_validity(30e6, 500.0, 35000.0))
    assert any("pressure" in w for w in check_validity(0.5e6, 80.0))


def test_wood_mixing_collapses_the_fluid_modulus_with_a_little_gas():
    """A few percent of gas halves the fluid modulus - the whole reason a
    gas 4D anomaly is detectable when the equivalent oil change is not."""
    brine = brine_properties(30e6, 80.0, 50000.0)
    gas = gas_properties(30e6, 80.0)
    k = [float(mix_fluids({"brine": brine, "gas": gas},
                          {"brine": 1 - sg, "gas": sg}).k) for sg in (0.0, 0.02, 0.10)]
    assert k[1] < 0.65 * k[0]
    assert k[2] < 0.25 * k[0]
    assert k[0] > k[1] > k[2]


def test_brie_mixing_is_stiffer_than_wood_at_low_gas_saturation():
    brine = brine_properties(30e6, 80.0, 50000.0)
    gas = gas_properties(30e6, 80.0)
    sat = {"brine": 0.95, "gas": 0.05}
    phases = {"brine": brine, "gas": gas}
    wood = float(mix_fluids(phases, sat, "wood").k)
    brie = float(mix_fluids(phases, sat, "brie", brie_exponent=3.0).k)
    linear = float(mix_fluids(phases, sat, "brie", brie_exponent=1.0).k)
    assert wood < brie < linear


def test_density_mixes_linearly_whatever_the_modulus_law():
    brine = brine_properties(30e6, 80.0, 50000.0)
    gas = gas_properties(30e6, 80.0)
    sat = {"brine": 0.7, "gas": 0.3}
    expected = 0.7 * as_scalar(brine.rho) + 0.3 * as_scalar(gas.rho)
    for model in ("wood", "brie"):
        assert float(mix_fluids({"brine": brine, "gas": gas}, sat, model).rho) \
            == pytest.approx(expected)


def test_saturations_must_close():
    brine = brine_properties(30e6, 80.0)
    gas = gas_properties(30e6, 80.0)
    with pytest.raises(ValidationError, match="sum to 1"):
        mix_fluids({"brine": brine, "gas": gas}, {"brine": 0.7, "gas": 0.4})


# ------------------------------------------------------------------ dry frame
def test_hertz_mindlin_stiffens_as_the_cube_root_of_effective_stress():
    k1, mu1 = hertz_mindlin(QUARTZ.k, QUARTZ.mu, 10e6)
    k2, mu2 = hertz_mindlin(QUARTZ.k, QUARTZ.mu, 80e6)
    assert float(k2 / k1) == pytest.approx(8.0 ** (1.0 / 3.0), rel=1e-6)
    assert float(mu2 / mu1) == pytest.approx(8.0 ** (1.0 / 3.0), rel=1e-6)


def test_hertz_mindlin_refuses_a_non_positive_effective_stress():
    with pytest.raises(ValidationError, match="strictly positive"):
        hertz_mindlin(QUARTZ.k, QUARTZ.mu, 0.0)


def test_dry_frame_softens_with_porosity_and_stays_below_the_mineral():
    phi = np.array([0.05, 0.15, 0.25, 0.35])
    for model in (soft_sand, stiff_sand):
        k, mu = model(QUARTZ.k, QUARTZ.mu, phi, 25e6)
        assert np.all(np.diff(k) < 0)
        assert np.all(np.diff(mu) < 0)
        assert np.all(k < QUARTZ.k) and np.all(mu < QUARTZ.mu)


def test_stiff_sand_is_stiffer_than_soft_sand_at_the_same_porosity():
    phi = np.array([0.10, 0.20, 0.30])
    k_soft, _ = soft_sand(QUARTZ.k, QUARTZ.mu, phi, 25e6)
    k_stiff, _ = stiff_sand(QUARTZ.k, QUARTZ.mu, phi, 25e6)
    assert np.all(k_stiff > k_soft)


def test_stiff_sand_is_less_pressure_sensitive_than_soft_sand():
    """The choice between the two changes the pressure half of a 4D signal."""
    phi = np.array([0.25])
    soft = [soft_sand(QUARTZ.k, QUARTZ.mu, phi, p)[0] for p in (15e6, 35e6)]
    stiff = [stiff_sand(QUARTZ.k, QUARTZ.mu, phi, p)[0] for p in (15e6, 35e6)]
    assert as_scalar(soft[1]) / as_scalar(soft[0]) > as_scalar(stiff[1]) / as_scalar(stiff[0])


def test_critical_porosity_model_is_linear_and_pressure_independent():
    k, mu = critical_porosity_model(QUARTZ.k, QUARTZ.mu, np.array([0.0, 0.2, 0.4]),
                                    critical_porosity=0.4)
    assert k == pytest.approx([QUARTZ.k, QUARTZ.k * 0.5, 0.0])
    assert mu == pytest.approx([QUARTZ.mu, QUARTZ.mu * 0.5, 0.0])


# ------------------------------------------------------------------- Gassmann
def test_gassmann_saturated_modulus_exceeds_the_dry_frame():
    k_sat = gassmann_saturated_modulus(10 * GPA, 8 * GPA, 37 * GPA, 2.5 * GPA, 0.25)
    assert 10 * GPA < as_scalar(k_sat) < 37 * GPA


def test_a_stiffer_fluid_gives_a_stiffer_rock():
    args = (10 * GPA, 8 * GPA, 37 * GPA)
    soft = gassmann_saturated_modulus(*args, 0.05 * GPA, 0.25)
    stiff = gassmann_saturated_modulus(*args, 2.5 * GPA, 0.25)
    assert as_scalar(stiff) > as_scalar(soft)


def test_gassmann_round_trips_through_its_own_inverse():
    k_dry_true, mu, k_min, phi = 10 * GPA, 8 * GPA, 37 * GPA, 0.25
    k_sat0 = gassmann_saturated_modulus(k_dry_true, mu, k_min, 2.5 * GPA, phi)
    k_sat1, k_dry = gassmann_substitute(k_sat0, mu, k_min, 2.5 * GPA, 0.8 * GPA, phi)
    assert as_scalar(k_dry) == pytest.approx(k_dry_true, rel=1e-9)
    assert as_scalar(k_sat1) == pytest.approx(
        float(gassmann_saturated_modulus(k_dry_true, mu, k_min, 0.8 * GPA, phi)), rel=1e-9
    )


def test_gassmann_rejects_a_frame_stiffer_than_its_mineral():
    with pytest.raises(ValidationError, match="stiffer than its own mineral"):
        gassmann_saturated_modulus(40 * GPA, 8 * GPA, 37 * GPA, 2.5 * GPA, 0.25)


def test_velocities_follow_their_definitions():
    vp, vs = velocities(10 * GPA, 8 * GPA, 2400.0)
    assert as_scalar(vp) == pytest.approx(np.sqrt((10e9 + 4 * 8e9 / 3) / 2400))
    assert as_scalar(vs) == pytest.approx(np.sqrt(8e9 / 2400))


# ------------------------------------------------------------------- pressure
def test_effective_stress_definition():
    assert effective_stress(55e6, 30e6) == pytest.approx(25e6)
    assert effective_stress(55e6, 30e6, biot=0.8) == pytest.approx(31e6)


def test_the_none_model_removes_all_pressure_sensitivity():
    model = PressureModel(model="none")
    f_k, f_mu = model.multipliers(np.array([5e6, 50e6]))
    assert np.allclose(f_k, 1.0) and np.allclose(f_mu, 1.0)


def test_the_empirical_curve_stiffens_with_effective_stress():
    model = PressureModel(model="empirical")
    f_k, _ = model.multipliers(np.array([5e6, 20e6, 60e6]))
    assert np.all(np.diff(f_k) > 0)
    assert float(model.multipliers(np.array([20e6]))[0][0]) == pytest.approx(1.0)


def test_a_calibration_table_is_not_extrapolated():
    model = PressureModel(model="table", table_pressure=np.array([10e6, 40e6]),
                          table_k_multiplier=np.array([0.8, 1.2]),
                          reference_pressure=25e6)
    assert model.multipliers(np.array([25e6]))[0] == pytest.approx(1.0)
    with pytest.raises(ValidationError, match="Extend the table"):
        model.multipliers(np.array([80e6]))


# ---------------------------------------------------------------- full chain
def _state(**over):
    base = dict(
        porosity=np.array([0.25]),
        composition={"quartz": np.array([0.8]), "clay": np.array([0.2])},
        saturations={"brine": np.array([0.3]), "oil": np.array([0.7])},
        pore_pressure=np.array([30e6]),
        confining_pressure=np.array([55e6]),
        config=RockPhysicsConfig(temperature=80.0),
    )
    base.update(over)
    return base


def test_the_full_chain_gives_plausible_reservoir_sand_properties():
    r = elastic_from_state(**_state())
    assert 2000 < as_scalar(r.vp) < 3500
    assert 1.4 < as_scalar(r.vp_vs) < 2.4
    assert 2000 < as_scalar(r.rho) < 2500
    assert as_scalar(r.k_dry) < as_scalar(r.k_sat) < as_scalar(r.k_mineral)
    assert as_scalar(r.mu_sat) == as_scalar(r.mu_dry)  # Gassmann leaves shear alone


def test_water_replacing_oil_raises_impedance():
    oil = elastic_from_state(**_state())
    water = elastic_from_state(**_state(
        saturations={"brine": np.array([0.9]), "oil": np.array([0.1])}))
    assert as_scalar(water.ai) > as_scalar(oil.ai)
    assert as_scalar(water.vp) > as_scalar(oil.vp)


def test_a_little_free_gas_drops_impedance_and_does_so_most_against_brine():
    """5% gas is visible against oil and much louder against brine.

    Gas exsolving from a brine-filled rock removes a 2.8 GPa fluid; from an
    oil-filled rock it removes a 0.85 GPa one, so the same gas saturation
    produces a much smaller impedance drop. Any detectability threshold
    quoted for "gas" without saying what it displaced is meaningless.
    """
    oil = elastic_from_state(**_state())
    brine = elastic_from_state(**_state(saturations={"brine": np.array([1.0])}))
    gassy_oil = elastic_from_state(**_state(
        saturations={"brine": np.array([0.3]), "oil": np.array([0.65]),
                     "gas": np.array([0.05])}))
    gassy_brine = elastic_from_state(**_state(
        saturations={"brine": np.array([0.95]), "gas": np.array([0.05])}))

    drop_oil = 1.0 - as_scalar(gassy_oil.ai) / as_scalar(oil.ai)
    drop_brine = 1.0 - as_scalar(gassy_brine.ai) / as_scalar(brine.ai)
    assert drop_oil > 0.02
    assert drop_brine > 2.0 * drop_oil


def test_raising_pore_pressure_softens_the_rock_and_lowers_vp():
    base = elastic_from_state(**_state())
    pumped = elastic_from_state(**_state(pore_pressure=np.array([35e6])))
    assert as_scalar(pumped.vp) < as_scalar(base.vp)
    assert as_scalar(pumped.k_dry) < as_scalar(base.k_dry)
    assert as_scalar(pumped.effective_pressure) < as_scalar(base.effective_pressure)


def test_shear_velocity_responds_to_pressure_but_not_to_fluid():
    base = elastic_from_state(**_state())
    fluid_change = elastic_from_state(**_state(
        saturations={"brine": np.array([0.9]), "oil": np.array([0.1])}))
    pressure_change = elastic_from_state(**_state(pore_pressure=np.array([35e6])))
    # Vs moves only through density when the fluid changes...
    assert as_scalar(fluid_change.mu_sat) == pytest.approx(as_scalar(base.mu_sat))
    # ...but the frame itself softens under pressure.
    assert as_scalar(pressure_change.mu_sat) < as_scalar(base.mu_sat)


def test_pore_pressure_above_the_confining_stress_is_refused():
    with pytest.raises(ValidationError, match="fracture territory"):
        elastic_from_state(**_state(pore_pressure=np.array([60e6])))


def test_the_shallow_section_is_floored_and_the_floor_is_reported():
    """At the free surface both pressures are atmospheric, so the true
    effective stress is zero and every grain-contact model degenerates.
    The floor is applied and counted, never applied quietly."""
    result = elastic_from_state(**_state(
        pore_pressure=np.array([0.2e6]), confining_pressure=np.array([0.4e6])))
    assert as_scalar(result.effective_pressure) == pytest.approx(
        result.config.min_effective_pressure)
    assert any("floor" in w for w in result.warnings)

    with pytest.raises(ValidationError, match="strictly positive"):
        elastic_from_state(**_state(
            pore_pressure=np.array([0.4e6]), confining_pressure=np.array([0.4e6]),
            config=RockPhysicsConfig(temperature=80.0, min_effective_pressure=0.0)))


def test_the_chain_works_on_a_whole_3d_volume():
    shape = (4, 5, 6)
    rng = np.random.default_rng(0)
    phi = rng.uniform(0.15, 0.30, shape)
    vsh = rng.uniform(0.0, 0.4, shape)
    sw = rng.uniform(0.2, 0.6, shape)
    r = elastic_from_state(
        porosity=phi, composition={"quartz": 1 - vsh, "clay": vsh},
        saturations={"brine": sw, "oil": 1 - sw},
        pore_pressure=np.full(shape, 30e6), confining_pressure=np.full(shape, 55e6),
        config=RockPhysicsConfig(temperature=80.0),
    )
    assert r.vp.shape == shape and r.ai.shape == shape
    assert np.all(np.isfinite(r.vp)) and np.all(r.vp > r.vs)


def test_the_configuration_reports_every_model_it_used():
    text = RockPhysicsConfig().describe()
    for token in ("mineral mixing", "dry frame", "fluid mixing", "Batzle-Wang",
                  "Gassmann", "pressure model"):
        assert token in text
