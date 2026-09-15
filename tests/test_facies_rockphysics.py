"""Per-facies rock physics.

One global dry-frame model is not a simplification, it is a bias, and it
falls on exactly the contrast the angle stacks are built on: with a single
soft-sand frame the shale of the flat template comes out *slower* than the
reservoir sand, so the top of the reservoir has almost no impedance
contrast at all. That measurement is the first test here.
"""

import numpy as np
import pytest

from sim3d.core.errors import ConfigError
from sim3d.core.grid import Grid3D
from sim3d.fourd.scenarios import build_earth_models, build_states
from sim3d.geology.builder import build_geology
from sim3d.geology.facies import FACIES
from sim3d.geology.templates import template
from sim3d.reservoir.mechanistic import ReservoirScenario
from sim3d.reservoir.state import initial_state
from sim3d.rockphysics.model import (
    FACIES_OVERRIDABLE, RockPhysicsConfig, with_facies_overrides,
)
from sim3d.wells.well import Well, WellSet

STIFF_SHALE = {"shale": {"dry_frame_model": "stiff_sand",
                         "critical_porosity": 0.55, "coordination": 6.0}}


@pytest.fixture
def model():
    grid = Grid3D(origin=(0.0, 0.0, 0.0), spacing=(100.0, 100.0, 10.0),
                  shape=(7, 7, 181))
    layers, faults = template("flat", z_reservoir=1150.0, gross=150.0)
    geology = build_geology(grid, layers, faults)
    wells = WellSet([Well("I1", "injector", 200.0, 200.0),
                     Well("P1", "producer", 400.0, 400.0)])
    states = build_states(initial_state(geology, sw=0.3),
                          ReservoirScenario(name="m"), wells)
    return geology, states


def elastic(model, facies_configs):
    geology, states = model
    earth = build_earth_models(states, geology, RockPhysicsConfig(),
                               facies_configs=facies_configs)
    return earth.rock_physics["baseline"]


# ------------------------------------------------------------ the override
def test_one_global_frame_flattens_the_shale_over_sand_contrast(model):
    """The bias this exists to remove, stated as a measurement."""
    geology, _ = model
    shale = geology.facies_code == FACIES["shale"].code
    sand = geology.facies_code == FACIES["clean_sandstone"].code

    one = elastic(model, None)
    many = elastic(model, STIFF_SHALE)

    def contrast(rp):
        return float((rp.vp[sand] * rp.rho[sand]).mean()
                     - (rp.vp[shale] * rp.rho[shale]).mean())

    # With one soft frame the shale comes out *slower* than the reservoir
    # sand, so the impedance step at the reservoir top has the wrong sign
    # and almost no size - a top-reservoir reflection that barely exists.
    assert one.vp[shale].mean() < one.vp[sand].mean()
    assert contrast(one) > 0.0
    # A stiff shale frame restores both: harder shale over softer sand.
    assert many.vp[shale].mean() > many.vp[sand].mean()
    assert contrast(many) < 0.0
    assert abs(contrast(many)) > 20.0 * abs(contrast(one))


def test_only_the_named_facies_moves(model):
    geology, _ = model
    shale = geology.facies_code == FACIES["shale"].code
    sand = geology.facies_code == FACIES["clean_sandstone"].code
    one, many = elastic(model, None), elastic(model, STIFF_SHALE)
    assert np.allclose(one.vp[sand], many.vp[sand])
    assert not np.allclose(one.vp[shale], many.vp[shale])


def test_no_overrides_is_identical_to_the_global_chain(model):
    """The composite path must not perturb a model that asked for nothing."""
    plain = elastic(model, None)
    for empty in (None, {}):
        assert np.array_equal(elastic(model, empty).vp, plain.vp)


def test_an_override_that_repeats_the_global_values_changes_nothing(model):
    plain = elastic(model, None)
    same = elastic(model, {"shale": {"dry_frame_model": "soft_sand"}})
    assert np.allclose(plain.vp, same.vp)


def test_two_facies_can_differ_at_once(model):
    geology, _ = model
    both = elastic(model, {
        "shale": {"dry_frame_model": "stiff_sand", "critical_porosity": 0.55},
        "clean_sandstone": {"critical_porosity": 0.36},
    })
    one = elastic(model, None)
    for name in ("shale", "clean_sandstone"):
        mask = geology.facies_code == FACIES[name].code
        assert not np.allclose(one.vp[mask], both.vp[mask])


def test_every_cell_still_has_a_configuration(model):
    """Facies with no override fall to the global one; a hole would show as
    a zero velocity, which no downstream stage would survive."""
    result = elastic(model, STIFF_SHALE)
    assert np.all(np.isfinite(result.vp))
    assert result.vp.min() > 0.0


# ------------------------------------------------------------- validation
def test_fluid_properties_cannot_be_varied_by_facies():
    """One connected reservoir has one fluid; varying it by lithology would
    describe a model this chain cannot build."""
    with pytest.raises(ConfigError, match="Fluid properties are global"):
        with_facies_overrides(RockPhysicsConfig(), {"temperature": 90.0})
    with pytest.raises(ConfigError, match="Fluid properties are global"):
        with_facies_overrides(RockPhysicsConfig(), {"gor": 200.0})


def test_an_unknown_override_lists_what_can_be_set():
    with pytest.raises(ConfigError, match="overridable keys"):
        with_facies_overrides(RockPhysicsConfig(), {"critical_porosty": 0.5})


def test_an_unknown_facies_lists_the_catalogue(model):
    geology, states = model
    with pytest.raises(ConfigError, match="not in the catalogue"):
        build_earth_models(states, geology, RockPhysicsConfig(),
                           facies_configs={"granite": {"coordination": 8.0}})


def test_the_override_does_not_mutate_the_base():
    base = RockPhysicsConfig()
    with_facies_overrides(base, {"dry_frame_model": "stiff_sand",
                                 "critical_porosity": 0.55, "biot": 0.7})
    assert base.dry_frame_model == "soft_sand"
    assert base.critical_porosity == pytest.approx(0.40)
    assert base.pressure.biot == pytest.approx(1.0)


def test_the_pressure_model_is_overridable_because_the_frame_owns_it():
    config = with_facies_overrides(RockPhysicsConfig(), {"biot": 0.8})
    assert config.pressure.biot == pytest.approx(0.8)
    assert "biot" in FACIES_OVERRIDABLE


def test_it_reaches_the_pipeline():
    from sim3d.core.config import ExperimentConfig
    from sim3d.experiments.pipeline import Pipeline
    config = ExperimentConfig.load("examples/configs/demo_small.yaml")
    config.rock_physics.facies = dict(STIFF_SHALE)
    pipeline = Pipeline(config)
    assert pipeline.config.rock_physics.facies["shale"]["coordination"] == 6.0
