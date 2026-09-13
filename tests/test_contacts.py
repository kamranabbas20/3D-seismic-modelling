"""Fluid contacts: the one reflector whose geometry owes nothing to layering.

A contact is a horizontal plane cutting across dipping stratigraphy, which
is what makes a flat spot a flat spot. The things worth pinning are that
the column is the right way up - depth increases downwards and getting the
sign wrong puts water above oil - and that asking for no contact still
gives the older uniform behaviour exactly.
"""

import numpy as np
import pytest

from sim3d.core.errors import ConfigError
from sim3d.core.grid import Grid3D
from sim3d.geology.builder import build_geology
from sim3d.geology.templates import template
from sim3d.reservoir.state import initial_state, saturation_from_contacts

SWIRR = 0.20


@pytest.fixture
def geology():
    grid = Grid3D(origin=(0.0, 0.0, 0.0), spacing=(100.0, 100.0, 10.0),
                  shape=(9, 9, 181))
    layers, faults = template("flat", z_reservoir=1150.0, gross=150.0)
    return build_geology(grid, layers, faults)


def column(**kwargs):
    z = np.linspace(1000.0, 1400.0, 41)
    return z, saturation_from_contacts(z, SWIRR, **kwargs)


# ------------------------------------------------------------- the column
def test_water_sits_below_the_contact_and_oil_above():
    """Depth increases downwards; a sign slip here floats the water."""
    z, (sw, so, sg) = column(owc=1300.0, goc=None)
    assert np.allclose(sw[z >= 1300.0], 1.0)
    assert np.allclose(sw[z < 1300.0], SWIRR)
    assert np.allclose(so[z < 1300.0], 1.0 - SWIRR)
    assert np.allclose(so[z >= 1300.0], 0.0)


def test_a_gas_cap_sits_above_the_oil():
    z, (sw, so, sg) = column(owc=1300.0, goc=1200.0)
    assert np.allclose(sg[z < 1200.0], 1.0 - SWIRR)
    assert np.allclose(sg[(z >= 1200.0)], 0.0)
    assert np.allclose(so[(z >= 1200.0) & (z < 1300.0)], 1.0 - SWIRR)


def test_a_gas_cap_displaces_oil_not_the_irreducible_water():
    """Irreducible water is held by capillarity and does not move for gas."""
    _, (sw, so, sg) = column(owc=1300.0, goc=1200.0)
    assert np.allclose(sw.min(), SWIRR)
    assert np.all(sw >= SWIRR - 1e-12)


def test_the_transition_zone_ramps_between_the_two():
    z, (sw, _, _) = column(owc=1300.0, goc=None, transition=40.0)
    inside = (z > 1260.0) & (z < 1300.0)
    assert inside.any()
    assert np.all(sw[inside] > SWIRR) and np.all(sw[inside] < 1.0)
    # Monotonically wetter downwards.
    assert np.all(np.diff(sw[inside]) >= -1e-12)


def test_a_sharp_contact_is_the_zero_transition_limit():
    z, (sharp, _, _) = column(owc=1300.0, goc=None, transition=0.0)
    _, (thin, _, _) = column(owc=1300.0, goc=None, transition=1e-9)
    assert np.allclose(sharp, thin)


def test_saturations_close_everywhere():
    for kwargs in ({"owc": 1300.0, "goc": None},
                   {"owc": 1300.0, "goc": 1200.0},
                   {"owc": 1300.0, "goc": 1200.0, "transition": 60.0}):
        _, (sw, so, sg) = column(**kwargs)
        assert np.allclose(sw + so + sg, 1.0)


def test_a_gas_cap_below_the_oil_water_contact_is_refused():
    with pytest.raises(ConfigError, match="above the oil-water contact"):
        column(owc=1200.0, goc=1300.0)


def test_a_negative_transition_and_a_silly_saturation_are_refused():
    with pytest.raises(ConfigError, match="must not be negative"):
        column(owc=1300.0, goc=None, transition=-5.0)
    with pytest.raises(ConfigError, match=r"\[0, 1\]"):
        saturation_from_contacts(np.array([1000.0]), 1.4, 1300.0, None)


# --------------------------------------------------------------- the state
def test_no_contact_keeps_the_uniform_behaviour_exactly(geology):
    """The older path has to stay bit-identical: a mechanistic sweep test
    wants a uniform column, and silently giving it a water leg would change
    every result that predates contacts."""
    state = initial_state(geology, sw=0.3, sg=0.0)
    mask = geology.reservoir_mask
    assert np.allclose(state.sw[mask], 0.3)
    assert np.allclose(state.sw[~mask], 1.0)


def test_a_contact_gives_the_reservoir_a_water_leg(geology):
    state = initial_state(geology, sw=SWIRR, owc=1250.0)
    mask = geology.reservoir_mask
    z = geology.grid.axis(2)[None, None, :] * np.ones(geology.grid.shape)
    assert np.allclose(state.sw[mask & (z >= 1250.0)], 1.0)
    assert np.allclose(state.sw[mask & (z < 1250.0)], SWIRR)


def test_cells_outside_the_reservoir_stay_wet_whatever_the_contacts(geology):
    state = initial_state(geology, sw=SWIRR, owc=1250.0, goc=1180.0)
    outside = ~geology.reservoir_mask
    assert np.allclose(state.sw[outside], 1.0)
    assert np.allclose(state.sg[outside], 0.0)


def test_the_contacts_are_recorded_in_the_provenance(geology):
    state = initial_state(geology, sw=SWIRR, owc=1250.0, goc=1180.0,
                          transition=20.0)
    text = " ".join(state.provenance)
    assert "OWC 1250" in text and "GOC 1180" in text and "20 m transition" in text


def test_a_contact_reaches_the_pipeline(geology):
    from sim3d.core.config import ExperimentConfig
    from sim3d.experiments.pipeline import Pipeline
    config = ExperimentConfig.load("examples/configs/demo_small.yaml")
    config.reservoir.baseline.sw = SWIRR
    config.reservoir.baseline.owc = 1200.0
    pipeline = Pipeline(config)
    state = pipeline.baseline_state()
    assert state.sw.max() == pytest.approx(1.0)
    assert state.sw[state.reservoir_mask].min() == pytest.approx(SWIRR)
