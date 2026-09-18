"""Faults built from a drawn map trace.

The trace fixes the origin, the strike and the along-strike extent at once,
so the tests here are mostly about those three agreeing with the line that
was drawn - and about the fault actually reaching the model, which is the
difference between a fault and a picture of one.
"""

import numpy as np
import pytest

from sim3d.core.config import ExperimentConfig
from sim3d.core.errors import ConfigError
from sim3d.core.grid import Grid3D
from sim3d.geology.builder import build_geology
from sim3d.geology.faults import Fault, build_faults, fault_from_trace
from sim3d.geology.templates import layer_cake
from sim3d.experiments.pipeline import Pipeline

EXTENT = (3000.0, 3000.0, 2200.0)


def grid() -> Grid3D:
    return Grid3D((0.0, 0.0, 0.0), (50.0, 50.0, 10.0), (61, 61, 221))


def flat_model(faults=None):
    units = [{"name": "overburden", "facies": "shale", "thickness": 1000.0},
             {"name": "reservoir", "facies": "clean_sandstone",
              "thickness": 120.0, "is_reservoir": True},
             {"name": "underburden", "facies": "shale", "thickness": 500.0}]
    layers, template_faults = layer_cake(extent=EXTENT, units=units)
    from sim3d.geology.faults import FaultSet
    return build_geology(grid(), layers, FaultSet(faults or []))


def top_depth(model, name: str) -> np.ndarray:
    """Depth of a unit's top as the *built model* has it, faulting included."""
    index = [layer.name for layer in model.layers].index(name)
    z = model.grid.axis(2)
    deep_enough = model.layer_index >= index
    first = np.argmax(deep_enough, axis=2)
    return np.where(deep_enough.any(axis=2), z[first], np.nan)


# ------------------------------------------------------------------ trace
def test_the_trace_sets_the_origin_strike_and_length():
    fault = fault_from_trace("F1", [[1000.0, 500.0], [1000.0, 2500.0]], 1400.0)
    assert fault.origin == pytest.approx((1000.0, 1500.0, 1400.0))
    assert fault.strike == pytest.approx(0.0)        # the trace runs +y
    assert fault.strike_extent == pytest.approx(1000.0)   # half its length
    assert fault.strike_vector == pytest.approx([0.0, 1.0, 0.0])


def test_strike_is_the_bearing_of_the_line_drawn():
    for (x1, y1), expected in ((( 500.0, 1500.0), 270.0),   # due -x
                               ((2500.0, 1500.0),  90.0),   # due +x
                               ((1500.0, 2500.0),   0.0),   # due +y
                               ((2500.0, 2500.0),  45.0)):
        fault = fault_from_trace("F", [[1500.0, 1500.0], [x1, y1]], 1400.0)
        assert fault.strike % 360.0 == pytest.approx(expected)


def test_drawing_the_line_backwards_flips_the_hanging_wall():
    """A fault dips to the right of the drawn direction, so the trace is
    directed and normalising the strike would throw that choice away."""
    forward = fault_from_trace("F", [[1500.0, 500.0], [1500.0, 2500.0]], 1400.0)
    backward = fault_from_trace("F", [[1500.0, 2500.0], [1500.0, 500.0]], 1400.0)
    probe = (2500.0, 1500.0, 1400.0)      # a point east of the trace
    assert forward.signed_distance(*probe) * backward.signed_distance(*probe) < 0
    # Only the horizontal part of the normal flips. Both faults still dip
    # downwards, so the vertical component is the same in each - negating
    # the whole normal would describe a fault dipping up out of the ground.
    assert forward.normal[:2] == pytest.approx(-backward.normal[:2])
    assert forward.normal[2] == pytest.approx(backward.normal[2])


def test_a_trace_that_is_not_a_line_is_refused():
    with pytest.raises(ConfigError):
        fault_from_trace("F", [[100.0, 100.0]], 1400.0)
    with pytest.raises(ConfigError):
        fault_from_trace("F", [[100.0, 100.0], [100.0, 100.0]], 1400.0)


def test_build_faults_takes_a_trace_or_a_plane_but_not_both():
    drawn, explicit = build_faults([
        {"name": "A", "trace": [[0.0, 0.0], [1000.0, 0.0]], "depth": 1200.0,
         "throw": 50.0},
        {"name": "B", "origin": [100.0, 100.0, 900.0], "strike": 30.0,
         "dip": 70.0}])
    assert drawn.strike == pytest.approx(90.0) and drawn.throw == 50.0
    assert isinstance(explicit, Fault) and explicit.strike == 30.0

    with pytest.raises(ConfigError):
        build_faults([{"name": "C", "trace": [[0.0, 0.0], [1.0, 0.0]],
                       "depth": 1.0, "strike": 12.0}])
    with pytest.raises(ConfigError):
        build_faults([{"name": "D", "trace": [[0.0, 0.0], [1.0, 0.0]]}])


# ------------------------------------------------------------- the model
def test_a_drawn_fault_offsets_the_stratigraphy_by_its_throw():
    """The difference between a fault and a picture of one."""
    plain = top_depth(flat_model(), "reservoir")
    assert np.nanmax(plain) == pytest.approx(np.nanmin(plain))    # flat

    fault = fault_from_trace("F1", [[1500.0, 0.0], [1500.0, 3000.0]], 1050.0,
                             dip=89.0, throw=80.0, zone_width=30.0)
    faulted = top_depth(flat_model([fault]), "reservoir")
    offset = np.nanmax(faulted) - np.nanmin(faulted)
    assert offset == pytest.approx(80.0, abs=10.0)   # within one cell
    # Down on the hanging wall, which is east of a trace drawn northwards.
    assert faulted[-1, 30] > faulted[0, 30]


def test_a_negative_throw_is_a_reverse_fault():
    fault = fault_from_trace("F1", [[1500.0, 0.0], [1500.0, 3000.0]], 1050.0,
                             dip=89.0, throw=-80.0, zone_width=30.0)
    faulted = top_depth(flat_model([fault]), "reservoir")
    assert faulted[-1, 30] < faulted[0, 30]      # hanging wall up
    assert "reverse" in fault.describe()


def test_a_sealing_fault_splits_the_model_into_compartments():
    fault = fault_from_trace("F1", [[1500.0, 0.0], [1500.0, 3000.0]], 1050.0,
                             dip=89.0, throw=40.0, transmissibility=0.0)
    model = flat_model([fault])
    x, y, z = np.meshgrid(*[model.grid.axis(i) for i in range(3)], indexing="ij")
    assert set(np.unique(model.faults.compartment_id(x, y, z))) == {0, 1}
    multiplier = model.faults.transmissibility_multiplier(x, y, z)
    assert multiplier.min() < 0.05          # sealing on the plane
    assert multiplier.max() == pytest.approx(1.0)   # untouched away from it


def test_a_configured_fault_is_added_to_the_templates_own():
    """A template that ships a fault is making a structural statement;
    drawing another must not silently drop it."""
    config = ExperimentConfig.load("examples/configs/demo_small.yaml")
    config.geology.template = "fault_compartment"
    config.geology.parameters = {"extent": list(EXTENT), "throw": 40.0}
    before = len(Pipeline(config).geology().faults)
    assert before == 1

    config.geology.faults = [{"name": "D1", "depth": 1250.0,
                              "trace": [[500.0, 500.0], [2500.0, 2500.0]]}]
    after = Pipeline(config).geology().faults
    assert len(after) == before + 1
    assert {f.name for f in after} == {"F1", "D1"}


def test_faults_round_trip_through_the_configuration(tmp_path):
    config = ExperimentConfig()
    assert config.geology.faults == []          # none unless asked for
    original = config.content_hash()
    config.geology.faults = [
        {"name": "D1", "trace": [[0.0, 0.0], [1000.0, 1000.0]],
         "depth": 1200.0, "throw": 55.0, "transmissibility": 0.0}]
    assert config.content_hash() != original, "a fault must invalidate a cache"

    reloaded = ExperimentConfig.load(config.save(tmp_path / "faulted.yaml"))
    assert reloaded.geology.faults[0]["throw"] == 55.0
    assert reloaded.content_hash() == config.content_hash()


def test_the_map_trace_of_a_fault_is_the_line_it_was_drawn_from():
    from sim3d.ui.streamlit_app import _fault_trace

    trace = [[600.0, 400.0], [2400.0, 2600.0]]
    fault = fault_from_trace("F1", trace, 1200.0)
    xs, ys = _fault_trace(fault, grid())
    ends = sorted(zip(xs, ys))
    assert ends[0] == pytest.approx(tuple(trace[0]))
    assert ends[1] == pytest.approx(tuple(trace[1]))


def test_an_unlimited_fault_is_drawn_right_across_the_model():
    from sim3d.ui.streamlit_app import _fault_trace

    unlimited = Fault(name="F", origin=(1500.0, 1500.0, 1200.0), strike=90.0,
                      strike_extent=None)
    xs, _ = _fault_trace(unlimited, grid())
    assert min(xs) < 0.0 and max(xs) > EXTENT[0]
