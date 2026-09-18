"""What the geology editor shows, and the geometry behind it.

The editor drew one section through the model centre while the app's own
three-panel viewer and its 3D scene sat on other pages. These cover the
pieces that closed that gap: the depth range a dip actually produces, the
inversion that puts faulted horizons on a section, the click targeting that
removed the layer dropdown, and the three new views.
"""

import numpy as np
import pytest

from sim3d.core.errors import ConfigError
from sim3d.core.grid import Grid3D
from sim3d.geology.builder import build_geology
from sim3d.geology.faults import FaultSet, fault_from_trace, present_day_depth
from sim3d.geology.picking import nearest_horizon
from sim3d.geology.templates import layer_cake, structure_summary
from sim3d.ui import components as ui

EXTENT = (3000.0, 3000.0, 2400.0)


def grid(dz: float = 5.0) -> Grid3D:
    return Grid3D((0.0, 0.0, 0.0), (25.0, 25.0, dz),
                  (121, 121, int(EXTENT[2] / dz) + 1))


UNITS = [{"name": "overburden", "facies": "shale", "thickness": 1000.0},
         {"name": "seal", "facies": "shale", "thickness": 60.0},
         {"name": "reservoir", "facies": "clean_sandstone", "thickness": 110.0,
          "is_reservoir": True},
         {"name": "underburden", "facies": "sandy_shale", "thickness": 500.0}]


def stack(structure=None):
    return layer_cake(extent=EXTENT, units=UNITS, structure=structure)[0]


# ------------------------------------------------- the dip's consequence
def test_a_dip_is_reported_as_the_depth_range_it_makes():
    """The angle is chosen; the relief is meant."""
    summary = structure_summary({"style": "dipping", "dip": 6.0,
                                 "azimuth": 90.0}, EXTENT, datum=900.0)
    # tan(6°) x 3,000 m = 315 m, centred on the datum.
    assert summary["relief"] == pytest.approx(3000.0 * np.tan(np.radians(6.0)),
                                              rel=1e-6)
    assert summary["deepest"] - 900.0 == pytest.approx(
        900.0 - summary["shallowest"], rel=1e-6)
    assert "315 m" in summary["text"] and "6°" in summary["text"]


def test_a_flat_structure_makes_no_relief_and_says_so():
    summary = structure_summary(None, EXTENT, datum=900.0)
    assert summary["relief"] == 0.0
    assert summary["shallowest"] == summary["deepest"] == 900.0
    assert "Flat" in summary["text"]


def test_a_fold_reports_its_own_amplitude():
    summary = structure_summary({"style": "anticline", "amplitude": 120.0},
                                EXTENT, datum=900.0)
    assert 0.0 < summary["relief"] <= 120.0 + 1e-6
    assert summary["shallowest"] < 900.0      # the crest comes up


# ---------------------------------------- horizons after faulting, exactly
def test_the_faulted_depth_of_a_horizon_matches_the_built_model():
    """Horizons are defined before faulting, so a section that draws them
    straight is missing every offset - by the whole throw."""
    fault = fault_from_trace("F1", [[1500.0, 0.0], [1500.0, 3000.0]], 1050.0,
                             dip=70.0, throw=80.0, zone_width=40.0)
    g = grid()
    model = build_geology(g, stack(), FaultSet([fault]))
    index = [layer.name for layer in model.layers].index("reservoir")
    deep = model.layer_index >= index
    built = np.where(deep.any(axis=2), g.axis(2)[np.argmax(deep, axis=2)], np.nan)

    x, y = np.meshgrid(g.axis(0), g.axis(1), indexing="ij")
    solved = present_day_depth([fault], x, y, model.horizons["reservoir"])
    assert np.nanmax(np.abs(solved - built)) <= g.dz + 1e-9
    # And the unfaulted surface really is out by the throw, so this has teeth.
    assert np.nanmax(np.abs(model.horizons["reservoir"] - built)) == \
        pytest.approx(80.0, abs=g.dz)


def test_with_no_faults_the_depth_is_returned_untouched():
    restored = np.array([900.0, 1000.0])
    assert present_day_depth([], np.zeros(2), np.zeros(2), restored) is restored


# ------------------------------------------------- clicking, not navigating
def test_a_click_lands_on_the_nearest_horizon():
    layers = stack()
    g = grid()
    names = [layer.name for layer in layers]
    # Bases sit at 1,000, 1,060 and 1,170 m.
    for depth, expected in ((980.0, "overburden"), (1010.0, "overburden"),
                            (1055.0, "seal"), (1100.0, "seal"),
                            (1165.0, "reservoir"), (1900.0, "reservoir")):
        index = nearest_horizon(layers, g, 0, 1500.0, 1500.0, depth)
        assert names[index] == expected, f"{depth} m went to {names[index]}"


def test_the_bottom_layer_is_never_the_target():
    """It runs to the base of the model and has no base to move."""
    layers = stack()
    g = grid()
    for depth in np.linspace(900.0, 2400.0, 40):
        assert nearest_horizon(layers, g, 0, 1500.0, 1500.0, float(depth)) \
            < len(layers) - 1


def test_a_single_layer_has_nothing_to_aim_at():
    layers = layer_cake(extent=EXTENT, units=[UNITS[0]])[0]
    with pytest.raises(ConfigError):
        nearest_horizon(layers, grid(), 0, 1500.0, 1500.0, 1000.0)


# ------------------------------------------------------------- the views
def test_the_isopach_marks_the_pinchout_and_the_section_lines():
    layers = layer_cake(extent=EXTENT, units=[
        UNITS[0],
        {**UNITS[2], "pinch_out": {"shape": "wedge", "start": 0.4, "end": 0.8}},
        UNITS[3]])[0]
    g = grid()
    thickness = layers[2].top.on_grid(g) - layers[1].top.on_grid(g)
    figure = ui.isopach_figure(thickness, g, sections=[500.0, 2500.0], axis=0,
                               title="t")
    kinds = [trace.type for trace in figure.data]
    assert "heatmap" in kinds
    assert "contour" in kinds, "a pinchout needs its zero line drawn"
    assert len(figure.layout.shapes) == 2, "one line per section"
    assert figure.layout.yaxis.scaleanchor == "x", "a map is not stretched"


def test_the_stack_column_is_drawn_at_true_proportions():
    layers, _ = layer_cake(extent=EXTENT, units=UNITS)
    figure = ui.layer_stack_figure(layers, grid(), highlight="reservoir")
    heights = {trace.name: float(trace.y[0]) for trace in figure.data}
    assert heights["overburden"] == pytest.approx(1000.0)
    assert heights["reservoir"] == pytest.approx(110.0)
    assert heights["overburden"] > 9 * heights["reservoir"], (
        "a table gives these the same row height; this is the point of the view")
    assert figure.layout.yaxis.autorange == "reversed"


def test_the_fault_section_draws_the_plane_at_its_true_dip():
    fault = fault_from_trace("F1", [[1500.0, 0.0], [1500.0, 3000.0]], 1050.0,
                             dip=60.0, throw=70.0, dip_extent=400.0)
    figure = ui.fault_section_figure(fault, stack(), grid())
    plane = next(t for t in figure.data if t.name.endswith("plane"))
    run = float(plane.x[1] - plane.x[0])
    rise = float(plane.y[1] - plane.y[0])
    assert rise / run == pytest.approx(np.tan(np.radians(60.0)), rel=1e-6)
    # It stops where dip_extent says it stops.
    assert max(abs(float(v)) for v in plane.x) == pytest.approx(
        400.0 * np.cos(np.radians(60.0)), rel=1e-6)
    assert "70 m throw" in figure.layout.title.text
    assert "normal" in figure.layout.title.text


def test_the_section_shows_horizons_after_faulting_when_asked():
    fault = fault_from_trace("F1", [[1500.0, 0.0], [1500.0, 3000.0]], 1050.0,
                             dip=85.0, throw=90.0, zone_width=30.0)
    layers, g = stack(), grid()
    plain = ui.stratigraphy_figure(layers, g, axis=0)
    faulted = ui.stratigraphy_figure(layers, g, axis=0, faults=FaultSet([fault]))
    band = lambda fig: np.asarray(   # noqa: E731
        next(t for t in fig.data if t.name.startswith("reservoir")).y, dtype=float)
    assert np.nanmax(band(faulted)) - np.nanmax(band(plain)) == \
        pytest.approx(90.0, abs=2.0)
