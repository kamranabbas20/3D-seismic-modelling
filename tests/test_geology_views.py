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


# ------------------------------------------------------ heterogeneity
def test_heterogeneity_controls_reach_the_built_model():
    """The four numbers that decide whether a flood fingers or fronts.

    A layer's porosity is a random field, and its correlation lengths and
    variance are what set the sweep. The template fixed them and nothing -
    configuration or interface - could reach them.
    """
    from sim3d.geology.builder import build_geology, override_layers

    units = [{"name": "over", "facies": "shale", "thickness": 900.0},
             {"name": "res", "facies": "clean_sandstone", "thickness": 200.0,
              "is_reservoir": True, "sublayer_porosity_range": 0.0},
             {"name": "under", "facies": "shale", "thickness": 400.0}]

    def spread(spec):
        layers, _ = layer_cake(extent=EXTENT, units=units)
        if spec is not _MISSING:
            override_layers(layers, [{"name": "res", "heterogeneity": spec}])
        model = build_geology(grid(dz=20.0), layers)
        return float(model.porosity[model.reservoir_mask].std())

    default = spread(_MISSING)
    assert default == pytest.approx(0.030, abs=0.006)
    assert spread({"std": 0.07}) == pytest.approx(0.070, abs=0.012)
    assert spread(False) == pytest.approx(0.0, abs=1e-9), (
        "off is uniform, which is not the same as a field of zero variance")


_MISSING = object()


def test_a_unit_can_carry_its_own_heterogeneity():
    """Spelled out on the unit, it beats the default the reservoir flag picks."""
    from sim3d.geology.builder import build_geology

    def spread(**extra):
        layers, _ = layer_cake(extent=EXTENT, units=[
            {"name": "over", "facies": "shale", "thickness": 900.0},
            {"name": "res", "facies": "clean_sandstone", "thickness": 200.0,
             "is_reservoir": True, "sublayer_porosity_range": 0.0, **extra},
            {"name": "under", "facies": "shale", "thickness": 400.0}])
        model = build_geology(grid(dz=20.0), layers)
        return float(model.porosity[model.reservoir_mask].std())

    assert spread(heterogeneity={"std": 0.07}) > spread() + 0.02
    assert spread(heterogeneity=False) == pytest.approx(0.0, abs=1e-9)


def test_the_correlation_geometry_actually_changes_the_fabric():
    """Long-against-short is a channelised rock; equal is patchy. If the
    ranges did not reach the field, both would look the same."""
    from sim3d.geology.builder import build_geology

    def fabric(major, minor):
        layers, _ = layer_cake(extent=EXTENT, units=[
            {"name": "over", "facies": "shale", "thickness": 900.0},
            {"name": "res", "facies": "clean_sandstone", "thickness": 200.0,
             "is_reservoir": True, "sublayer_porosity_range": 0.0,
             "heterogeneity": {"std": 0.05, "azimuth": 90.0, "seed": 3,
                               "correlation_major": major,
                               "correlation_minor": minor}},
            {"name": "under", "facies": "shale", "thickness": 400.0}])
        model = build_geology(grid(dz=20.0), layers)
        phi = np.where(model.reservoir_mask, model.porosity, np.nan)
        plane = np.nanmean(phi, axis=2)
        # Roughness along x against along y: a fabric elongated along x is
        # smoother along x than across it.
        return (float(np.nanmean(np.abs(np.diff(plane, axis=0))))
                / float(np.nanmean(np.abs(np.diff(plane, axis=1)))))

    elongated = fabric(1200.0, 120.0)
    isotropic = fabric(400.0, 400.0)
    assert elongated < 0.75 * isotropic, (
        f"elongated {elongated:.3f} vs isotropic {isotropic:.3f}")


def test_heterogeneity_reads_back_as_the_controls_that_made_it():
    """So an editor can seed from what a layer already carries."""
    from sim3d.geology.builder import heterogeneity_of, heterogeneity_specs

    asked = {"std": 0.044, "vsh_std": 0.066, "correlation_major": 210.0,
             "correlation_minor": 130.0, "correlation_vertical": 9.0,
             "azimuth": 75.0, "model": "spherical", "seed": 12}
    layers, _ = layer_cake(extent=EXTENT, units=[
        {"name": "a", "facies": "shale", "thickness": 900.0,
         "heterogeneity": asked},
        {"name": "b", "facies": "shale", "thickness": 400.0}])
    assert heterogeneity_of(layers[0]) == asked
    assert heterogeneity_of(layers[1]) is None, "no field is not an empty field"

    porosity, shale = heterogeneity_specs(asked)
    assert porosity.seed + 1 == shale.seed, (
        "correlated realisations of one fabric, not two unrelated noises")
    assert porosity.correlation_major == shale.correlation_major


def test_an_unknown_heterogeneity_control_is_refused():
    from sim3d.geology.builder import heterogeneity_specs

    with pytest.raises(ConfigError):
        heterogeneity_specs({"std": 0.03, "correlation_lenght": 400.0})
