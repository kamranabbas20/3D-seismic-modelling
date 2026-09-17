"""The layer-cake builder, pinchouts, and the surfaces they are made of.

The property that matters throughout: horizons may touch but must never
cross. `build_geology` rejects a crossing, so a pinchout has to arrive as a
thickness that reaches zero rather than as a horizon that dives through the
one above it.
"""

import numpy as np
import pytest

from sim3d.core.errors import ConfigError, ValidationError
from sim3d.core.grid import Grid3D
from sim3d.geology.builder import build_geology
from sim3d.geology.surfaces import Flat, Lens, Truncated, Wedge
from sim3d.geology.templates import TEMPLATES, layer_cake, template

EXTENT = (3000.0, 3000.0, 2200.0)


def grid(dz: float = 10.0) -> Grid3D:
    return Grid3D((0.0, 0.0, 0.0), (50.0, 50.0, dz),
                  (61, 61, int(EXTENT[2] / dz) + 1))


def thicknesses(model, name: str) -> np.ndarray:
    """Thickness map of one unit, from the built model's own horizons."""
    names = [layer.name for layer in model.layers]
    index = names.index(name)
    top = model.horizons[name]
    if index + 1 < len(names):
        return model.horizons[names[index + 1]] - top
    base = model.grid.origin[2] + model.grid.extent[2]
    return base - top


# --------------------------------------------------------------- surfaces
def test_wedge_runs_from_full_thickness_to_zero_and_stops():
    x, y = np.meshgrid(np.linspace(0, 2000, 21), np.linspace(0, 2000, 3),
                       indexing="ij")
    t = Wedge(60.0, azimuth=90.0, start=400.0, end=1400.0).depth(x, y)
    assert t[0, 0] == pytest.approx(60.0)
    assert t[4, 0] == pytest.approx(60.0)           # x = 400, still full
    assert t[14, 0] == pytest.approx(0.0)           # x = 1400, gone
    assert t[-1, 0] == pytest.approx(0.0)           # and stays gone
    assert np.all(np.diff(t[:, 0]) <= 1e-12)        # monotone, never negative
    assert np.all(t >= 0.0)


def test_lens_is_thickest_at_its_centre_and_zero_outside():
    x, y = np.meshgrid(np.linspace(0, 2000, 41), np.linspace(0, 2000, 41),
                       indexing="ij")
    t = Lens(80.0, centre=(1000.0, 1000.0), radius=(600.0, 400.0)).depth(x, y)
    assert t.max() == pytest.approx(80.0)
    assert t[20, 20] == pytest.approx(80.0)
    assert t[0, 0] == pytest.approx(0.0)
    assert np.all(t >= 0.0)


def test_truncation_clamps_rather_than_crosses():
    x, y = np.meshgrid(np.linspace(0, 1000, 5), np.linspace(0, 1000, 5),
                       indexing="ij")
    clamped = Truncated(Flat(900.0), Flat(1000.0)).depth(x, y)
    assert np.all(clamped == 1000.0)


def test_impossible_surfaces_are_refused():
    x = y = np.zeros((2, 2))
    with pytest.raises(ConfigError):
        Wedge(60.0, start=800.0, end=400.0).depth(x, y)
    with pytest.raises(ConfigError):
        Wedge(-1.0).depth(x, y)
    with pytest.raises(ConfigError):
        Lens(60.0, radius=(0.0, 100.0)).depth(x, y)
    with pytest.raises(ConfigError):
        Lens(60.0, taper=0.0).depth(x, y)


# ------------------------------------------------------------- layer cake
def test_layer_cake_builds_exactly_the_units_it_is_given():
    units = [{"name": f"unit_{i}", "facies": "shale", "thickness": 100.0}
             for i in range(7)]
    layers, _ = layer_cake(extent=EXTENT, units=units)
    assert [layer.name for layer in layers] == [u["name"] for u in units]
    model = build_geology(grid(), layers)
    for unit in units[:-1]:                 # the last one runs to the model base
        assert thicknesses(model, unit["name"]).mean() == pytest.approx(100.0)


def test_thickness_accumulates_so_horizons_can_never_cross():
    """The reason pinchouts work at all.

    Naming a depth per horizon lets a structure drive one through another;
    stacking non-negative thicknesses cannot, whatever the thicknesses do.
    """
    units = [
        {"name": "overburden", "facies": "shale", "thickness": 800.0},
        {"name": "wedge", "facies": "clean_sandstone", "thickness": 90.0,
         "pinch_out": {"shape": "wedge", "azimuth": 90.0,
                       "start": 0.25, "end": 0.75}},
        {"name": "reservoir", "facies": "clean_sandstone", "thickness": 120.0,
         "is_reservoir": True},
        {"name": "underburden", "facies": "sandy_shale", "thickness": 400.0},
    ]
    layers, _ = layer_cake(extent=EXTENT, units=units,
                           structure={"style": "dipping", "dip": 10.0})
    model = build_geology(grid(), layers)      # would raise on a crossing
    names = [layer.name for layer in model.layers]
    tops = [model.horizons[n] for n in names]
    for upper, lower in zip(tops, tops[1:]):
        assert np.all(lower >= upper - 1e-9)


def test_a_pinching_unit_really_does_vanish():
    units = [
        {"name": "overburden", "facies": "shale", "thickness": 800.0},
        {"name": "wedge", "facies": "clean_sandstone", "thickness": 90.0,
         "pinch_out": {"shape": "wedge", "azimuth": 90.0,
                       "start": 0.25, "end": 0.75}},
        {"name": "underburden", "facies": "shale", "thickness": 400.0},
    ]
    model = build_geology(grid(), layer_cake(extent=EXTENT, units=units)[0])
    thickness = thicknesses(model, "wedge")
    assert thickness.max() == pytest.approx(90.0)
    assert thickness.min() == pytest.approx(0.0)
    # And the unit below rises to meet it rather than leaving a gap.
    assert np.all(np.diff(model.horizons["underburden"][:, 30]) <= 1e-9)


def test_a_lens_pinches_out_all_round():
    units = [
        {"name": "overburden", "facies": "shale", "thickness": 900.0},
        {"name": "body", "facies": "clean_sandstone", "thickness": 100.0,
         "is_reservoir": True,
         "pinch_out": {"shape": "lens", "radius": [700.0, 500.0]}},
        {"name": "underburden", "facies": "shale", "thickness": 400.0},
    ]
    model = build_geology(grid(), layer_cake(extent=EXTENT, units=units)[0])
    thickness = thicknesses(model, "body")
    assert thickness.max() == pytest.approx(100.0)
    assert thickness[0, 0] == pytest.approx(0.0)       # absent in every corner
    assert thickness[-1, -1] == pytest.approx(0.0)
    reservoir = model.reservoir_mask
    assert reservoir.any() and not reservoir.all()


def test_structure_tilts_the_whole_package_together():
    units = [{"name": "a", "facies": "shale", "thickness": 900.0},
             {"name": "b", "facies": "clean_sandstone", "thickness": 100.0,
              "is_reservoir": True},
             {"name": "c", "facies": "shale", "thickness": 400.0}]
    layers, _ = layer_cake(extent=EXTENT, units=units,
                           structure={"style": "dipping", "dip": 8.0,
                                      "azimuth": 90.0})
    model = build_geology(grid(), layers)
    for name in ("a", "b", "c"):
        horizon = model.horizons[name][:, 30]
        assert horizon[-1] > horizon[0]          # deeper downdip
    # Tilting together means constant thickness, which is what stops a
    # structure from becoming a crossing horizon.
    assert thicknesses(model, "b").std() < 1e-6


def test_a_fold_shallows_over_its_crest():
    units = [{"name": "a", "facies": "shale", "thickness": 900.0},
             {"name": "b", "facies": "shale", "thickness": 400.0}]
    layers, _ = layer_cake(extent=EXTENT, units=units,
                           structure={"style": "anticline", "amplitude": 120.0})
    horizon = build_geology(grid(), layers).horizons["a"]
    assert horizon[30, 30] < horizon[0, 0]
    assert horizon[0, 0] - horizon[30, 30] == pytest.approx(120.0, rel=0.05)


def test_layer_cake_refuses_what_it_cannot_build():
    with pytest.raises(ConfigError):
        layer_cake(extent=EXTENT, units=[])
    with pytest.raises(ConfigError):
        layer_cake(extent=EXTENT, units=[
            {"name": "same", "facies": "shale", "thickness": 100.0},
            {"name": "same", "facies": "shale", "thickness": 100.0}])
    with pytest.raises(ConfigError):
        layer_cake(extent=EXTENT, units=[
            {"name": "a", "facies": "shale", "thickness": 0.0}])
    with pytest.raises(ConfigError):
        layer_cake(extent=EXTENT, units=[{"name": "a", "facies": "shale",
                                          "thickness": 100.0}],
                   structure={"style": "overturned"})
    with pytest.raises(ConfigError):
        layer_cake(extent=EXTENT, units=[
            {"name": "a", "facies": "shale", "thickness": 100.0,
             "pinch_out": {"shape": "wedge", "start": 0.9, "end": 0.2}}])


def test_reservoir_flag_follows_the_facies_unless_overridden():
    units = [{"name": "sand", "facies": "clean_sandstone", "thickness": 100.0},
             {"name": "inert", "facies": "clean_sandstone", "thickness": 100.0,
              "is_reservoir": False},
             {"name": "shale", "facies": "shale", "thickness": 100.0}]
    model = build_geology(grid(), layer_cake(extent=EXTENT, units=units)[0])
    flags = {layer.name: bool(model.reservoir_mask[model.layer_index == i].any())
             for i, layer in enumerate(model.layers)}
    assert flags == {"sand": True, "inert": False, "shale": False}


def test_layer_cake_is_registered_like_any_other_template():
    assert "layer_cake" in TEMPLATES
    layers, faults = template("layer_cake", extent=EXTENT)
    assert layers and not faults


# ----------------------------------------------------- the editor's helpers
def test_a_blank_property_cell_stays_unset_rather_than_becoming_zero():
    """An empty porosity means *take the facies default*, not zero.

    Writing 0.0 into the configuration would be a silent, catastrophic edit -
    a layer with no pore space - made by a user who typed nothing at all.
    """
    from sim3d.ui.streamlit_app import _rows_to_units, _units_to_rows

    rows = [{"name": "a", "facies": "shale", "thickness": 100.0,
             "reservoir": False, "pinch out": "none", "azimuth": 90.0,
             "from": 0.3, "to": 0.8,
             "porosity": None, "Vsh": float("nan"), "NTG": 0.85}]
    units = _rows_to_units(rows, EXTENT)
    assert "porosity" not in units[0]
    assert "vsh" not in units[0]
    assert units[0]["ntg"] == pytest.approx(0.85)

    # And an unnamed row is one being typed, not a unit.
    assert _rows_to_units([{**rows[0], "name": "  "}], EXTENT) == []

    # Round trip: rows -> units -> rows keeps what was set.
    assert _units_to_rows(units)[0]["NTG"] == pytest.approx(0.85)


def test_editor_rows_round_trip_a_pinchout():
    from sim3d.ui.streamlit_app import _rows_to_units, _units_to_rows

    units = _rows_to_units([{
        "name": "wedge", "facies": "clean_sandstone", "thickness": 80.0,
        "reservoir": True, "pinch out": "wedge", "azimuth": 45.0,
        "from": 0.2, "to": 0.7, "porosity": 0.28, "Vsh": None, "NTG": None}],
        EXTENT)
    assert units[0]["pinch_out"] == {"shape": "wedge", "azimuth": 45.0,
                                     "start": 0.2, "end": 0.7}
    layer_cake(extent=EXTENT, units=units)       # and it builds
    back = _units_to_rows(units)[0]
    assert back["pinch out"] == "wedge" and back["to"] == pytest.approx(0.7)


def test_a_lens_row_becomes_radii_in_metres():
    from sim3d.ui.streamlit_app import _rows_to_units

    units = _rows_to_units([{
        "name": "body", "facies": "clean_sandstone", "thickness": 60.0,
        "reservoir": True, "pinch out": "lens", "azimuth": 30.0,
        "from": 0.25, "to": 0.4, "porosity": None, "Vsh": None, "NTG": None}],
        EXTENT)
    assert units[0]["pinch_out"]["radius"] == pytest.approx(
        [0.25 * EXTENT[0], 0.4 * EXTENT[1]])
    layer_cake(extent=EXTENT, units=units)


def test_apply_is_offered_for_a_real_change_and_not_for_a_no_op():
    from sim3d.ui.streamlit_app import _describe_layers

    parameters = {"extent": list(EXTENT), "gross": 150.0}
    assert (_describe_layers("three_layer", parameters)
            == _describe_layers("three_layer", dict(parameters)))
    assert (_describe_layers("three_layer", {**parameters, "dip": 12.0})
            != _describe_layers("three_layer", parameters))
    # A configuration that cannot build is always worth applying away from.
    broken = _describe_layers("three_layer", {"not_a_parameter": 1})
    assert broken.startswith("unbuildable")
    assert broken != _describe_layers("three_layer", parameters)


def test_warnings_name_what_the_grid_and_the_wavelet_cannot_carry():
    from sim3d.ui.streamlit_app import _stratigraphy_warnings

    coarse = grid(dz=25.0)
    units = [{"name": "overburden", "facies": "shale", "thickness": 900.0},
             {"name": "sliver", "facies": "clean_sandstone", "thickness": 20.0,
              "is_reservoir": True},
             {"name": "wedge", "facies": "shale", "thickness": 200.0,
              "pinch_out": {"shape": "wedge", "start": 0.3, "end": 0.7}},
             {"name": "underburden", "facies": "shale", "thickness": 400.0}]
    layers, _ = layer_cake(extent=EXTENT, units=units)
    notes = " ".join(_stratigraphy_warnings(layers, coarse, 20.0))
    assert "sliver" in notes and "fewer than two cells" in notes
    assert "wedge" in notes and "pinches out" in notes


def test_a_stack_deeper_than_the_model_is_reported_not_silently_cut():
    from sim3d.ui.streamlit_app import _stratigraphy_warnings

    # Two units of 2,000 m in a 2,200 m model is NOT the failure: the last
    # unit always runs to the base, so it simply fills what is left.
    fine = layer_cake(extent=EXTENT, units=[
        {"name": "a", "facies": "shale", "thickness": 2000.0},
        {"name": "b", "facies": "shale", "thickness": 2000.0}])[0]
    assert "cut off" not in " ".join(_stratigraphy_warnings(fine, grid(), 20.0))

    # A unit whose top falls past the model base is.
    layers, _ = layer_cake(extent=EXTENT, units=[
        {"name": "a", "facies": "shale", "thickness": 1500.0},
        {"name": "b", "facies": "shale", "thickness": 1000.0},
        {"name": "c", "facies": "clean_sandstone", "thickness": 200.0,
         "is_reservoir": True}])
    notes = " ".join(_stratigraphy_warnings(layers, grid(), 20.0))
    assert "cut off" in notes
    assert "**c** has no thickness inside the model" in notes
