"""Hand-drawn geobodies painted over a template.

Two things carry the weight. The body must land where it was drawn - a
channel is a flow barrier or a flow path and a margin in the wrong place
is a different experiment - and it must reach the *property cube*, not
the display, or "draw a channel and simulate it" is a drawing.
"""

import numpy as np
import pytest

from sim3d.core.config import ExperimentConfig
from sim3d.core.errors import ConfigError
from sim3d.core.grid import Grid3D
from sim3d.core.units import MILLIDARCY
from sim3d.experiments.pipeline import Pipeline
from sim3d.geology.bodies import (
    GeoBody, apply_bodies, body_mask, build_bodies,
)


@pytest.fixture
def grid():
    return Grid3D(origin=(0.0, 0.0, 0.0), spacing=(25.0, 25.0, 10.0),
                  shape=(81, 81, 181))


def channel(**kwargs):
    spec = dict(name="ch1", path=[[200.0, 200.0], [1800.0, 1200.0]],
                width=300.0, top=1200.0, thickness=40.0)
    spec.update(kwargs)
    return GeoBody(**spec)


# ---------------------------------------------------------------- geometry
def test_the_channel_lands_within_half_a_width_of_its_centreline(grid):
    """Distance to the *segment*, not to the vertices: a centreline drawn
    with four clicks would otherwise pinch to nothing between them."""
    body = channel()
    mask = body_mask(body, grid)
    xs, ys = grid.axis(0), grid.axis(1)
    ix, iy, _ = np.nonzero(mask)
    a, b = np.array(body.path[0]), np.array(body.path[1])
    d = b - a
    for x, y in zip(xs[ix], ys[iy]):
        t = np.clip(((x - a[0]) * d[0] + (y - a[1]) * d[1]) / (d @ d), 0.0, 1.0)
        assert np.hypot(x - (a[0] + t * d[0]),
                        y - (a[1] + t * d[1])) <= 0.5 * body.width + 1e-9


def test_the_midpoint_of_a_two_click_channel_is_inside_it(grid):
    """The failure a vertex-distance implementation would show: a body that
    exists only as two blobs at the ends."""
    mask = body_mask(channel(), grid)
    mid = grid.nearest_index((1000.0, 700.0, 1210.0))
    assert mask[mid]


def test_the_body_occupies_exactly_its_depth_interval(grid):
    mask = body_mask(channel(top=1200.0, thickness=40.0), grid)
    occupied = grid.axis(2)[mask.any(axis=(0, 1))]
    assert occupied.min() >= 1200.0
    assert occupied.max() < 1240.0


def test_a_lens_fills_its_outline(grid):
    lens = GeoBody(name="l1", type="lens", top=1200.0, thickness=20.0,
                   path=[[400.0, 400.0], [1200.0, 400.0],
                         [1200.0, 1200.0], [400.0, 1200.0]])
    mask = body_mask(lens, grid)
    assert mask[grid.nearest_index((800.0, 800.0, 1205.0))]
    assert not mask[grid.nearest_index((1600.0, 800.0, 1205.0))]


def test_a_wider_channel_takes_more_cells(grid):
    narrow = body_mask(channel(width=100.0), grid).sum()
    wide = body_mask(channel(width=400.0), grid).sum()
    assert wide > narrow


# -------------------------------------------------------------- validation
def test_a_channel_needs_two_points_and_a_lens_needs_three():
    with pytest.raises(ConfigError, match="at least 2 path points"):
        GeoBody(type="channel", path=[[0.0, 0.0]])
    with pytest.raises(ConfigError, match="at least 3 path points"):
        GeoBody(type="lens", path=[[0.0, 0.0], [1.0, 1.0]])


def test_an_unknown_body_type_lists_the_ones_that_exist():
    with pytest.raises(ConfigError, match="channel"):
        GeoBody(type="turbidite", path=[[0.0, 0.0], [1.0, 1.0]])


def test_a_zero_width_or_thickness_is_refused():
    with pytest.raises(ConfigError, match="positive width"):
        GeoBody(path=[[0.0, 0.0], [1.0, 1.0]], width=0.0)
    with pytest.raises(ConfigError, match="positive thickness"):
        GeoBody(path=[[0.0, 0.0], [1.0, 1.0]], thickness=0.0)


def test_a_body_that_misses_the_model_says_so_rather_than_doing_nothing(grid):
    """Silently painting nothing would leave the user looking for a channel
    that was never there, with no way to tell drawing from a typo."""
    from sim3d.geology.builder import build_geology
    from sim3d.geology.templates import template
    layers, faults = template("flat", z_reservoir=1200.0, gross=100.0)
    model = build_geology(grid, layers, faults)
    with pytest.raises(ConfigError, match="does not intersect"):
        apply_bodies(model, [channel(top=9000.0)])


def test_an_unknown_key_is_an_error_not_a_default():
    with pytest.raises(ConfigError, match="unknown key"):
        build_bodies([{"name": "c", "path": [[0, 0], [1, 1]], "widht": 300.0}])


def test_duplicate_names_are_refused():
    entry = {"name": "c", "path": [[0, 0], [1, 1]]}
    with pytest.raises(ConfigError, match="duplicate geobody name"):
        build_bodies([entry, dict(entry)])


# ---------------------------------------------------------------- painting
def _flat_model(grid):
    from sim3d.geology.builder import build_geology
    from sim3d.geology.templates import template
    layers, faults = template("flat", z_reservoir=1200.0, gross=100.0)
    return build_geology(grid, layers, faults)


def test_painting_changes_the_properties_the_simulator_reads(grid):
    """Porosity and permeability, not a colour: the water has to go where
    the channel goes or the body is decoration."""
    model = _flat_model(grid)
    body = channel(porosity=0.31, permeability_md=2500.0, vsh=0.04, ntg=0.98)
    mask = body_mask(body, grid)
    before = model.permeability_md[mask].copy()
    apply_bodies(model, [body])
    assert np.allclose(model.porosity[mask], 0.31)
    assert np.allclose(model.permeability[mask], 2500.0 * MILLIDARCY)
    assert np.allclose(model.vsh[mask], 0.04)
    assert np.allclose(model.ntg[mask], 0.98)
    # Compared in mD, not m^2: permeability in SI is ~1e-12, where
    # allclose's default atol of 1e-8 calls every value equal to every other.
    assert not np.allclose(model.permeability_md[mask], before)


def test_painting_leaves_everything_outside_the_body_alone(grid):
    model = _flat_model(grid)
    body = channel(porosity=0.31)
    mask = body_mask(body, grid)
    before = model.porosity.copy()
    apply_bodies(model, [body])
    assert np.array_equal(model.porosity[~mask], before[~mask])


def test_an_unset_property_takes_the_facies_midpoint(grid):
    model = _flat_model(grid)
    body = channel(facies="clean_sandstone")
    apply_bodies(model, [body])
    assert np.allclose(model.porosity[body_mask(body, grid)], 0.26)


def test_a_later_body_overwrites_an_earlier_one_where_they_cross(grid):
    """The rule a drawing program uses, and the only one that makes
    'draw another channel across that one' mean what it looks like."""
    model = _flat_model(grid)
    first = channel(name="a", porosity=0.20)
    second = channel(name="b", path=[[200.0, 1200.0], [1800.0, 200.0]],
                     porosity=0.30)
    apply_bodies(model, [first, second])
    overlap = body_mask(first, grid) & body_mask(second, grid)
    assert overlap.any()
    assert np.allclose(model.porosity[overlap], 0.30)


def test_a_shale_body_can_cut_the_reservoir_in_two(grid):
    """The other half of the point: a drawn body may remove reservoir as
    well as add it, which is how a sealing feature gets tested."""
    model = _flat_model(grid)
    body = channel(facies="shale", top=1200.0, thickness=60.0)
    mask = body_mask(body, grid)
    assert model.reservoir_mask[mask].any()
    apply_bodies(model, [body])
    assert not model.reservoir_mask[mask].any()


# ----------------------------------------------------------------- wiring
def test_a_configured_body_reaches_the_built_geology():
    """End to end through the pipeline, because the value of the feature is
    entirely in whether the rest of the chain sees it."""
    config = ExperimentConfig.load("examples/configs/demo_small.yaml")
    config.geology.bodies = [{
        "name": "ch1", "path": [[600.0, 700.0], [1400.0, 1300.0]],
        "width": 250.0, "top": 1150.0, "thickness": 60.0,
        "facies": "clean_sandstone", "permeability_md": 2500.0}]
    pipeline = Pipeline(config)
    geology = pipeline.geology()
    mask = body_mask(build_bodies(config.geology.bodies)[0], geology.grid)
    assert np.allclose(geology.permeability_md[mask], 2500.0)
    assert any("geobody: ch1" in n for n in pipeline.result.notes)


def test_a_drawn_seal_changes_where_the_water_goes():
    """The claim the feature rests on, tested end to end through the flow.

    A shale channel laid across the line between injector and producer is
    the interesting case: it removes reservoir rather than adding it, and
    if the simulator does not see it then "draw a channel and simulate it"
    is a drawing with a caption.
    """
    from tests.test_pipeline import tiny_config

    plain = Pipeline(tiny_config())
    config = tiny_config()
    (x0, x1), (y0, y1), _ = plain.domains.target.bounds
    config.geology.bodies = [{
        "name": "seal", "type": "channel", "facies": "shale",
        "path": [[0.5 * (x0 + x1), y0 - 200.0], [0.5 * (x0 + x1), y1 + 200.0]],
        "width": 120.0, "top": 880.0, "thickness": 120.0}]
    sealed = Pipeline(config)

    before = plain.reservoir().combined.sw
    after = sealed.reservoir().combined.sw
    assert not np.allclose(before, after)
    assert np.abs(before - after).max() > 0.01
