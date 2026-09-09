"""Display layer: colour policy, figure construction, and the app's contract.

The GUI is a frontend, so what is worth testing is that it *stays* one -
that it holds no science of its own - plus the colour rules, which are the
part a screenshot review would not catch reliably.
"""

import importlib.util

import numpy as np
import pytest

from sim3d.core.grid import Grid3D

plotly = pytest.importorskip("plotly")

from sim3d.ui import components as ui   # noqa: E402
from sim3d.ui import theme               # noqa: E402


@pytest.fixture(scope="module")
def grid():
    return Grid3D.from_bounds(((0, 1000), (0, 1000), (0, 800)), (50, 50, 40))


# ------------------------------------------------------------------- colour
def test_the_sequential_ramp_is_one_hue_and_monotonically_darker():
    """A magnitude ramp must not change hue, and must never be a rainbow."""
    def luminance(hexcode):
        r, g, b = (int(hexcode[i:i + 2], 16) / 255 for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    steps = [c for _, c in theme.SEQUENTIAL]
    values = [luminance(c) for c in steps]
    assert all(a > b for a, b in zip(values, values[1:]))
    # One hue: blue dominates every step.
    for c in steps:
        r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
        assert b >= g >= r


def test_the_diverging_ramp_is_symmetric_with_a_neutral_midpoint():
    steps = [c for _, c in theme.DIVERGING]
    assert len(steps) % 2 == 1
    middle = steps[len(steps) // 2]
    assert middle == theme.DIVERGING_MID
    r, g, b = (int(middle[i:i + 2], 16) for i in (1, 3, 5))
    assert max(r, g, b) - min(r, g, b) < 12   # the midpoint reads as "nothing"
    # Equal arms, and the poles are opposite in hue.
    low, high = steps[0], steps[-1]
    assert int(low[1:3], 16) > int(low[5:7], 16)    # negative pole is red
    assert int(high[5:7], 16) > int(high[1:3], 16)  # positive pole is blue


def test_a_scenario_keeps_its_colour_everywhere():
    """Colour follows the entity, never its rank, so a filter that changes
    which scenarios are shown must not repaint the survivors."""
    assert len(set(theme.SCENARIO_COLOUR.values())) == len(theme.SCENARIO_COLOUR)
    for name in ("baseline", "pressure_only", "saturation_only", "combined"):
        assert name in theme.SCENARIO_COLOUR


def test_status_colours_are_distinct_from_the_series_palette():
    assert not (set(theme.STATUS.values()) & set(theme.SERIES))
    assert set(theme.STATUS) == {"PASS", "WARNING", "FAIL"}


# ------------------------------------------------------------------ figures
def test_signed_volumes_get_symmetric_limits_and_unsigned_ones_do_not(grid):
    """A signed field on a one-sided scale would hide its sign - the single
    most consequential display mistake this application could make."""
    skewed = np.linspace(-1.0, 9.0, grid.n_cells).reshape(grid.shape)

    diverging = ui.slice_figure(skewed, grid, (500, 500, 400), title="d",
                                kind="diverging")
    zmin, zmax = diverging.data[0].zmin, diverging.data[0].zmax
    assert zmin == pytest.approx(-zmax)

    sequential = ui.slice_figure(skewed, grid, (500, 500, 400), title="s",
                                 kind="sequential")
    assert sequential.data[0].zmin != pytest.approx(-sequential.data[0].zmax)


def test_a_single_outlier_does_not_flatten_the_display(grid):
    volume = np.random.default_rng(0).standard_normal(grid.shape)
    volume[0, 0, 0] = 1e6
    figure = ui.slice_figure(volume, grid, (500, 500, 400), title="t",
                             kind="diverging")
    assert figure.data[0].zmax < 100.0


def test_slice_figure_cuts_at_the_requested_point(grid):
    volume = np.zeros(grid.shape)
    volume[:, :, 5] = 1.0                       # a single bright layer
    z = grid.axis(2)[5]
    figure = ui.slice_figure(volume, grid, (500.0, 500.0, z), title="t")
    depth_panel = np.asarray(figure.data[2].z)  # the depth slice
    assert np.allclose(depth_panel, 1.0)


def test_slice_figure_shows_three_panels_and_the_wells(grid):
    class Well:
        name, role, x, y = "I1", "injector", 400.0, 600.0

    figure = ui.slice_figure(np.zeros(grid.shape), grid, (500, 500, 400),
                             title="t", wells=[Well()])
    heatmaps = [t for t in figure.data if t.type == "heatmap"]
    scatters = [t for t in figure.data if t.type == "scatter"]
    assert len(heatmaps) == 3 and len(scatters) == 1
    assert scatters[0].x == (400.0,)


def test_depth_axes_point_downwards(grid):
    figure = ui.slice_figure(np.zeros(grid.shape), grid, (500, 500, 400), title="t")
    assert figure.layout.yaxis.autorange == "reversed"    # inline section
    assert figure.layout.yaxis2.autorange == "reversed"   # crossline section
    assert figure.layout.yaxis3.autorange != "reversed"   # map view


def test_a_histogram_is_drawn_as_bars_not_a_line():
    """Joining bin tops with a line implies continuity the data lacks."""
    figure = ui.bar_figure(np.arange(5) * 100.0, np.array([3, 9, 14, 8, 2]),
                           xlabel="offset (m)", ylabel="traces")
    assert figure.data[0].type == "bar"


def test_one_series_carries_no_legend_and_several_carry_both_cues():
    x = np.arange(10.0)
    single = ui.series_figure(x, {"only": x}, xlabel="r", ylabel="v")
    assert not single.data[0].showlegend
    assert len(single.layout.annotations) == 0

    several = ui.series_figure(x, {"a": x, "b": x * 2, "c": x * 3},
                               xlabel="r", ylabel="v")
    assert all(trace.showlegend for trace in several.data)
    # Direct end-labels as well, because two categorical slots are below 3:1.
    assert len(several.layout.annotations) == 3


def test_series_colours_can_be_pinned_per_entity():
    x = np.arange(5.0)
    figure = ui.series_figure(x, {"combined": x}, xlabel="r", ylabel="v",
                              colours=theme.SCENARIO_COLOUR)
    assert figure.data[0].line.color == theme.SCENARIO_COLOUR["combined"]


def test_gather_figure_puts_time_downwards_and_uses_a_signed_scale():
    traces = np.random.default_rng(0).standard_normal((12, 200))
    figure = ui.gather_figure(traces, 0.002, title="shot 1")
    assert figure.layout.yaxis.autorange == "reversed"
    assert figure.data[0].zmin == pytest.approx(-figure.data[0].zmax)


def test_map_figure_draws_the_domain_and_the_absorbing_layer(grid):
    class Well:
        name, role, x, y = "P1", "producer", 500.0, 500.0

    class Acquisition:
        sources = np.array([[300.0, 300.0, 100.0]])
        receivers = np.array([[600.0, 600.0, 100.0]])
        n_sources, n_receivers = 1, 1

    figure = ui.map_figure(wells=[Well()], acquisition=Acquisition(), grid=grid,
                           pml_nodes=3)
    assert len(figure.layout.shapes) == 2      # domain and inner PML edge
    assert figure.layout.yaxis.scaleanchor == "x"   # map view is not stretched


def test_nearest_snaps_to_the_closest_node(grid):
    assert ui.nearest(grid.axis(0), 137.0) == 3    # 150 m at 50 m spacing
    assert ui.nearest(grid.axis(2), -1e9) == 0


# ---------------------------------------------------------------- the app
@pytest.mark.skipif(importlib.util.find_spec("streamlit") is None,
                    reason="streamlit not installed")
def test_the_app_holds_no_science_of_its_own():
    """Spec section 119: Streamlit is a frontend.

    The app may import the engine, but it must not reimplement it: nothing
    in the UI package may import a solver, a rock-physics model or an
    imaging routine directly.
    """
    from pathlib import Path

    source = Path("src/sim3d/ui/streamlit_app.py").read_text()
    for forbidden in ("AcousticSolver", "elastic_from_state", "migrate_survey",
                      "gassmann", "build_geology", "PointSource"):
        assert forbidden not in source, (
            f"{forbidden} appears in the GUI; it belongs in the pipeline")


@pytest.mark.slow
def test_the_simulation_and_4d_views_build_from_real_results():
    """Exercise the display paths a screenshot of an idle app cannot reach.

    Runs the whole chain on the smallest configuration and constructs the
    figures the Simulation and 4D pages build, so a change that breaks them
    fails here rather than in front of a user with a migrated survey.
    """
    from test_pipeline import tiny_config

    from sim3d.experiments.pipeline import Pipeline
    from sim3d.fourd.metrics import radial_profile
    from sim3d.fourd.scenarios import SCENARIO_NAMES

    pipe = Pipeline(tiny_config())
    pipe.simulate()
    pipe.migrate()
    parts = pipe.decompose()

    record = pipe.result.gathers["baseline"][0]
    gather = ui.gather_figure(record.traces, record.dt, title="shot 1")
    assert np.isfinite(gather.data[0].zmax) and gather.data[0].zmax > 0

    grid = pipe.domains.propagation
    point = tuple(0.5 * (lo + hi) for lo, hi in pipe.domains.target.bounds)
    for name in SCENARIO_NAMES:
        figure = ui.slice_figure(pipe.result.images[name].image, grid, point,
                                 title=name, kind="diverging", unit="amplitude",
                                 wells=list(pipe.wells()))
        assert figure.data[0].zmin == pytest.approx(-figure.data[0].zmax)

    seismic = parts["seismic"]["rtm"]
    for key in ("d_pressure", "d_saturation", "d_combined", "d_interaction"):
        assert ui.slice_figure(seismic[key], grid, point, title=key,
                               kind="diverging").data[0].zmax >= 0

    well = list(pipe.wells())[0]
    radii, means, _ = radial_profile(grid, well, np.abs(seismic["d_combined"]),
                                     bin_width=100.0, max_radius=400.0)
    profile = ui.series_figure(radii, {"combined": means, "pressure": means * 0.5},
                               xlabel="distance (m)", ylabel="|amplitude|",
                               colours=theme.SCENARIO_COLOUR)
    assert len(profile.data) == 2


@pytest.mark.skipif(importlib.util.find_spec("streamlit") is None,
                    reason="streamlit not installed")
def test_every_advertised_page_has_a_function():
    from sim3d.ui import streamlit_app

    assert set(streamlit_app.PAGES) == set(streamlit_app.PAGE_FUNCTIONS)
    assert all(callable(f) for f in streamlit_app.PAGE_FUNCTIONS.values())


def test_every_pressure_field_is_displayed_in_psi():
    """Requirement 8, at the layer that actually paints the colourbar.

    ``core.units.to_display`` was already pinned to psi, but the figures do
    not go through it - they read ``theme.DISPLAY`` - so the two tables can
    disagree, and did: pressure and dP were scaled to bar.
    """
    from sim3d.core.units import PSI

    for field in ("pressure", "dP"):
        scale, unit, _ = theme.DISPLAY[field]
        assert unit == "psi"
        assert scale == pytest.approx(PSI)
    assert not [f for f, (_, unit, _) in theme.DISPLAY.items() if unit == "bar"]


def test_a_trace_is_drawn_against_a_downward_axis():
    """A trace is read top-down, like every other section in the app."""
    y = np.linspace(0.0, 1.0, 64)
    figure = ui.trace_figure(y, {"baseline": np.sin(y * 20)}, ylabel="two-way time (s)")
    assert figure.layout.yaxis.autorange == "reversed"
    assert figure.data[0].y[0] == pytest.approx(0.0)


def test_two_traces_carry_a_legend_and_direct_labels():
    y = np.linspace(0.0, 1.0, 32)
    figure = ui.trace_figure(y, {"baseline": np.sin(y * 9), "combined": np.ones(32)},
                             ylabel="depth (m)")
    assert all(trace.showlegend for trace in figure.data)
    assert len(figure.layout.annotations) == 2


def test_one_trace_needs_no_legend():
    y = np.linspace(0.0, 1.0, 32)
    figure = ui.trace_figure(y, {"baseline": np.zeros(32)}, ylabel="depth (m)")
    assert not figure.data[0].showlegend
    assert not figure.layout.annotations


def test_trace_labels_sit_at_each_extremum_not_at_the_tail():
    """A trace tails off to zero, so end labels would collide into a smear."""
    y = np.linspace(0.0, 1.0, 100)
    a, b = np.zeros(100), np.zeros(100)
    a[20], b[70] = 1.0, -1.0
    figure = ui.trace_figure(y, {"a": a, "b": b}, ylabel="two-way time (s)")
    at = {n.text.strip(): (n.x, n.y) for n in figure.layout.annotations}
    assert len(set(at.values())) == 2                # not stacked
    assert at["a"][0] == 1.0 and at["a"][1] == pytest.approx(y[20])
    assert at["b"][0] == -1.0 and at["b"][1] == pytest.approx(y[70])
    # A negative peak is labelled on its own side, so the text runs outward.
    sides = {n.text.strip(): n.xanchor for n in figure.layout.annotations}
    assert sides["a"] == "left" and sides["b"] == "right"


def test_an_all_zero_trace_is_not_labelled():
    y = np.linspace(0.0, 1.0, 32)
    figure = ui.trace_figure(y, {"flat": np.zeros(32), "also": np.zeros(32)},
                             ylabel="depth (m)")
    assert not figure.layout.annotations


def test_labels_are_dropped_when_the_extrema_coincide():
    """Four near-identical traces peak on the same event; no anchor separates
    them, so overprinting is worse than leaving the legend to do the work."""
    y = np.linspace(0.0, 1.0, 100)
    base = np.zeros(100)
    base[40] = 1.0
    figure = ui.trace_figure(
        y, {"baseline": base, "combined": base * 1.001, "pressure_only": base * 0.999},
        ylabel="two-way time (s)")
    assert not figure.layout.annotations
    assert all(trace.showlegend for trace in figure.data)   # identity survives


def test_a_section_carries_real_units_on_both_axes():
    """Unlike a gather, which numbers its traces."""
    data = np.random.default_rng(0).normal(size=(40, 20))
    figure = ui.section_figure(data, np.linspace(0, 1000, 20),
                               np.linspace(0, 1.6, 40),
                               xlabel="x (m)", ylabel="two-way time (s)")
    assert figure.layout.yaxis.autorange == "reversed"
    assert figure.data[0].x[-1] == pytest.approx(1000.0)
    assert figure.data[0].y[-1] == pytest.approx(1.6)


def test_a_section_is_symmetric_about_zero():
    """Signed amplitude needs a diverging scale with matched limits."""
    data = np.concatenate([np.full((5, 4), -1.0), np.full((5, 4), 4.0)])
    figure = ui.section_figure(data, np.arange(4), np.arange(10),
                               xlabel="x (m)", ylabel="t (s)", robust=False)
    assert figure.data[0].zmin == pytest.approx(-figure.data[0].zmax)
