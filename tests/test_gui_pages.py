"""Every page of the app opens, on a real configuration.

The gap this closes: two imports were lost from page_acquisition in a
refactor and nothing failed, because no test in the suite opened a page.
The NameError waited for a user to click "Acquisition & QC".

These use Streamlit's AppTest, which runs the real script - widgets,
session state, reruns - without a browser. They deliberately do not click
anything expensive: the point is that every page *renders*, which is where
import and attribute errors surface. Running the physics is what the rest
of the suite is for.
"""

import pathlib

import pytest

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest    # noqa: E402

#: Absolute, because pytest's working directory is not the repository root.
APP = str(pathlib.Path(__file__).resolve().parent.parent
          / "src" / "sim3d" / "ui" / "streamlit_app.py")
PAGES = [
    "Project", "Geology", "3D Model", "Wells & Completions", "Flow Simulation",
    "Rock Physics", "Synthetic Volume", "Acquisition & QC",
    "Simulation & Imaging", "4D Analysis", "Scenarios",
]


def _app(timeout: int = 600) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=timeout)
    app.run()
    return app


def _goto(app: AppTest, page: str) -> AppTest:
    for radio in app.radio:
        if radio.options and page in radio.options:
            radio.set_value(page)
            break
    else:
        raise AssertionError(f"no page selector offers {page!r}")
    app.run()
    return app


def test_the_app_starts():
    app = _app()
    assert not app.exception, app.exception[0].value


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    """One test per page, so a failure names the page rather than the suite."""
    app = _goto(_app(), page)
    assert not app.exception, f"{page}: {app.exception[0].value}"


def test_the_page_list_matches_the_selector():
    """A page function nobody can reach is dead code; a selector entry with
    no function is a crash waiting for a click."""
    from sim3d.ui.streamlit_app import PAGE_FUNCTIONS
    assert set(PAGE_FUNCTIONS) == set(PAGES)


def test_switching_configuration_keeps_every_page_working():
    """The dipping wedge exercises paths the default never reaches: fluid
    contacts, per-facies rock physics and a 40-degree structure."""
    app = _app()
    box = app.sidebar.selectbox[0]
    wedge = [o for o in box.options if "dipping_wedge_4d" in o]
    if not wedge:
        pytest.skip("dipping_wedge_4d.yaml is not on the example list")
    box.set_value(wedge[0])
    app.run()
    assert not app.exception, app.exception[0].value
    for page in ("Geology", "Wells & Completions", "Acquisition & QC"):
        _goto(app, page)
        assert not app.exception, f"{page}: {app.exception[0].value}"


# ------------------------------------------------- the acquisition geometry
def _number(app: AppTest, key: str):
    for widget in app.number_input:
        if widget.key == key:
            return widget
    raise AssertionError(f"no number_input keyed {key!r}; "
                         f"found {[w.key for w in app.number_input]}")


def test_the_acquisition_geometry_is_editable_from_the_page():
    """The gap this closes: the page reported that the standoff was too small
    and the trace spacing aliased the operator, and offered no way to change
    either - everything under `acquisition` and `domains` was file-only."""
    app = _goto(_app(), "Acquisition & QC")
    assert not app.exception, app.exception[0].value

    before = app.session_state.config.acquisition.receiver_spacing
    _number(app, "acq_receiver_spacing").set_value(float(before) / 2.0)
    app.run()
    assert not app.exception, app.exception[0].value
    assert app.session_state.config.acquisition.receiver_spacing == before / 2.0


def test_editing_the_geometry_drops_the_stale_pipeline():
    """Geometry is a scientific input: changing it must invalidate whatever
    was modelled against the old one, not quietly keep it."""
    app = _goto(_app(), "Acquisition & QC")
    assert "pipeline_hash" in app.session_state
    stale = app.session_state["pipeline_hash"]
    _number(app, "acq_source_spacing").set_value(
        float(app.session_state.config.acquisition.source_spacing) + 50.0)
    app.run()
    assert not app.exception, app.exception[0].value
    fresh = (app.session_state["pipeline_hash"]
             if "pipeline_hash" in app.session_state else None)
    assert fresh != stale


def test_the_acquisition_may_not_be_placed_inside_the_absorbing_layer():
    """The inner edge moves with the top of the propagation domain, so the
    widget has to enforce it rather than leave it to QC after the fact."""
    app = _goto(_app(), "Acquisition & QC")
    cfg = app.session_state.config
    spacing = (cfg.domains.propagation_spacing or cfg.domains.geology_spacing)[2]
    pml = cfg.solver.pml_nodes * float(spacing)
    top = (cfg.domains.propagation_bounds or cfg.domains.geology_bounds)[2][0]
    widget = _number(app, "acq_receiver_depth")
    assert widget.min == pytest.approx(float(top) + pml), (
        "the receiver-depth floor must be the absorbing layer's inner edge")


def test_raising_the_domain_top_buys_standoff():
    """The fix the warnings ask for, end to end: the reservoir in demo_small
    sits 1.0 wavelengths below the acquisition, and below about 2 the
    injection near-field overlaps the target."""
    app = _goto(_app(), "Acquisition & QC")
    cfg = app.session_state.config
    original = (cfg.domains.propagation_bounds or cfg.domains.geology_bounds)[2][0]
    geology = [list(b) for b in cfg.domains.geology_bounds]
    _number(app, "acq_domain_top").set_value(float(original) - 400.0)
    app.run()
    assert not app.exception, app.exception[0].value
    moved = app.session_state.config.domains
    assert moved.propagation_bounds[2][0] == original - 400.0
    # The geology domain is untouched: only the modelling window moved.
    assert [list(b) for b in moved.geology_bounds] == geology


# ---------------------------------------------------- choosing the scenarios
def _multiselect(app: AppTest, fragment: str):
    for widget in app.multiselect:
        if fragment.lower() in (widget.label or "").lower():
            return widget
    raise AssertionError(f"no multiselect labelled like {fragment!r}")


def test_the_scenarios_to_simulate_are_chosen_on_the_page():
    """Cost is linear in the list: four earth models is four independent
    propagations of every shot, and "is the image clean?" needs one."""
    app = _goto(_app(), "Simulation & Imaging")
    assert not app.exception, app.exception[0].value
    _multiselect(app, "Earth models").set_value(["baseline"])
    app.run()
    assert not app.exception, app.exception[0].value
    assert list(app.session_state.config.fourd.scenarios) == ["baseline"]


def test_the_baseline_is_always_simulated():
    """Every difference is measured against it; a monitor with nothing to
    subtract is not a 4D result."""
    app = _goto(_app(), "Simulation & Imaging")
    _multiselect(app, "Earth models").set_value(["combined"])
    app.run()
    assert not app.exception, app.exception[0].value
    assert list(app.session_state.config.fourd.scenarios) == ["baseline", "combined"]


def test_the_scenario_order_is_canonical_however_it_was_clicked():
    """`decompose` indexes SCENARIO_NAMES positionally, so the stored order
    cannot be the order the boxes happened to be ticked in."""
    from sim3d.fourd.scenarios import SCENARIO_NAMES
    app = _goto(_app(), "Simulation & Imaging")
    _multiselect(app, "Earth models").set_value(["combined", "pressure_only",
                                                 "baseline"])
    app.run()
    assert not app.exception, app.exception[0].value
    stored = list(app.session_state.config.fourd.scenarios)
    assert stored == [n for n in SCENARIO_NAMES if n in stored]


# ------------------------------------------- a new well joins the pattern
def test_a_new_well_inherits_the_controls_of_its_own_role():
    """The gap this closes: left to the automatic suggestion, a well dropped
    next to an existing one is rated by pattern scale over a drainage radius
    of half the distance to its neighbour. On the dipping wedge that gives a
    producer 200 m from P1 a target of 247 STB/day against P1's 2,250."""
    from sim3d.ui.streamlit_app import _new_well_spec
    from sim3d.core.config import ExperimentConfig
    from sim3d.experiments.pipeline import Pipeline

    cfg = ExperimentConfig.load("examples/configs/dipping_wedge_4d.yaml")
    grid = Pipeline(cfg).domains.geology
    specs = cfg.wells.wells
    producer = next(w for w in specs if w["role"] == "producer")

    fresh = _new_well_spec(specs, "producer", 400.0, 700.0, grid)
    assert fresh["target"] == producer["target"]
    assert fresh["control"] == producer["control"]
    assert fresh["bhp_limit_psi"] == producer["bhp_limit_psi"]
    assert fresh["completions"] == producer["completions"]
    assert fresh["name"] not in {w["name"] for w in specs}


def test_a_new_well_copies_its_own_role_not_the_first_well_it_finds():
    """An injector must not inherit a producer's control mode: the two are
    not interchangeable, and water_rate on a producer is nonsense."""
    from sim3d.ui.streamlit_app import _new_well_spec
    from sim3d.core.config import ExperimentConfig
    from sim3d.experiments.pipeline import Pipeline

    cfg = ExperimentConfig.load("examples/configs/dipping_wedge_4d.yaml")
    grid = Pipeline(cfg).domains.geology
    injector = next(w for w in cfg.wells.wells if w["role"] == "injector")
    fresh = _new_well_spec(cfg.wells.wells, "injector", 600.0, 700.0, grid)
    assert fresh["role"] == "injector"
    assert fresh["control"] == injector["control"]
    assert fresh["target"] == injector["target"]


def test_the_first_well_of_a_role_keeps_the_automatic_suggestion():
    """With no well of that role to copy there is no pattern to be out of
    step with, so the suggestion stands."""
    from sim3d.ui.streamlit_app import _new_well_spec
    from sim3d.core.config import ExperimentConfig
    from sim3d.experiments.pipeline import Pipeline

    cfg = ExperimentConfig.load("examples/configs/dipping_wedge_4d.yaml")
    grid = Pipeline(cfg).domains.geology
    producers = [w for w in cfg.wells.wells if w["role"] == "producer"]
    fresh = _new_well_spec(producers, "injector", 600.0, 700.0, grid)
    assert fresh["target"] is None
    assert fresh["control"] == "water_rate"


# --------------------------------------------------------------- time spent
def test_every_timed_stage_is_named_in_the_table_order():
    """A stage the pipeline times but the table does not know about falls to
    the end of the list, which is survivable but reads as an afterthought.
    This fails when a new stage is added and the order is not updated."""
    import re
    import pathlib as _pathlib
    from sim3d.ui.streamlit_app import STAGE_ORDER

    source = (_pathlib.Path(__file__).resolve().parent.parent / "src" / "sim3d"
              / "experiments" / "pipeline.py").read_text(encoding="utf-8")
    timed = set(re.findall(r'_timed\(\s*"([^"]+)"', source))
    timed |= set(re.findall(r'timings\["([^"]+)"\]', source))
    missing = timed - set(STAGE_ORDER)
    assert not missing, f"pipeline times {sorted(missing)}, STAGE_ORDER does not"


def test_the_timings_table_reports_a_share_of_the_total():
    from sim3d.ui.streamlit_app import _timings_table

    class _Result:
        timings = {"geology": 1.0, "flow": 3.0}

    class _Pipe:
        result = _Result()

    table = _timings_table(_Pipe())
    assert list(table["elapsed"]) == ["geology", "flow"]      # pipeline order
    assert table["share"]["flow"] == "75%"
    assert table["elapsed"]["geology"] == "1.0 s"


def test_nothing_computed_yet_is_not_an_empty_table():
    from sim3d.ui.streamlit_app import _timings_table

    class _Pipe:
        class result:
            timings: dict = {}

    assert _timings_table(_Pipe()) is None


def test_durations_read_without_counting_zeros():
    from sim3d.ui.streamlit_app import _format_seconds
    assert _format_seconds(0.42) == "420 ms"
    assert _format_seconds(12.3) == "12.3 s"
    assert _format_seconds(600.0) == "10.0 min"
    assert _format_seconds(9489.0) == "2.64 h"
