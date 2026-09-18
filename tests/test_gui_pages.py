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
    "Migration Setup", "4D Analysis", "Scenarios",
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
    app = _goto(_app(), "Migration Setup")
    assert not app.exception, app.exception[0].value
    _multiselect(app, "Earth models").set_value(["baseline"])
    app.run()
    assert not app.exception, app.exception[0].value
    assert list(app.session_state.config.fourd.scenarios) == ["baseline"]


def test_the_baseline_is_always_simulated():
    """Every difference is measured against it; a monitor with nothing to
    subtract is not a 4D result."""
    app = _goto(_app(), "Migration Setup")
    _multiselect(app, "Earth models").set_value(["combined"])
    app.run()
    assert not app.exception, app.exception[0].value
    assert list(app.session_state.config.fourd.scenarios) == ["baseline", "combined"]


def test_the_scenario_order_is_canonical_however_it_was_clicked():
    """`decompose` indexes SCENARIO_NAMES positionally, so the stored order
    cannot be the order the boxes happened to be ticked in."""
    from sim3d.fourd.scenarios import SCENARIO_NAMES
    app = _goto(_app(), "Migration Setup")
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


# ------------------------------------- setup and modelling, not execution
def test_no_page_offers_to_run_the_wave_modelling_or_the_migration():
    """The architectural claim, asserted rather than described.

    A survey worth migrating is hours of work and a browser session is not a
    batch queue: the app decides *what* to run and writes it out, and the
    running happens in `run_migration.py` somewhere with cores. If a button
    that propagates a wavefield ever reappears here, this fails.
    """
    banned = ("run full-wave", "run 3d rtm", "run rtm", "migrate")
    seen = []
    for page in PAGES:
        app = _goto(_app(), page)
        assert not app.exception, f"{page}: {app.exception[0].value}"
        for button in app.button:
            label = (button.label or "").lower()
            seen.append(label)
            assert not any(b in label for b in banned), \
                f"{page!r} offers {button.label!r}"
    # Without this the test passes just as happily if the buttons stop being
    # discoverable at all, which would make it worthless exactly when the app
    # had been restructured.
    assert seen, "found no buttons anywhere - the test is not looking at them"


def test_the_migration_page_exports_a_configuration_that_loads_back():
    """The page's whole output is a file, so the file has to be real.

    `to_dict` round-trips through `from_dict`, and this is the check that it
    still does for whatever the widgets have just written into the config -
    a YAML the runner cannot load is worse than no YAML at all.
    """
    import yaml
    from sim3d.core.config import ExperimentConfig

    app = _goto(_app(), "Migration Setup")
    assert not app.exception, app.exception[0].value
    text = yaml.safe_dump(app.session_state.config.to_dict(), sort_keys=False)
    rebuilt = ExperimentConfig.from_dict(yaml.safe_load(text))
    assert rebuilt.content_hash() == app.session_state.config.content_hash()


def test_the_configuration_can_be_saved_from_every_page():
    """Losing a session's setup to a browser reload is how people stop using
    a tool, so the download is in the sidebar rather than on one page."""
    for page in ("Project", "Geology", "Migration Setup", "4D Analysis"):
        app = _goto(_app(), page)
        assert not app.exception, f"{page}: {app.exception[0].value}"
        # A download is its own element type, not a button.
        labels = [(b.label or "").lower() for b in app.download_button]
        assert any("save configuration" in l for l in labels), \
            f"{page!r} has no save button, only {labels}"


# --------------------------------------------- the survey footprint editor
def _checkbox(app, fragment: str):
    for widget in app.checkbox:
        if fragment.lower() in (widget.label or "").lower():
            return widget
    raise AssertionError(
        f"no checkbox like {fragment!r}; saw {[c.label for c in app.checkbox]}")


def test_the_survey_footprint_is_editable_from_the_migration_page():
    """Coverage was configuration-file only, and it is the knob that decides
    whether the anomaly is illuminated at all.

    Aperture scales with depth rather than with target size, so the derived
    footprint - target plus a margin - is right until the target is narrowed
    for a 2.5D line, at which point the inline extent shrinks with it and the
    survey collapses to a couple of shots. Being able to pin it matters.
    """
    app = _goto(_app(), "Migration Setup")
    assert not app.exception, app.exception[0].value
    cfg = app.session_state.config
    assert cfg.acquisition.receiver_extent is None    # derived, to begin with

    _checkbox(app, "Set the survey footprint explicitly").set_value(True)
    app.run()
    assert not app.exception, app.exception[0].value

    _number(app, "acq_source_extent").set_value(1800.0)
    app.run()
    assert not app.exception, app.exception[0].value
    assert app.session_state.config.acquisition.source_extent == 1800.0


def test_the_shot_spacing_is_editable_where_the_operator_limit_is_reported():
    """The number that decides whether the migration aliases, on the page
    that reports the limit it has to beat."""
    app = _goto(_app(), "Migration Setup")
    before = app.session_state.config.acquisition.source_spacing
    _number(app, "acq_source_spacing").set_value(float(before) / 2.0)
    app.run()
    assert not app.exception, app.exception[0].value
    assert app.session_state.config.acquisition.source_spacing == before / 2.0


def test_the_record_length_appears_once_on_the_migration_page():
    """It belongs to the recording, and the acquisition expander also owns
    one — two widgets writing the same field on one page is a bug."""
    app = _goto(_app(), "Migration Setup")
    labels = [(n.label or "").lower() for n in app.number_input]
    assert sum("record length" in l for l in labels) == 1, labels


def test_pinning_the_footprint_does_not_start_from_an_illegal_survey():
    """Ticking the box seeded the extents from the propagation span, which by
    definition contains the absorbing layer: 124 receivers landed inside it
    and the page greeted the user with its own validation error. The seed is
    the footprint already in force — the target plus its margin — which is
    legal by construction."""
    app = _goto(_app(), "Migration Setup")
    _checkbox(app, "Set the survey footprint explicitly").set_value(True)
    app.run()
    assert not app.exception, app.exception[0].value

    cfg = app.session_state.config
    target = cfg.domains.target_bounds
    expected = (target[0][1] - target[0][0]) + 2 * cfg.acquisition.target_margin
    assert _number(app, "acq_source_extent").value == pytest.approx(expected)

    # And the survey it describes actually fits.
    from sim3d.experiments.pipeline import Pipeline
    app.session_state.config.acquisition.source_extent = expected
    app.session_state.config.acquisition.receiver_extent = expected
    assert not Pipeline(app.session_state.config).geometry_qc().failed


def test_the_survey_is_drawn_against_the_model_while_it_is_designed():
    """Distances are read wrongly as numbers.

    A spread that "extends 1,800 m" means nothing until it is seen against
    the model it sits in and the absorbing layer taking its share off each
    face. Both views are geometry only, so they redraw on every edit rather
    than behind a button.
    """
    app = _goto(_app(), "Migration Setup")
    assert not app.exception, app.exception[0].value
    assert len(app.tabs) >= 2
    assert [t.label for t in app.tabs][:2] == ["Plan view", "Elevation"]


def test_the_elevation_pins_its_x_axis_to_the_model():
    """Unpinned, the equal-scale constraint stretched a 2,000 m model out to
    -1,000 - 3,000 m and drew the survey in the middle third of an empty
    figure. Equal scale is the point of the view - a standoff has to read as
    a distance - so the range is pinned instead of the constraint dropped."""
    from sim3d.core.config import ExperimentConfig
    from sim3d.experiments.pipeline import Pipeline
    from sim3d.ui import components as ui

    pipe = Pipeline(ExperimentConfig.load("examples/configs/demo_small.yaml"))
    fig = ui.elevation_figure(acquisition=pipe.acquisition(),
                              domains=pipe.domains,
                              pml_nodes=pipe.config.solver.pml_nodes)
    (gx0, gx1), _, _ = pipe.domains.geology.bounds
    low, high = fig.layout.xaxis.range
    assert low == pytest.approx(gx0, abs=0.05 * (gx1 - gx0))
    assert high == pytest.approx(gx1, abs=0.05 * (gx1 - gx0))
    assert fig.layout.yaxis.scaleanchor == "x"      # still to scale
    assert fig.layout.yaxis.autorange == "reversed"  # depth downward


def _widget(app: AppTest, collection: str, key: str):
    for widget in getattr(app, collection):
        if widget.key == key:
            return widget
    raise AssertionError(f"no {collection} with key {key!r}")


def test_geology_page_offers_every_template():
    """The structure was configuration-only: the page could show you the
    earth and let you retouch one layer's petrophysics, but not change how
    many layers there were, what dipped, or what pinched out."""
    from sim3d.geology.templates import TEMPLATES

    app = _goto(_app(), "Geology")
    chooser = _widget(app, "selectbox", "geo_template")
    assert set(chooser.options) == set(TEMPLATES)
    assert "layer_cake" in chooser.options


def test_choosing_a_template_exposes_its_own_parameters():
    """Read off the signature, so a template that gains a dial gains a
    control without this page being touched."""
    app = _goto(_app(), "Geology")
    _widget(app, "selectbox", "geo_template").set_value("three_layer")
    app.run()
    keys = {widget.key for widget in app.number_input}
    assert "tmpl_three_layer_dip" in keys
    assert "tmpl_three_layer_gross" in keys


def test_the_layer_cake_editor_appears_with_its_units_table():
    app = _goto(_app(), "Geology")
    _widget(app, "selectbox", "geo_template").set_value("layer_cake")
    app.run()
    assert _widget(app, "selectbox", "cake_style").options == [
        "flat", "dipping", "anticline", "syncline"]
    assert _widget(app, "number_input", "cake_datum") is not None
    assert not app.exception


def test_the_geology_page_still_has_no_execution_button():
    """Setup, not execution — the rule the Migration restructure set."""
    app = _goto(_app(), "Geology")
    labels = [button.label for button in app.button]
    assert labels, "no buttons found at all, so this check proves nothing"
    assert not [label for label in labels
                if any(word in label.lower() for word in ("run", "migrate"))]


def test_the_geology_page_is_a_numbered_sequence_of_steps():
    """The controls existed but were scattered; the ask was for structure."""
    app = _goto(_app(), "Geology")
    labels = [label for tab in app.tabs for label in ([tab.label]
              if hasattr(tab, "label") else [])]
    for step in ("1 · Layers", "2 · Structure", "3 · Shape", "4 · Faults"):
        assert step in labels, f"{step!r} missing from {labels}"
    assert "5 · Review and apply" in app.markdown[-1].value or any(
        "5 · Review and apply" in block.value for block in app.markdown)


def test_step_three_shapes_a_layer_with_knee_points():
    """Knee points are typed or clicked - the same numbers either way."""
    app = _goto(_app(), "Geology")
    _widget(app, "selectbox", "geo_template").set_value("layer_cake")
    app.run()
    assert _widget(app, "selectbox", "shape_unit").options
    assert _widget(app, "radio", "shape_axis").options == ["x", "y"]
    # A section to work on, and the option of another.
    assert "+ new section" in _widget(app, "selectbox", "shape_section").options
    assert _widget(app, "button", "shape_undo").disabled
    assert not app.exception


def test_clicking_the_section_is_off_until_it_is_asked_for():
    """The review section is always on screen. If it captured clicks by
    default, a click while reading step 1 would add a knee point to whichever
    layer step 3 happened to have selected."""
    app = _goto(_app(), "Geology")
    _widget(app, "selectbox", "geo_template").set_value("layer_cake")
    app.run()
    toggle = _widget(app, "toggle", "shape_clicking")
    assert not toggle.value
    toggle.set_value(True)
    app.run()
    assert not app.exception


def test_a_fault_can_be_drawn_on_the_map():
    """Faults reached the model only through `fault_compartment`, which makes
    exactly one, at the model centre, with a fixed strike."""
    app = _goto(_app(), "Geology")
    toggle = _widget(app, "toggle", "drawing_fault")
    assert not toggle.value, "drawing must be off until it is asked for"
    toggle.set_value(True)
    app.run()
    assert not app.exception
    for key in ("fault_depth", "fault_dip", "fault_throw", "fault_zone"):
        assert _widget(app, "number_input", key) is not None
    assert _widget(app, "slider", "fault_trans").value == 0.0
    # Nothing to place or clear before a trace is drawn.
    assert _widget(app, "button", "fault_place").disabled
    assert _widget(app, "button", "fault_clear").disabled


def test_placing_a_fault_needs_two_clicks_and_reaches_the_configuration():
    app = _goto(_app(), "Geology")
    _widget(app, "toggle", "drawing_fault").set_value(True)
    app.run()
    app.session_state["fault_path"] = [[600.0, 400.0], [2400.0, 2600.0]]
    app.run()
    place = _widget(app, "button", "fault_place")
    assert not place.disabled, "two points is a trace"
    place.click()
    app.run()
    assert not app.exception
    faults = app.session_state["config"].geology.faults
    assert len(faults) == 1
    assert faults[0]["trace"] == [[600.0, 400.0], [2400.0, 2600.0]]
    assert app.session_state["fault_path"] == [], "the trace is spent"
