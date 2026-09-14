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
