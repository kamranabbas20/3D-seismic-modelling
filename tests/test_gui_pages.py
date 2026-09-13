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
