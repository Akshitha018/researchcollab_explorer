"""End-to-end smoke test: run the real Streamlit app headlessly on the bundled sample.

Uses Streamlit's built-in ``AppTest`` (no browser needed). It checks that every page
renders without an exception and that search, filters, reset and export behave.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_FILE = str(Path(__file__).resolve().parents[1] / "app.py")

PAGES = [
    "📊 Overview",
    "🕸️ Collaboration network",
    "🧩 Communities",
    "👤 Researcher analytics",
    "⬇️ Export & download",
    "📘 About & methodology",
]


@pytest.fixture
def app():
    at = AppTest.from_file(APP_FILE, default_timeout=120)
    at.session_state["source_mode"] = "Bundled sample"
    at.run()
    return at


def go(at, page):
    at.sidebar.radio[0].set_value(page).run()


def search_box(at):
    return [w for w in at.sidebar.text_input if w.label == "Search researcher ID"][0]


def metrics(at):
    return {m.label: m.value for m in at.metric}


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exceptions(app, page):
    go(app, page)
    assert not app.exception
    assert not app.error


def test_overview_has_no_dataset_error_and_shows_dataframe(app):
    assert not app.exception
    assert len(app.dataframe) >= 1  # the cleaning report


def test_search_selects_researcher_and_validates_input(app):
    go(app, "🕸️ Collaboration network")
    search_box(app).set_value("abc").run()
    assert any("whole numbers" in w.value for w in app.sidebar.warning)
    search_box(app).set_value("99999999").run()
    assert any("not in the loaded network" in w.value for w in app.sidebar.warning)
    search_box(app).set_value("4").run()
    assert any("Selected researcher 4" in s.value for s in app.sidebar.success)
    assert "Degree" in metrics(app)  # researcher detail panel is shown


def test_neighbourhood_filter_shrinks_the_view(app):
    go(app, "🕸️ Collaboration network")
    search_box(app).set_value("4").run()
    box = [c for c in app.sidebar.checkbox if "neighbourhood" in c.label][0]
    box.set_value(True).run()
    # researcher 4 has 6 collaborators in the sample -> 7 researchers shown
    assert metrics(app)["Researchers shown"] == "7 of 30"


def test_min_degree_filter_and_reset(app):
    go(app, "🕸️ Collaboration network")
    [s for s in app.sidebar.slider if s.label == "Minimum degree"][0].set_value(5).run()
    shown = int(metrics(app)["Researchers shown"].split(" of ")[0])
    assert shown < 30
    [b for b in app.sidebar.button if b.label == "Reset filters"][0].click().run()
    assert metrics(app)["Researchers shown"] == "30 of 30"


def test_export_scope_follows_filters(app):
    go(app, "🕸️ Collaboration network")
    [s for s in app.sidebar.slider if s.label == "Minimum degree"][0].set_value(5).run()
    go(app, "⬇️ Export & download")
    assert metrics(app)["Researchers in export"] == "30"
    [r for r in app.radio if r.label.startswith("What should")][0].set_value(
        "Current filtered view"
    ).run()
    assert int(metrics(app)["Researchers in export"]) < 30
    assert not app.exception


def test_missing_local_file_shows_helpful_message_not_a_crash():
    at = AppTest.from_file(APP_FILE, default_timeout=120)
    at.session_state["source_mode"] = "Local file"
    at.session_state["dataset_path"] = "data/this_file_does_not_exist.txt.gz"
    at.run()
    assert not at.exception
    assert any("not found" in w.value for w in at.sidebar.warning)
    assert any("No dataset loaded" in md.value for md in at.markdown)
