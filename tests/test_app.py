from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


def test_dashboard_pages_search_filters_and_exports():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=45).run()
    assert not app.exception
    assert any("OFFLINE SAMPLE DATASET" in item.value for item in app.markdown)

    app.radio[1].set_value("Interactive Network Graph").run()
    assert not app.exception

    researcher_filter = next(
        item for item in app.selectbox
        if "Ego-Network" in item.label
    )
    researcher_filter.set_value(researcher_filter.options[1]).run()
    app.slider[0].set_value((app.slider[0].value[1], app.slider[0].value[1])).run()
    app.slider[1].set_value(1).run()
    search = next(item for item in app.text_input if "Quick Search" in item.label)
    search.set_value("Kipf").run()
    assert not app.exception

    app.radio[1].set_value("Graph & Centrality Analysis").run()
    assert len(app.get("plotly_chart")) == 2
    institution_filter = next(
        item for item in app.selectbox
        if item.label == "Filter by Institution"
    )
    institution_filter.set_value(institution_filter.options[1]).run()
    assert not app.exception

    app.radio[1].set_value("Community Detection").run()
    assert len(app.get("plotly_chart")) == 1
    app.radio[1].set_value("Researcher Profiles").run()
    profile_selector = next(
        item for item in app.selectbox
        if item.label == "Select Researcher Dossier"
    )
    profile_selector.set_value(profile_selector.options[1]).run()
    assert len(app.get("plotly_chart")) == 1
    assert not app.exception

    app.radio[1].set_value("Data Tables & Exports").run()
    download_labels = [item.label for item in app.get("download_button")]
    assert any("Authorships CSV" in label for label in download_labels)
    assert any("GraphML" in label for label in download_labels)
    assert not app.exception

    app.radio[1].set_value("Methodology & Viva Guide").run()
    assert not app.exception


def test_switching_ingestion_mode_keeps_sample_data_explicit():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=45).run()
    app.radio[0].set_value("Live OpenAlex API").run()
    assert any("OFFLINE SAMPLE DATASET" in item.value for item in app.markdown)

    with patch(
        "modules.data_loader.fetch_openalex_works",
        return_value=(None, "Could not connect to OpenAlex.")
    ):
        api_query = next(item for item in app.button if item.label == "Query API")
        api_query.click().run()
    assert any("OFFLINE SAMPLE DATASET" in item.value for item in app.markdown)
    assert any("loaded dataset was not changed" in item.value for item in app.info)
    assert not app.exception


def test_successful_api_query_sets_live_source_badge():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=45).run()
    app.radio[0].set_value("Live OpenAlex API").run()
    live_dataset = {
        "is_live": True,
        "dataset_name": "Live OpenAlex test corpus",
        "source_note": "Live OpenAlex API Ingestion",
        "publications": [{
            "id": "W1",
            "title": "Live test work",
            "publication_year": 2024,
            "cited_by_count": 1,
            "primary_topic": "Test",
            "doi": "",
            "authors": [
                {"id": "A1", "name": "Author One"},
                {"id": "A2", "name": "Author Two"}
            ]
        }],
        "authors": {
            "A1": {"name": "Author One", "institution": "Unknown Institution", "works_count": 1, "total_citations": 1},
            "A2": {"name": "Author Two", "institution": "Unknown Institution", "works_count": 1, "total_citations": 1}
        },
        "authorships": []
    }
    with patch(
        "modules.data_loader.fetch_openalex_works",
        return_value=(live_dataset, None)
    ):
        next(item for item in app.button if item.label == "Query API").click().run()

    assert any("LIVE OPENALEX API DATA" in item.value for item in app.markdown)
    assert not any("OFFLINE SAMPLE DATASET" in item.value for item in app.markdown)
    assert not app.exception

    app.radio[0].set_value("Offline Benchmark Sample").run()
    assert any("OFFLINE SAMPLE DATASET" in item.value for item in app.markdown)
    assert not app.exception
