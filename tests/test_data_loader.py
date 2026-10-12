from unittest.mock import Mock, patch

import requests

from modules.data_loader import (
    _fetch_openalex_works_cached,
    fetch_openalex_works,
)


def _successful_response():
    response = Mock(status_code=200, headers={}, text="")
    response.json.return_value = {
        "meta": {"count": 1},
        "results": [{
            "id": "https://openalex.org/W1",
            "display_name": "A test publication",
            "publication_year": 2024,
            "authorships": [{
                "author": {
                    "id": "https://openalex.org/A1",
                    "display_name": "Test Researcher"
                },
                "institutions": []
            }]
        }]
    }
    return response


def test_openalex_success_is_cached_and_uses_bounded_timeout():
    _fetch_openalex_works_cached.clear()
    response = _successful_response()
    with patch("modules.data_loader.requests.get", return_value=response) as get:
        first, first_error = fetch_openalex_works(
            "cache regression test",
            limit=500,
            email="researcher@example.org"
        )
        second, second_error = fetch_openalex_works(
            "cache regression test",
            limit=500,
            email="researcher@example.org"
        )

    assert first_error is None
    assert second_error is None
    assert first == second
    assert first["is_live"] is True
    assert get.call_count == 1
    assert get.call_args.kwargs["params"]["per_page"] == 100
    assert get.call_args.kwargs["params"]["search"] == "cache regression test"
    assert get.call_args.kwargs["params"]["filter"] == "is_paratext:false"
    assert get.call_args.kwargs["params"]["sort"] == "cited_by_count:desc"
    assert get.call_args.kwargs["timeout"] == (3.05, 12)
    _fetch_openalex_works_cached.clear()


def test_openalex_timeout_returns_actionable_error():
    with patch(
        "modules.data_loader.requests.get",
        side_effect=requests.exceptions.Timeout
    ):
        data, error = fetch_openalex_works("timeout regression test")

    assert data is None
    assert error is not None
    assert "timed out" in error
    assert "offline sample" in error


def test_openalex_rate_limit_and_malformed_response_errors():
    rate_limited = Mock(
        status_code=429,
        headers={"Retry-After": "30"},
        text="rate limited"
    )
    with patch("modules.data_loader.requests.get", return_value=rate_limited):
        data, error = fetch_openalex_works("rate limit regression test")
    assert data is None
    assert "Retry after 30 seconds" in error

    malformed = Mock(status_code=200, headers={}, text="")
    malformed.json.side_effect = ValueError("not json")
    with patch("modules.data_loader.requests.get", return_value=malformed):
        data, error = fetch_openalex_works("malformed response regression test")
    assert data is None
    assert "invalid JSON" in error


def test_openalex_empty_results_and_invalid_request_errors():
    empty = Mock(status_code=200, headers={}, text="")
    empty.json.return_value = {"meta": {"count": 0}, "results": []}
    with patch("modules.data_loader.requests.get", return_value=empty):
        data, error = fetch_openalex_works("empty result regression test")
    assert data is None
    assert "No research publications found" in error

    invalid = Mock(
        status_code=400,
        headers={},
        text='{"error":"Invalid query"}'
    )
    with patch("modules.data_loader.requests.get", return_value=invalid):
        data, error = fetch_openalex_works("invalid query regression test")
    assert data is None
    assert "HTTP 400" in error


def test_openalex_connection_failure_does_not_leak_request_details():
    with patch(
        "modules.data_loader.requests.get",
        side_effect=requests.exceptions.ConnectionError(
            "failed request for sensitive-search-term@example.org"
        )
    ):
        data, error = fetch_openalex_works("connection failure regression test")

    assert data is None
    assert "Could not connect to OpenAlex" in error
    assert "sensitive-search-term" not in error


def test_openalex_does_not_send_placeholder_contact_email():
    import os

    response = _successful_response()
    with patch.dict(os.environ, {"OPENALEX_EMAIL": ""}):
        with patch("modules.data_loader.requests.get", return_value=response) as get:
            data, error = fetch_openalex_works("no contact header regression test")

    assert error is None
    assert data is not None
    user_agent = get.call_args.kwargs["headers"]["User-Agent"]
    assert user_agent == "ResearchCollab-Explorer/1.0"


def test_openalex_parser_handles_current_topic_and_missing_metadata():
    from modules.data_loader import parse_openalex_api_results

    parsed = parse_openalex_api_results(
        [{
            "id": "https://openalex.org/W2",
            "display_name": "Topic and missing metadata",
            "publication_year": None,
            "primary_topic": {"display_name": "Primary topic"},
            "topics": [{"display_name": "Other topic"}],
            "authorships": [
                {
                    "author": {
                        "id": "https://openalex.org/A2",
                        "display_name": "Named author"
                    },
                    "institutions": []
                },
                {"author": None, "institutions": []}
            ]
        }],
        query="parser regression test",
        total_count=None
    )

    publication = parsed["publications"][0]
    assert publication["publication_year"] == 0
    assert publication["primary_topic"] == "Primary topic"
    assert [author["id"] for author in publication["authors"]] == ["A2"]
    assert list(parsed["authors"]) == ["A2"]
    assert "Matched ~" not in parsed["source_note"]


def test_openalex_rejects_empty_query_and_invalid_timeout_without_request():
    with patch("modules.data_loader.requests.get") as get:
        empty_data, empty_error = fetch_openalex_works("  ")
        timeout_data, timeout_error = fetch_openalex_works("valid query", timeout=0)

    assert empty_data is None
    assert "cannot be empty" in empty_error
    assert timeout_data is None
    assert "greater than zero" in timeout_error
    get.assert_not_called()
