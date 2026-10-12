"""
Data loader module for ResearchCollab Explorer.
Handles real OpenAlex API ingestion, polite pool rate limits, caching,
missing-data sanitation, and curated offline fallback datasets.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import requests
import streamlit as st

OPENALEX_WORKS_ENDPOINT = "https://api.openalex.org/works"
DEFAULT_TIMEOUT_SECONDS = 12


class OpenAlexAPIError(Exception):
    """Raised for expected OpenAlex request or response failures."""

OFFLINE_DATASET_MAP = {
    "gnn": {
        "name": "Graph Neural Networks & Geometric DL",
        "file": "sample_gnn.json",
        "description": "Foundational works in GCNs, GATs, GraphSAGE, and Geometric Deep Learning (Kipf, Welling, Leskovec, Bronstein, etc.)"
    },
    "quantum": {
        "name": "Quantum Computing & Information",
        "file": "sample_quantum.json",
        "description": "Milestones in superconducting qubits, quantum supremacy, surface codes, and algorithms (Shor, Preskill, Martinis, Gambetta, etc.)"
    },
    "crispr": {
        "name": "CRISPR-Cas Systems & Genome Editing",
        "file": "sample_crispr.json",
        "description": "Groundbreaking discoveries in Cas9, base editing, prime editing, and screening (Doudna, Charpentier, Zhang, Church, Liu, etc.)"
    }
}


def sanitize_author_id(raw_id: Optional[str], fallback_name: str) -> str:
    """
    Produce a stable, unique author ID from OpenAlex URI or fallback string.
    """
    if raw_id:
        # e.g., 'https://openalex.org/A5023888001' -> 'A5023888001'
        cleaned = raw_id.strip().rstrip("/").split("/")[-1]
        if cleaned:
            return cleaned
    normalized = fallback_name.strip().lower().replace(" ", "_")
    return f"author_{normalized}" if normalized else "author_unknown"


def sanitize_work_id(raw_id: Optional[str], fallback_title: str) -> str:
    """
    Produce a stable, unique publication ID from OpenAlex URI or fallback string.
    """
    if raw_id:
        cleaned = raw_id.strip().rstrip("/").split("/")[-1]
        if cleaned:
            return cleaned
    normalized = fallback_title.strip().lower()[:20].replace(" ", "_")
    return f"work_{normalized}" if normalized else "work_unknown"


def load_offline_dataset(dataset_key: str = "gnn") -> Dict[str, Any]:
    """
    Load a verified local JSON sample dataset for reliable offline analysis.
    Guarantees no live network calls are made.
    """
    key = dataset_key if dataset_key in OFFLINE_DATASET_MAP else "gnn"
    filename = OFFLINE_DATASET_MAP[key]["file"]
    
    # Try relative to current directory or script location
    possible_paths = [
        os.path.join("sample_data", filename),
        os.path.join(os.path.dirname(__file__), "..", "sample_data", filename),
        os.path.abspath(os.path.join("sample_data", filename))
    ]
    
    data = None
    for p in possible_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            break
            
    if not data:
        raise FileNotFoundError(f"Offline sample dataset '{filename}' could not be located.")

    return parse_raw_publications(
        publications_raw=data.get("publications", []),
        dataset_name=data.get("dataset_name", OFFLINE_DATASET_MAP[key]["name"]),
        is_live=False,
        source_note="Curated Offline Sample Dataset (Not Live API)"
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_openalex_works_cached(
    query: str,
    limit: int,
    email: str,
    timeout: int
) -> Dict[str, Any]:
    """Fetch and parse a successful OpenAlex query; failures are not cached."""
    params = {
        "search": query,
        "per_page": limit,
        "filter": "is_paratext:false",
        "sort": "cited_by_count:desc"
    }

    user_agent = "ResearchCollab-Explorer/1.0"
    if email:
        user_agent += f" (mailto:{email})"
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json"
    }

    try:
        response = requests.get(
            OPENALEX_WORKS_ENDPOINT,
            params=params,
            headers=headers,
            timeout=(3.05, timeout)
        )
    except requests.exceptions.Timeout as exc:
        raise OpenAlexAPIError(
            f"Request to OpenAlex timed out after {timeout} seconds. "
            "Check your connection or use an offline sample."
        ) from exc
    except requests.exceptions.RequestException as exc:
        # Request exception strings can contain the full URL, including the search query.
        raise OpenAlexAPIError(
            "Could not connect to OpenAlex. Check your internet connection "
            "or use an offline sample."
        ) from exc

    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        retry_message = f" Retry after {retry_after} seconds." if retry_after else ""
        raise OpenAlexAPIError(
            f"OpenAlex rate limit exceeded.{retry_message} You can use an offline sample."
        )
    if response.status_code >= 500:
        raise OpenAlexAPIError(
            f"OpenAlex server returned HTTP {response.status_code}; try again later."
        )
    if response.status_code != 200:
        raise OpenAlexAPIError(
            f"OpenAlex returned HTTP {response.status_code}: {response.text[:150]}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise OpenAlexAPIError("OpenAlex returned invalid JSON. Please try again later.") from exc

    if not isinstance(payload, dict):
        raise OpenAlexAPIError("OpenAlex returned an unexpected response format.")

    results = payload.get("results")
    if not isinstance(results, list):
        raise OpenAlexAPIError("OpenAlex response is missing a valid results list.")
    if not results:
        raise OpenAlexAPIError(
            f"No research publications found on OpenAlex matching: '{query}'."
        )

    meta = payload.get("meta")
    total_count = meta.get("count", len(results)) if isinstance(meta, dict) else len(results)
    try:
        return parse_openalex_api_results(
            results=results,
            query=query,
            total_count=total_count
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise OpenAlexAPIError(
            "OpenAlex returned publication data in an unexpected format."
        ) from exc


def fetch_openalex_works(
    search_query: str,
    limit: int = 50,
    email: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Query the real OpenAlex API works endpoint.
    Includes polite pool headers, timeouts, rate limit safety, and error handling.
    
    Returns:
        (parsed_data_dict, error_message_if_any)
    """
    query = search_query.strip()
    if not query:
        return None, "Search query cannot be empty."

    if timeout <= 0:
        return None, "Request timeout must be greater than zero seconds."

    # Bound the response size and keep request identity in the polite pool.
    clamped_limit = max(10, min(limit, 100))
    request_email = (
        (email or "").strip()
        or os.getenv("OPENALEX_EMAIL", "").strip()
    )
    try:
        return _fetch_openalex_works_cached(
            query,
            clamped_limit,
            request_email,
            timeout
        ), None
    except OpenAlexAPIError as exc:
        return None, str(exc)


def parse_openalex_api_results(
    results: List[Dict[str, Any]],
    query: str,
    total_count: int
) -> Dict[str, Any]:
    """
    Parse native OpenAlex /works API JSON results into clean relational records.
    """
    publications = []
    authors_catalog: Dict[str, Dict[str, Any]] = {}
    authorships = []

    for item in results:
        w_id = sanitize_work_id(item.get("id"), item.get("title") or "work")
        title = item.get("display_name") or item.get("title") or "Untitled Publication"
        pub_year = item.get("publication_year") or 0
        cited_by = item.get("cited_by_count") or 0
        doi = item.get("doi") or ""
        
        # OpenAlex returns primary_topic as an object; topics is a fallback.
        topic_name = "General Research"
        primary_topic = item.get("primary_topic")
        if isinstance(primary_topic, dict):
            topic_name = primary_topic.get("display_name") or topic_name
        topics = item.get("topics") or []
        if topic_name == "General Research" and topics and isinstance(topics, list):
            first_topic = topics[0]
            if isinstance(first_topic, dict):
                topic_name = first_topic.get("display_name") or topic_name
        if topic_name == "General Research" and item.get("concepts"):
            topic_name = item["concepts"][0].get("display_name", topic_name)

        work_authors = []
        raw_authorships = item.get("authorships") or []

        for position_idx, auth_item in enumerate(raw_authorships):
            if not isinstance(auth_item, dict):
                continue
            author_obj = auth_item.get("author")
            author_obj = author_obj if isinstance(author_obj, dict) else {}
            raw_auth_id = author_obj.get("id")
            raw_auth_name = author_obj.get("display_name")
            if not raw_auth_id and not raw_auth_name:
                continue
            auth_name = raw_auth_name or "Unknown Researcher"
            auth_id = sanitize_author_id(raw_auth_id, auth_name)

            # Extract institutions
            institutions = auth_item.get("institutions") or []
            inst_name = "Unknown Institution"
            inst_country = ""
            if institutions and isinstance(institutions, list) and len(institutions) > 0:
                first_inst = institutions[0]
                if isinstance(first_inst, dict):
                    inst_name = first_inst.get("display_name") or inst_name
                    inst_country = first_inst.get("country_code") or ""

            work_authors.append({
                "id": auth_id,
                "name": auth_name,
                "institution": inst_name,
                "country": inst_country
            })

            # Update author catalog
            if auth_id not in authors_catalog:
                authors_catalog[auth_id] = {
                    "id": auth_id,
                    "name": auth_name,
                    "institution": inst_name,
                    "country": inst_country,
                    "works_count": 0,
                    "total_citations": 0,
                    "publication_ids": []
                }
            authors_catalog[auth_id]["works_count"] += 1
            authors_catalog[auth_id]["total_citations"] += cited_by
            authors_catalog[auth_id]["publication_ids"].append(w_id)

            authorships.append({
                "publication_id": w_id,
                "author_id": auth_id,
                "author_name": auth_name,
                "position": position_idx + 1,
                "institution": inst_name
            })

        publications.append({
            "id": w_id,
            "title": title,
            "publication_year": pub_year,
            "cited_by_count": cited_by,
            "primary_topic": topic_name,
            "doi": doi,
            "authors": work_authors
        })

    return {
        "publications": publications,
        "authors": authors_catalog,
        "authorships": authorships,
        "is_live": True,
        "dataset_name": f"Live OpenAlex Query: '{query}' ({len(publications)} works)",
        "source_note": (
            f"Live OpenAlex API Ingestion "
            f"(Matched ~{total_count:,} total records)"
            if isinstance(total_count, int)
            else "Live OpenAlex API Ingestion"
        )
    }


def parse_raw_publications(
    publications_raw: List[Dict[str, Any]],
    dataset_name: str,
    is_live: bool,
    source_note: str
) -> Dict[str, Any]:
    """
    Standardize raw publication entries (used by offline datasets).
    """
    publications = []
    authors_catalog: Dict[str, Dict[str, Any]] = {}
    authorships = []

    for item in publications_raw:
        w_id = sanitize_work_id(item.get("id"), item.get("title") or "work")
        title = item.get("title") or "Untitled Publication"
        pub_year = item.get("publication_year") or 0
        cited_by = item.get("cited_by_count") or 0
        topic_name = item.get("primary_topic") or "General"
        doi = item.get("doi") or ""

        work_authors = []
        raw_authors = item.get("authors") or []

        for position_idx, auth_item in enumerate(raw_authors):
            auth_name = auth_item.get("name") or "Unknown Researcher"
            raw_auth_id = auth_item.get("id")
            auth_id = sanitize_author_id(raw_auth_id, auth_name)
            inst_name = auth_item.get("institution") or "Unknown Institution"
            inst_country = auth_item.get("country") or ""

            work_authors.append({
                "id": auth_id,
                "name": auth_name,
                "institution": inst_name,
                "country": inst_country
            })

            if auth_id not in authors_catalog:
                authors_catalog[auth_id] = {
                    "id": auth_id,
                    "name": auth_name,
                    "institution": inst_name,
                    "country": inst_country,
                    "works_count": 0,
                    "total_citations": 0,
                    "publication_ids": []
                }
            authors_catalog[auth_id]["works_count"] += 1
            authors_catalog[auth_id]["total_citations"] += cited_by
            authors_catalog[auth_id]["publication_ids"].append(w_id)

            authorships.append({
                "publication_id": w_id,
                "author_id": auth_id,
                "author_name": auth_name,
                "position": position_idx + 1,
                "institution": inst_name
            })

        publications.append({
            "id": w_id,
            "title": title,
            "publication_year": pub_year,
            "cited_by_count": cited_by,
            "primary_topic": topic_name,
            "doi": doi,
            "authors": work_authors
        })

    return {
        "publications": publications,
        "authors": authors_catalog,
        "authorships": authorships,
        "is_live": is_live,
        "dataset_name": dataset_name,
        "source_note": source_note
    }
