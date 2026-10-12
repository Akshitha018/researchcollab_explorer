"""
Unit tests for network construction, edge weights, graph metrics,
community detection, and filtering pipelines.
"""

import pytest
import networkx as nx
import pandas as pd
from modules.data_loader import load_offline_dataset, sanitize_author_id
from modules.network_builder import (
    build_coauthorship_network,
    filter_network,
    get_edge_dataframe,
    get_node_dataframe
)
from modules.graph_metrics import (
    compute_network_summary_metrics,
    compute_node_centrality_metrics
)
from modules.community_detection import detect_communities, get_community_summary


@pytest.fixture
def mock_dataset():
    """
    Create a controlled synthetic research corpus for exact metric assertion.
    """
    publications = [
        {
            "id": "W1",
            "title": "Foundation of Graph Learning",
            "publication_year": 2020,
            "cited_by_count": 100,
            "primary_topic": "Graph Theory",
            "authors": [
                {"id": "A1", "name": "Alice Researcher", "institution": "Univ A"},
                {"id": "A2", "name": "Bob Scientist", "institution": "Univ B"},
                {"id": "A3", "name": "Charlie Engineer", "institution": "Univ C"}
            ]
        },
        {
            "id": "W2",
            "title": "Advanced Graph Attention",
            "publication_year": 2021,
            "cited_by_count": 50,
            "primary_topic": "Graph Theory",
            "authors": [
                {"id": "A1", "name": "Alice Researcher", "institution": "Univ A"},
                {"id": "A2", "name": "Bob Scientist", "institution": "Univ B"}
            ]
        },
        {
            "id": "W3",
            "title": "Solo Survey on Deep Learning",
            "publication_year": 2022,
            "cited_by_count": 20,
            "primary_topic": "Deep Learning",
            "authors": [
                {"id": "A4", "name": "David Solo", "institution": "Univ D"}
            ]
        }
    ]

    authors_catalog = {
        "A1": {"name": "Alice Researcher", "institution": "Univ A", "works_count": 2, "total_citations": 150, "publication_ids": ["W1", "W2"]},
        "A2": {"name": "Bob Scientist", "institution": "Univ B", "works_count": 2, "total_citations": 150, "publication_ids": ["W1", "W2"]},
        "A3": {"name": "Charlie Engineer", "institution": "Univ C", "works_count": 1, "total_citations": 100, "publication_ids": ["W1"]},
        "A4": {"name": "David Solo", "institution": "Univ D", "works_count": 1, "total_citations": 20, "publication_ids": ["W3"]}
    }

    return publications, authors_catalog


def test_sanitize_author_id():
    assert sanitize_author_id("https://openalex.org/A12345", "Alice") == "A12345"
    assert sanitize_author_id(None, "Bob Builder") == "author_bob_builder"
    assert sanitize_author_id("", "") == "author_unknown"


def test_graph_construction_no_self_loops(mock_dataset):
    pubs, authors = mock_dataset
    G = build_coauthorship_network(pubs, authors)

    # Check node count
    assert G.number_of_nodes() == 4
    assert G.has_node("A1")
    assert G.has_node("A2")
    assert G.has_node("A3")
    assert G.has_node("A4")

    # Strict check: No self loops
    assert list(nx.selfloop_edges(G)) == []

    # Undirected simple graph check
    assert not G.is_directed()
    assert not G.is_multigraph()


def test_edge_weights_and_shared_publications(mock_dataset):
    pubs, authors = mock_dataset
    G = build_coauthorship_network(pubs, authors)

    # A1 and A2 co-authored W1 and W2 -> weight must be exactly 2
    assert G.has_edge("A1", "A2")
    assert G["A1"]["A2"]["weight"] == 2
    assert len(G["A1"]["A2"]["shared_publications"]) == 2

    # A1 and A3 co-authored only W1 -> weight must be 1
    assert G.has_edge("A1", "A3")
    assert G["A1"]["A3"]["weight"] == 1

    # A2 and A3 co-authored only W1 -> weight must be 1
    assert G.has_edge("A2", "A3")
    assert G["A2"]["A3"]["weight"] == 1

    # A4 is a solo author -> 0 edges
    assert G.degree("A4") == 0


def test_network_filtering_by_weight(mock_dataset):
    pubs, authors = mock_dataset
    G = build_coauthorship_network(pubs, authors)

    # Filter with min_weight = 2 (only A1-A2 should remain)
    filtered = filter_network(G, min_weight=2)
    assert filtered.number_of_edges() == 1
    assert filtered.has_edge("A1", "A2")
    assert not filtered.has_edge("A1", "A3")


def test_network_filtering_by_year(mock_dataset):
    pubs, authors = mock_dataset
    G = build_coauthorship_network(pubs, authors)

    # Filter for year 2021 only (W2 was in 2021 with A1 and A2)
    filtered = filter_network(G, min_weight=1, year_range=(2021, 2021))
    assert filtered.number_of_edges() == 1
    assert filtered.has_edge("A1", "A2")
    # Edge A1-A2 in 2021 should have effective weight 1
    assert filtered["A1"]["A2"]["weight"] == 1


def test_year_filter_excludes_publications_with_unknown_year():
    publications = [{
        "id": "W1",
        "title": "Unknown year collaboration",
        "publication_year": 0,
        "authors": [
            {"id": "A1", "name": "Alice"},
            {"id": "A2", "name": "Bob"}
        ]
    }]
    graph = build_coauthorship_network(publications, {})

    filtered = filter_network(graph, year_range=(2020, 2025))
    assert filtered.number_of_edges() == 0


def test_network_filtering_ego_subgraph(mock_dataset):
    pubs, authors = mock_dataset
    G = build_coauthorship_network(pubs, authors)

    # Ego network around A3 with radius 1
    ego_g = filter_network(G, target_researcher_id="A3", ego_radius=1)
    assert "A3" in ego_g
    assert "A1" in ego_g
    assert "A2" in ego_g
    assert "A4" not in ego_g


def test_graph_summary_metrics(mock_dataset):
    pubs, authors = mock_dataset
    G = build_coauthorship_network(pubs, authors)

    metrics = compute_network_summary_metrics(G)
    assert metrics["num_nodes"] == 4
    assert metrics["num_edges"] == 3
    assert metrics["total_collaborations"] == 4  # weight(A1,A2)=2 + weight(A1,A3)=1 + weight(A2,A3)=1
    assert metrics["num_components"] == 2  # {A1, A2, A3} and {A4}
    assert metrics["largest_component_size"] == 3
    assert metrics["density"] > 0
    assert metrics["avg_clustering_coefficient"] > 0


def test_node_centrality_metrics(mock_dataset):
    pubs, authors = mock_dataset
    G = build_coauthorship_network(pubs, authors)

    node_df = compute_node_centrality_metrics(G)
    assert not node_df.empty
    assert len(node_df) == 4
    assert "pagerank" in node_df.columns
    assert "degree_centrality" in node_df.columns
    assert "betweenness_centrality" in node_df.columns

    # Check degree and weighted degree comparisons
    row_a1 = node_df[node_df["author_id"] == "A1"].iloc[0]
    row_a4 = node_df[node_df["author_id"] == "A4"].iloc[0]
    assert row_a1["degree"] > row_a4["degree"]
    assert row_a1["weighted_degree"] > row_a4["weighted_degree"]
    assert row_a1["degree_centrality"] > row_a4["degree_centrality"]
    assert row_a4["degree"] == 0


def test_community_detection_and_fallbacks(mock_dataset):
    pubs, authors = mock_dataset
    G = build_coauthorship_network(pubs, authors)

    node_to_comm, comm_sets, algo_name, mod_score = detect_communities(G)
    assert len(node_to_comm) == 4
    assert len(comm_sets) >= 1
    assert isinstance(algo_name, str)
    assert isinstance(mod_score, float)

    # Verify every node in G is assigned a community
    for node in G.nodes():
        assert node in node_to_comm


def test_offline_datasets_load_cleanly():
    for dataset_key in ["gnn", "quantum", "crispr"]:
        data = load_offline_dataset(dataset_key)
        assert data["is_live"] is False
        assert "Not Live API" in data["source_note"]
        assert "publications" in data
        assert "authors" in data
        assert len(data["publications"]) > 5
        assert len(data["authors"]) > 5

        G = build_coauthorship_network(data["publications"], data["authors"])
        assert G.number_of_nodes() > 5
        assert G.number_of_edges() > 5
        # Ensure no self loops in any benchmark dataset
        assert list(nx.selfloop_edges(G)) == []


def test_export_utilities(mock_dataset):
    from modules.export_utils import (
        generate_publications_dataframe,
        generate_authorships_dataframe,
        generate_graphml_export,
        generate_summary_metrics_dataframe,
        dataframe_to_csv_bytes
    )
    pubs, authors = mock_dataset
    G = build_coauthorship_network(pubs, authors)

    # DataFrame conversions
    pubs_df = generate_publications_dataframe(pubs)
    assert len(pubs_df) == 3
    assert "citations" in pubs_df.columns
    assert "authors_list" in pubs_df.columns

    # Authorships
    dummy_authorships = [{"publication_id": "W1", "author_id": "A1", "author_name": "Alice", "position": 1, "institution": "Univ A"}]
    auth_df = generate_authorships_dataframe(dummy_authorships)
    assert len(auth_df) == 1

    # Summary metrics
    metrics = compute_network_summary_metrics(G)
    summary_df = generate_summary_metrics_dataframe(metrics)
    assert not summary_df.empty
    assert "Metric" in summary_df.columns

    # CSV bytes export
    csv_bytes = dataframe_to_csv_bytes(pubs_df)
    assert isinstance(csv_bytes, bytes)
    assert len(csv_bytes) > 0

    # GraphML export
    graphml_xml = generate_graphml_export(G)
    assert "<graphml" in graphml_xml or "<?xml" in graphml_xml


def test_community_summaries_and_colors(mock_dataset):
    from modules.community_detection import get_community_color
    pubs, authors = mock_dataset
    G = build_coauthorship_network(pubs, authors)

    node_to_comm, comm_sets, _, _ = detect_communities(G)
    summaries = get_community_summary(G, comm_sets, node_to_comm)
    assert len(summaries) == len(comm_sets)
    assert "community_id" in summaries[0]
    assert "color" in summaries[0]
    assert summaries[0]["color"].startswith("#")

    # Color index rollover test
    assert get_community_color(0).startswith("#")
    assert get_community_color(25).startswith("#")


def test_visualization_ego_and_layout(mock_dataset):
    from modules.visualization import (
        compute_graph_layout,
        create_ego_figure,
        create_network_figure,
        create_top_collaborators_chart
    )
    pubs, authors = mock_dataset
    G = build_coauthorship_network(pubs, authors)
    node_to_comm, _, _, _ = detect_communities(G)
    node_df = compute_node_centrality_metrics(G, node_to_comm)

    # Layout calculation
    pos_spring = compute_graph_layout(G, layout_type="spring")
    pos_circular = compute_graph_layout(G, layout_type="circular")
    assert len(pos_spring) == 4
    assert len(pos_circular) == 4

    # Ego figure for A1
    fig_ego = create_ego_figure(G, "A1", node_df)
    assert fig_ego is not None
    assert len(fig_ego.data) > 0

    fig_network = create_network_figure(G, node_df, layout_type="circular")
    weighted_edge_traces = [
        trace for trace in fig_network.data
        if "shared publication" in trace.name
    ]
    weight_one = next(trace for trace in weighted_edge_traces if trace.name.startswith("1 "))
    weight_two = next(trace for trace in weighted_edge_traces if trace.name.startswith("2 "))
    assert weight_two.line.width > weight_one.line.width
    assert any("Shared publications: 2" in text for text in weight_two.text)
    assert weight_two.hoverinfo == "text"

    # Top collaborators chart
    fig_top = create_top_collaborators_chart(node_df, metric_col="degree", top_n=3)
    assert fig_top is not None
    assert len(fig_top.data) > 0


def test_openalex_live_parser_edge_cases():
    from modules.data_loader import parse_openalex_api_results
    raw_results = [
        {
            "id": "https://openalex.org/W99999",
            "title": "Edge Case Paper",
            "publication_year": 2023,
            "cited_by_count": 5,
            "authorships": [
                {
                    "author": {"id": "https://openalex.org/A991", "display_name": "Test Author 1"},
                    "institutions": [{"display_name": "Test Lab", "country_code": "US"}]
                },
                {
                    "author": {"id": None, "display_name": "Anonymous Author"},
                    "institutions": []
                }
            ]
        }
    ]

    parsed = parse_openalex_api_results(raw_results, "Edge Case", 1)
    assert len(parsed["publications"]) == 1
    assert len(parsed["authors"]) == 2
    assert "A991" in parsed["authors"]
    assert "author_anonymous_author" in parsed["authors"]
    assert len(parsed["authorships"]) == 2


def test_duplicate_author_entries_do_not_create_self_loops():
    publications = [{
        "id": "W1",
        "title": "Repeated author record",
        "publication_year": 2024,
        "authors": [
            {"id": "A1", "name": "Alice"},
            {"id": "A1", "name": "Alice"},
            {"id": "A2", "name": "Bob"}
        ]
    }]
    graph = build_coauthorship_network(publications, {})

    assert list(nx.selfloop_edges(graph)) == []
    assert graph["A1"]["A2"]["weight"] == 1
    assert graph.number_of_edges() == 1
