import os

import pytest

from modules.community_detection import detect_communities
from modules.data_loader import _fetch_openalex_works_cached, fetch_openalex_works
from modules.graph_metrics import (
    compute_network_summary_metrics,
    compute_node_centrality_metrics,
)
from modules.network_builder import build_coauthorship_network
from modules.visualization import create_network_figure


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_OPENALEX_INTEGRATION") != "1",
    reason="Set RUN_OPENALEX_INTEGRATION=1 to make one real OpenAlex request.",
)


def test_live_openalex_works_flow_into_graph_metrics_and_visualization():
    _fetch_openalex_works_cached.clear()
    data, error = fetch_openalex_works(
        "graph neural networks",
        limit=10,
        timeout=8,
    )

    assert error is None
    assert data is not None
    assert data["is_live"] is True
    assert 1 <= len(data["publications"]) <= 10

    graph = build_coauthorship_network(data["publications"], data["authors"])
    assert graph.number_of_nodes() > 0

    community_map, communities, _, _ = detect_communities(graph)
    metrics = compute_node_centrality_metrics(graph, community_map)
    summary = compute_network_summary_metrics(graph)
    figure = create_network_figure(graph, metrics)

    assert len(metrics) == graph.number_of_nodes()
    assert summary["num_nodes"] == graph.number_of_nodes()
    assert len(community_map) == graph.number_of_nodes()
    assert sum(len(group) for group in communities) == graph.number_of_nodes()
    assert len(figure.data) > 0
    _fetch_openalex_works_cached.clear()
