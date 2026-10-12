"""Tests for src/graph_analysis.py (construction, metrics, communities, filtering)."""

import math

import networkx as nx
import pandas as pd
import pytest

from src import data_loader as dl
from src import graph_analysis as ga


def edges_df(pairs):
    return pd.DataFrame(pairs, columns=["source", "target"], dtype="int64")


# Two triangles (1-2-3 and 4-5-6) joined by the single bridge 3-4.
TWO_TRIANGLES = [(1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 6), (3, 4)]
STAR = [(0, 1), (0, 2), (0, 3), (0, 4)]  # centre 0, four leaves


# --------------------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------------------
def test_build_graph_counts():
    g = ga.build_graph(edges_df(TWO_TRIANGLES))
    assert g.number_of_nodes() == 6
    assert g.number_of_edges() == 7
    assert not isinstance(g, nx.DiGraph)


def test_build_graph_ignores_self_loops_and_duplicates():
    g = ga.build_graph(edges_df([(1, 2), (2, 1), (1, 2), (3, 3), (2, 3)]))
    assert g.number_of_edges() == 2
    assert nx.number_of_selfloops(g) == 0


def test_build_graph_from_none_or_empty():
    assert ga.build_graph(None).number_of_nodes() == 0
    assert ga.build_graph(edges_df([])).number_of_nodes() == 0


# --------------------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------------------
def test_summary_of_triangle():
    s = ga.network_summary(ga.build_graph(edges_df([(1, 2), (2, 3), (1, 3)])))
    assert s["nodes"] == 3 and s["edges"] == 3
    assert s["average_degree"] == pytest.approx(2.0)  # 2*3/3
    assert s["density"] == pytest.approx(1.0)  # complete graph
    assert s["average_clustering"] == pytest.approx(1.0)
    assert s["connected_components"] == 1
    assert s["isolated_nodes"] == 0


def test_summary_formulas_on_star():
    s = ga.network_summary(ga.build_graph(edges_df(STAR)))
    assert s["average_degree"] == pytest.approx(2 * 4 / 5)
    assert s["density"] == pytest.approx(4 / (5 * 4 / 2))
    assert s["average_clustering"] == pytest.approx(0.0)
    assert s["max_degree"] == 4


def test_summary_disconnected_graph():
    g = ga.build_graph(edges_df([(1, 2), (3, 4), (4, 5)]))
    s = ga.network_summary(g)
    assert s["connected_components"] == 2
    assert s["largest_component_size"] == 3
    assert s["largest_component_fraction"] == pytest.approx(3 / 5)


def test_summary_empty_graph_does_not_crash():
    s = ga.network_summary(nx.Graph(), {})
    assert s["nodes"] == 0 and s["edges"] == 0
    assert s["average_degree"] == 0.0 and s["density"] == 0.0
    assert s["connected_components"] == 0
    assert s["modularity"] is None


def test_summary_single_edge():
    s = ga.network_summary(ga.build_graph(edges_df([(7, 8)])))
    assert s["nodes"] == 2 and s["edges"] == 1
    assert s["connected_components"] == 1
    assert s["density"] == pytest.approx(1.0)


def test_degree_distribution():
    g = ga.build_graph(edges_df(STAR))
    dist = ga.degree_distribution(g)
    assert dist.set_index("degree")["count"].to_dict() == {1: 4, 4: 1}
    assert dist["fraction"].sum() == pytest.approx(1.0)
    assert ga.degree_distribution(nx.Graph()).empty


# --------------------------------------------------------------------------------------
# Communities
# --------------------------------------------------------------------------------------
def test_louvain_finds_the_two_triangles():
    g = ga.build_graph(edges_df(TWO_TRIANGLES))
    comm = ga.detect_communities(g, seed=42)
    assert set(comm) == set(g.nodes())
    assert comm[1] == comm[2] == comm[3]
    assert comm[4] == comm[5] == comm[6]
    assert comm[1] != comm[4]
    assert set(comm.values()) == {0, 1}


def test_communities_are_deterministic_for_same_seed():
    g = ga.build_graph(edges_df(TWO_TRIANGLES))
    assert ga.detect_communities(g, seed=7) == ga.detect_communities(g, seed=7)


def test_community_ids_are_ordered_by_size():
    # A 4-clique {1,2,3,4} and a separate triangle {10,11,12}
    pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (10, 11), (10, 12), (11, 12)]
    comm = ga.detect_communities(ga.build_graph(edges_df(pairs)), seed=1)
    assert comm[1] == 0  # bigger group gets ID 0
    assert comm[10] == 1


def test_relabel_is_deterministic():
    assert ga.relabel_communities({5: "b", 6: "b", 1: "a"}) == {5: 0, 6: 0, 1: 1}


def test_communities_edge_cases():
    assert ga.detect_communities(nx.Graph()) == {}
    lonely = nx.Graph()
    lonely.add_nodes_from([3, 1, 2])
    assert ga.detect_communities(lonely) == {1: 0, 2: 1, 3: 2}  # no edges -> singletons


def test_community_summary_adds_up():
    comm = ga.detect_communities(ga.build_graph(edges_df(TWO_TRIANGLES)), seed=42)
    summary = ga.community_summary(comm)
    assert summary["size"].sum() == 6
    assert summary["share"].sum() == pytest.approx(1.0)
    assert ga.community_summary({}).empty


def test_modularity_is_reported_and_positive_for_clear_structure():
    g = ga.build_graph(edges_df(TWO_TRIANGLES))
    comm = ga.detect_communities(g, seed=42)
    s = ga.network_summary(g, comm)
    assert s["communities"] == 2
    assert s["modularity"] > 0.3


# --------------------------------------------------------------------------------------
# Node metrics
# --------------------------------------------------------------------------------------
def test_metrics_on_star_graph():
    g = ga.build_graph(edges_df(STAR))
    comm = ga.detect_communities(g)
    m = ga.compute_node_metrics(g, comm).set_index("researcher_id")
    assert list(ga.compute_node_metrics(g, comm).columns) == ga.METRIC_COLUMNS
    assert m.loc[0, "degree"] == 4
    assert m.loc[0, "degree_centrality"] == pytest.approx(1.0)  # 4 / (5-1)
    assert m.loc[1, "degree_centrality"] == pytest.approx(0.25)
    assert m.loc[0, "betweenness_centrality"] == pytest.approx(1.0)  # every path crosses it
    assert m.loc[1, "betweenness_centrality"] == pytest.approx(0.0)
    assert m.loc[0, "closeness_centrality"] == pytest.approx(1.0)
    assert m.loc[1, "closeness_centrality"] == pytest.approx(4 / 7)  # 4 / (1+2+2+2)
    assert (m["clustering_coefficient"] == 0).all()


def test_metrics_on_triangle_clustering_is_one():
    g = ga.build_graph(edges_df([(1, 2), (2, 3), (1, 3)]))
    m = ga.compute_node_metrics(g, ga.detect_communities(g))
    assert m["clustering_coefficient"].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_bridge_nodes_have_highest_betweenness():
    g = ga.build_graph(edges_df(TWO_TRIANGLES))
    m = ga.compute_node_metrics(g, ga.detect_communities(g)).set_index("researcher_id")
    top_two = set(m["betweenness_centrality"].nlargest(2).index)
    assert top_two == {3, 4}


def test_metrics_match_networkx_directly():
    g = ga.build_graph(edges_df(TWO_TRIANGLES))
    m = ga.compute_node_metrics(g, ga.detect_communities(g)).set_index("researcher_id")
    expected = nx.betweenness_centrality(g, normalized=True)
    for node, value in expected.items():
        assert m.loc[node, "betweenness_centrality"] == pytest.approx(value)


def test_metrics_for_disconnected_graph_are_finite():
    g = ga.build_graph(edges_df([(1, 2), (3, 4), (4, 5)]))
    m = ga.compute_node_metrics(g, ga.detect_communities(g))
    numeric = m[["degree_centrality", "betweenness_centrality",
                 "closeness_centrality", "clustering_coefficient"]]
    assert numeric.notna().all().all()
    assert all(math.isfinite(x) for x in numeric.to_numpy().ravel())


def test_sampled_metrics_are_reproducible_and_valid():
    g = nx.gnm_random_graph(60, 150, seed=3)
    comm = ga.detect_communities(g)
    a = ga.compute_node_metrics(g, comm, sample_size=20, seed=5)
    b = ga.compute_node_metrics(g, comm, sample_size=20, seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert (a["betweenness_centrality"] >= 0).all()
    assert a["closeness_centrality"].between(0, 1).all()


def test_sample_size_larger_than_graph_or_zero_falls_back_to_exact():
    g = ga.build_graph(edges_df(STAR))
    comm = ga.detect_communities(g)
    exact = ga.compute_node_metrics(g, comm, sample_size=None)
    pd.testing.assert_frame_equal(exact, ga.compute_node_metrics(g, comm, sample_size=999))
    pd.testing.assert_frame_equal(exact, ga.compute_node_metrics(g, comm, sample_size=0))


def test_default_sample_size_policy():
    assert ga.default_sample_size(500) is None
    assert ga.default_sample_size(ga.EXACT_METRICS_MAX_NODES) is None
    assert ga.default_sample_size(5000) == ga.DEFAULT_SAMPLE_SIZE


def test_closeness_from_all_nodes_as_pivots_equals_exact():
    # Using EVERY node as a pivot, the estimator must reproduce exact closeness, also for a
    # disconnected graph (Wasserman-Faust scaling).
    g = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5), (10, 11), (11, 12)])
    g.add_edge(20, 21)
    estimate = ga._closeness_from_pivots(g, list(g.nodes()))
    exact = nx.closeness_centrality(g)
    for node, value in exact.items():
        assert estimate[node] == pytest.approx(value)


def test_closeness_handles_component_without_any_pivot():
    # Pivots only in the big path; the small component {100, 101} is never sampled, so its
    # nodes must fall back to an exact computation and still match networkx.
    g = nx.path_graph(30)
    g.add_edge(100, 101)
    estimate = ga._closeness_from_pivots(g, [0, 5, 10, 15, 20, 25])
    exact = nx.closeness_centrality(g)
    assert estimate[100] == pytest.approx(exact[100])
    assert estimate[101] == pytest.approx(exact[101])


def test_closeness_isolated_node_is_zero():
    g = nx.Graph([(1, 2)])
    g.add_node(9)
    assert ga._closeness_from_pivots(g, [1])[9] == 0.0


def test_approximate_closeness_is_close_to_exact_on_a_random_graph():
    g = nx.connected_watts_strogatz_graph(300, 6, 0.2, seed=2)
    approx = ga.approximate_closeness(g, 150, seed=1)
    exact = nx.closeness_centrality(g)
    mean_abs_error = sum(abs(approx[v] - exact[v]) for v in g) / len(g)
    assert mean_abs_error < 0.02  # closeness values here are around 0.25-0.30


def test_closeness_can_be_skipped():
    g = ga.build_graph(edges_df(STAR))
    m = ga.compute_node_metrics(g, ga.detect_communities(g), compute_closeness=False)
    assert m["closeness_centrality"].isna().all()


def test_empty_graph_metrics():
    m = ga.compute_node_metrics(nx.Graph(), {})
    assert m.empty and list(m.columns) == ga.METRIC_COLUMNS


def test_neighbors_table():
    g = ga.build_graph(edges_df(TWO_TRIANGLES))
    m = ga.compute_node_metrics(g, ga.detect_communities(g))
    table = ga.neighbors_table(g, m, 3)
    assert set(table["researcher_id"]) == {1, 2, 4}
    assert table.iloc[0]["researcher_id"] == 4  # highest degree first
    assert ga.neighbors_table(g, m, 999).empty


# --------------------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------------------
@pytest.fixture
def sample_graph():
    edges, _ = dl.load_edges_from_path(dl.SAMPLE_PATH)
    g = ga.build_graph(edges)
    return g, ga.detect_communities(g, seed=42)


def test_sample_graph_basic_facts(sample_graph):
    g, comm = sample_graph
    s = ga.network_summary(g, comm)
    assert s["nodes"] == 30 and s["edges"] == 55
    assert s["connected_components"] == 2  # the main network + the 31..35 group


def test_filter_min_degree(sample_graph):
    g, comm = sample_graph
    view = ga.filter_graph(g, comm, min_degree=5)
    assert view.number_of_nodes() > 0
    assert all(g.degree(n) >= 5 for n in view.nodes())


def test_filter_top_n_keeps_highest_degrees(sample_graph):
    g, comm = sample_graph
    view = ga.filter_graph(g, comm, top_n=5)
    assert view.number_of_nodes() == 5
    cutoff = sorted((d for _, d in g.degree()), reverse=True)[4]
    assert min(g.degree(n) for n in view.nodes()) >= cutoff


def test_filter_by_community(sample_graph):
    g, comm = sample_graph
    view = ga.filter_graph(g, comm, community_id=0)
    assert set(view.nodes()) == {n for n, c in comm.items() if c == 0}


def test_filter_neighbourhood_and_selected(sample_graph):
    g, comm = sample_graph
    view = ga.filter_graph(g, comm, selected=4, neighborhood_only=True)
    assert set(view.nodes()) == {4} | set(g.neighbors(4))


def test_filter_always_keeps_selected_even_with_top_n(sample_graph):
    g, comm = sample_graph
    low_degree_node = min(g.nodes(), key=lambda n: (g.degree(n), n))
    view = ga.filter_graph(g, comm, top_n=3, selected=low_degree_node)
    assert low_degree_node in view
    assert view.number_of_nodes() == 3


def test_filter_does_not_modify_original(sample_graph):
    g, comm = sample_graph
    before = g.number_of_nodes()
    ga.filter_graph(g, comm, top_n=3)
    assert g.number_of_nodes() == before


def test_filter_on_empty_graph():
    assert ga.filter_graph(nx.Graph(), {}, min_degree=2, top_n=5).number_of_nodes() == 0
