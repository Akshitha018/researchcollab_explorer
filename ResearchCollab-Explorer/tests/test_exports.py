"""Tests for src/exports.py and the PyVis HTML builder."""

import io
import json

import pandas as pd

from src import exports as ex
from src import graph_analysis as ga
from src import visualization as viz


def make_graph():
    edges = pd.DataFrame(
        [(1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 6), (3, 4)], columns=["source", "target"]
    )
    g = ga.build_graph(edges)
    comm = ga.detect_communities(g)
    return edges, g, comm, ga.compute_node_metrics(g, comm)


def test_edges_csv_round_trip():
    edges, *_ = make_graph()
    back = pd.read_csv(io.BytesIO(ex.edges_to_csv(edges)))
    assert list(back.columns) == ["source", "target"]
    assert len(back) == len(edges)


def test_graph_edges_dataframe_matches_graph():
    _, g, *_ = make_graph()
    df = ex.graph_edges_dataframe(g)
    assert len(df) == g.number_of_edges()
    assert (df["source"] < df["target"]).all()
    assert ex.graph_edges_dataframe(ga.build_graph(None)).empty


def test_metrics_csv_has_all_columns():
    *_, metrics = make_graph()
    back = pd.read_csv(io.BytesIO(ex.metrics_to_csv(metrics)))
    assert list(back.columns) == ga.METRIC_COLUMNS
    assert len(back) == 6


def test_communities_csv_and_restriction():
    _, _, comm, _ = make_graph()
    full = pd.read_csv(io.BytesIO(ex.communities_to_csv(comm)))
    assert len(full) == 6
    part = pd.read_csv(io.BytesIO(ex.communities_to_csv(comm, restrict_to={1, 2})))
    assert set(part["researcher_id"]) == {1, 2}


def test_report_json_and_csv():
    _, g, comm, _ = make_graph()
    summary = ga.network_summary(g, comm)
    report = ex.build_stats_report(summary, {"unique_edges": 7}, {"seed": 42},
                                   "Whole network", {}, "sample")
    parsed = json.loads(ex.report_to_json(report))
    assert parsed["network"]["nodes"] == 6
    assert parsed["scope"] == "Whole network"
    flat = pd.read_csv(io.BytesIO(ex.report_to_csv(report)))
    assert list(flat.columns) == ["section", "key", "value"]
    assert ((flat["section"] == "network") & (flat["key"] == "nodes")).any()


# --------------------------------------------------------------------------------------
# PyVis HTML
# --------------------------------------------------------------------------------------
def test_network_html_is_self_contained_and_contains_nodes():
    _, g, comm, metrics = make_graph()
    html = viz.build_network_html(g, comm, metrics, selected=3, height_px=500)
    assert "<html" in html.lower()
    assert "vis" in html and "Researcher ID: 3" in html
    assert "cdnjs" not in html and "unpkg" not in html  # JS is inlined: works offline


def test_network_html_without_selection_and_for_empty_graph():
    _, g, comm, metrics = make_graph()
    assert "Researcher ID" in viz.build_network_html(g, comm, metrics)
    empty = viz.build_network_html(ga.build_graph(None), {}, metrics.iloc[0:0])
    assert "<html" in empty.lower()


def test_scale_sizes_handles_equal_and_empty():
    assert viz.scale_sizes({}) == {}
    sizes = viz.scale_sizes({1: 3, 2: 3})
    assert sizes[1] == sizes[2]
    sizes = viz.scale_sizes({1: 1, 2: 10})
    assert sizes[1] < sizes[2]


def test_charts_build_for_data_and_for_empty_input():
    _, g, comm, metrics = make_graph()
    dist = ga.degree_distribution(g)
    summary = ga.community_summary(comm)
    for fig in (
        viz.degree_distribution_chart(dist, True),
        viz.degree_distribution_chart(dist, False),
        viz.top_metric_chart(metrics, "degree", 3),
        viz.community_size_chart(summary),
        viz.community_size_histogram(summary),
        viz.degree_vs_betweenness_chart(metrics),
    ):
        assert len(fig.data) >= 1
    empty = metrics.iloc[0:0]
    assert len(viz.top_metric_chart(empty).data) == 0
    assert len(viz.degree_vs_betweenness_chart(empty).data) == 0
    assert len(viz.degree_distribution_chart(dist.iloc[0:0]).data) == 0
    assert len(viz.community_size_chart(summary.iloc[0:0]).data) == 0
