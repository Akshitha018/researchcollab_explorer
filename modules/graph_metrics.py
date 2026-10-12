"""
Graph metrics computation module for ResearchCollab Explorer.
Calculates macro network statistics (density, components, diameter, clustering)
and micro node centralities (degree, betweenness, PageRank, closeness).
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_network_summary_metrics(G: nx.Graph) -> Dict[str, Any]:
    """
    Calculate high-level structural metrics for the co-authorship network.
    Includes density, clustering coefficient, connected component distribution,
    average shortest path length, and network diameter.
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "density": 0.0,
            "avg_degree": 0.0,
            "num_components": 0,
            "largest_component_size": 0,
            "largest_component_ratio": 0.0,
            "avg_clustering_coefficient": 0.0,
            "transitivity": 0.0,
            "avg_shortest_path_length": 0.0,
            "diameter": 0,
            "is_connected": False,
            "total_collaborations": 0
        }

    # Density: 2 * |E| / (|V| * (|V| - 1))
    density = nx.density(G)
    avg_degree = (2.0 * n_edges) / n_nodes if n_nodes > 0 else 0.0
    total_collabs = sum([d.get("weight", 1) for _, _, d in G.edges(data=True)])

    # Connected Components
    components = list(nx.connected_components(G))
    num_components = len(components)
    largest_comp = max(components, key=len) if components else set()
    largest_comp_size = len(largest_comp)
    largest_comp_ratio = (largest_comp_size / n_nodes) * 100.0 if n_nodes > 0 else 0.0

    # Clustering Coefficient & Transitivity
    try:
        avg_clustering = nx.average_clustering(G)
    except Exception:
        avg_clustering = 0.0

    try:
        transitivity = nx.transitivity(G)
    except Exception:
        transitivity = 0.0

    # Metrics on the Largest Connected Component (LCC)
    avg_path_length = 0.0
    diameter = 0
    is_connected = nx.is_connected(G)

    if largest_comp_size > 1:
        lcc_subgraph = G.subgraph(largest_comp)
        try:
            avg_path_length = round(nx.average_shortest_path_length(lcc_subgraph), 3)
            diameter = nx.diameter(lcc_subgraph)
        except Exception as e:
            logger.debug("Failed computing diameter or path length: %s", e)
            avg_path_length = 0.0
            diameter = 0

    return {
        "num_nodes": n_nodes,
        "num_edges": n_edges,
        "density": round(density, 4),
        "avg_degree": round(avg_degree, 2),
        "total_collaborations": total_collabs,
        "num_components": num_components,
        "largest_component_size": largest_comp_size,
        "largest_component_ratio": round(largest_comp_ratio, 1),
        "avg_clustering_coefficient": round(avg_clustering, 4),
        "transitivity": round(transitivity, 4),
        "avg_shortest_path_length": avg_path_length,
        "diameter": diameter,
        "is_connected": is_connected
    }


def compute_node_centrality_metrics(
    G: nx.Graph,
    community_map: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """
    Calculate node-level graph centrality measures:
    - Degree and Degree Centrality
    - Betweenness Centrality (brokerage role)
    - PageRank (weighted authority/prestige)
    - Closeness Centrality (reachability)
    - Local Clustering Coefficient
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=[
            "author_id", "name", "institution", "degree", "weighted_degree",
            "degree_centrality", "betweenness_centrality", "pagerank",
            "closeness_centrality", "clustering_coefficient", "community_id"
        ])

    # 1. Degree Centrality
    deg_centrality = nx.degree_centrality(G)

    # 2. Betweenness Centrality
    # For large graphs, betweenness can be computed efficiently or sampled
    try:
        bet_centrality = nx.betweenness_centrality(G, normalized=True)
    except Exception:
        bet_centrality = {node: 0.0 for node in G.nodes()}

    # 3. PageRank (using edge weight for collaboration intensity)
    try:
        pr_centrality = nx.pagerank(G, weight="weight", alpha=0.85, max_iter=200)
    except Exception:
        # Fallback if convergence fails
        pr_centrality = {node: 1.0 / len(G) for node in G.nodes()}

    # 4. Closeness Centrality
    try:
        close_centrality = nx.closeness_centrality(G)
    except Exception:
        close_centrality = {node: 0.0 for node in G.nodes()}

    # 5. Local Clustering Coefficient
    try:
        local_clustering = nx.clustering(G)
    except Exception:
        local_clustering = {node: 0.0 for node in G.nodes()}

    rows = []
    for node, data in G.nodes(data=True):
        raw_deg = G.degree(node)
        weighted_deg = sum([
            edata.get("weight", 1) for _, _, edata in G.edges(node, data=True)
        ])
        comm_id = community_map.get(node, 0) if community_map else 0

        rows.append({
            "author_id": node,
            "name": data.get("name", node),
            "institution": data.get("institution", "Unknown Institution"),
            "country": data.get("country", ""),
            "publications_count": data.get("works_count", 1),
            "citations": data.get("citations", 0),
            "degree": raw_deg,
            "weighted_degree": weighted_deg,
            "degree_centrality": round(deg_centrality.get(node, 0.0), 4),
            "betweenness_centrality": round(bet_centrality.get(node, 0.0), 4),
            "pagerank": round(pr_centrality.get(node, 0.0), 5),
            "closeness_centrality": round(close_centrality.get(node, 0.0), 4),
            "clustering_coefficient": round(local_clustering.get(node, 0.0), 4),
            "community_id": comm_id
        })

    df = pd.DataFrame(rows)
    # Default sort by PageRank descending
    return df.sort_values(by="pagerank", ascending=False).reset_index(drop=True)


def get_top_researchers(
    metrics_df: pd.DataFrame,
    metric_col: str = "pagerank",
    top_n: int = 10
) -> pd.DataFrame:
    """
    Get top N researchers ranked by a specific centrality metric.
    """
    if metrics_df.empty or metric_col not in metrics_df.columns:
        return metrics_df
    return metrics_df.sort_values(by=metric_col, ascending=False).head(top_n).reset_index(drop=True)
