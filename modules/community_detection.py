"""
Community detection module for ResearchCollab Explorer.
Detects research clusters using Louvain modularity optimization with a robust,
documented 4-stage fallback pipeline (Louvain -> Greedy Modularity ->
Label Propagation -> Connected Components).
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Set, Tuple
import networkx as nx

logger = logging.getLogger(__name__)

# Polished dark-theme color palette for research communities
COMMUNITY_COLOR_PALETTE = [
    "#36D6C3",  # Vibrant Teal (Primary)
    "#9B8AFB",  # Electric Violet (Secondary)
    "#38BDF8",  # Sky Cyan
    "#F472B6",  # Rose Pink
    "#FBBF24",  # Amber Gold
    "#34D399",  # Emerald Green
    "#A78BFA",  # Lilac Purple
    "#FB923C",  # Tangerine Orange
    "#F87171",  # Coral Red
    "#818CF8",  # Indigo
    "#2DD4BF",  # Aqua
    "#C084FC",  # Lavender
    "#E879F9",  # Fuchsia
    "#4ADE80",  # Bright Lime
    "#60A5FA"   # Ice Blue
]


def detect_communities(
    G: nx.Graph,
    resolution: float = 1.0,
    seed: int = 42
) -> Tuple[Dict[str, int], List[Set[str]], str, float]:
    """
    Detect communities in the collaboration network with a documented fallback chain.

    Pipeline:
    1. Louvain Communities (nx.community.louvain_communities):
       Optimizes Newman-Girvan modularity using heuristic local aggregation.
    2. Fallback 1: Greedy Modularity (nx.community.greedy_modularity_communities):
       Clauset-Newman-Moore agglomerative approach if Louvain fails or is unavailable.
    3. Fallback 2: Label Propagation (nx.community.label_propagation_communities):
       Near-linear time label dissemination algorithm for large/sparse graphs.
    4. Fallback 3: Connected Components (nx.connected_components):
       Deterministic partition based on topological reachability.

    Returns:
        (node_to_community_map, list_of_community_sets, algorithm_used_name, modularity_score)
    """
    if G.number_of_nodes() == 0:
        return {}, [], "None (Empty Graph)", 0.0

    if G.number_of_edges() == 0:
        # Each isolated node forms its own community
        isolated_sets = [{node} for node in G.nodes()]
        node_map = {node: idx for idx, node in enumerate(G.nodes())}
        return node_map, isolated_sets, "Singletons (No Edges)", 0.0

    communities_list: List[Set[str]] = []
    algorithm_used = ""

    # Stage 1: Louvain Modularity
    try:
        communities_generator = nx.community.louvain_communities(
            G,
            weight="weight",
            resolution=resolution,
            seed=seed
        )
        communities_list = [set(c) for c in communities_generator]
        algorithm_used = "Louvain Modularity Optimization (Standard)"
    except Exception as err_louvain:
        logger.warning("Louvain failed (%s), attempting Greedy Modularity fallback", err_louvain)

    # Stage 2: Fallback 1 - Greedy Modularity
    if not communities_list:
        try:
            communities_generator = nx.community.greedy_modularity_communities(
                G,
                weight="weight"
            )
            communities_list = [set(c) for c in communities_generator]
            algorithm_used = "Greedy Modularity (Clauset-Newman-Moore Fallback)"
        except Exception as err_greedy:
            logger.warning("Greedy modularity failed (%s), attempting Label Propagation", err_greedy)

    # Stage 3: Fallback 2 - Label Propagation
    if not communities_list:
        try:
            communities_generator = nx.community.label_propagation_communities(G)
            communities_list = [set(c) for c in communities_generator]
            algorithm_used = "Label Propagation (Asynchronous Fallback)"
        except Exception as err_lp:
            logger.warning("Label propagation failed (%s), attempting Connected Components", err_lp)

    # Stage 4: Fallback 3 - Connected Components
    if not communities_list:
        communities_list = [set(c) for c in nx.connected_components(G)]
        algorithm_used = "Connected Components Partitioning (Deterministic Fallback)"

    # Sort communities by size descending so Community 0 is the largest
    communities_list = sorted(communities_list, key=len, reverse=True)

    # Create mapping: node -> community_id
    node_to_comm: Dict[str, int] = {}
    for comm_idx, members in enumerate(communities_list):
        for node in members:
            node_to_comm[node] = comm_idx

    # Compute Newman-Girvan Modularity score Q
    try:
        modularity_score = nx.community.modularity(
            G,
            communities_list,
            weight="weight"
        )
    except Exception:
        modularity_score = 0.0

    return node_to_comm, communities_list, algorithm_used, round(modularity_score, 4)


def get_community_color(community_id: int) -> str:
    """
    Retrieve consistent dark-theme hex color for a given community ID.
    """
    return COMMUNITY_COLOR_PALETTE[community_id % len(COMMUNITY_COLOR_PALETTE)]


def get_community_summary(
    G: nx.Graph,
    communities_list: List[Set[str]],
    node_to_comm: Dict[str, int]
) -> List[Dict[str, Any]]:
    """
    Compile summary statistics for each detected community:
    - Member count and percentage
    - Top researchers by internal/overall degree
    - Most common institutions
    - Internal edge count vs external boundary edges
    """
    total_nodes = G.number_of_nodes()
    summaries = []

    for comm_id, members in enumerate(communities_list):
        size = len(members)
        pct = (size / total_nodes * 100.0) if total_nodes > 0 else 0.0

        # Subgraph of community members
        sub_g = G.subgraph(members)
        internal_edges = sub_g.number_of_edges()

        # Find top members by degree
        member_degrees = sorted(
            [(m, G.degree(m), G.nodes[m].get("name", m)) for m in members],
            key=lambda x: x[1],
            reverse=True
        )
        top_member_names = [name for _, _, name in member_degrees[:4]]

        # Institutional affiliations
        institutions: Dict[str, int] = {}
        for m in members:
            inst = G.nodes[m].get("institution", "Unknown Institution")
            if inst and inst != "Unknown Institution":
                institutions[inst] = institutions.get(inst, 0) + 1

        top_institutions = sorted(institutions.items(), key=lambda x: x[1], reverse=True)
        top_inst_str = ", ".join([f"{k} ({v})" for k, v in top_institutions[:2]]) if top_institutions else "Diverse / Independent"

        summaries.append({
            "community_id": comm_id,
            "color": get_community_color(comm_id),
            "size": size,
            "percentage": round(pct, 1),
            "internal_edges": internal_edges,
            "top_researchers": top_member_names,
            "key_institutions": top_inst_str
        })

    return summaries
