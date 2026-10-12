"""Graph construction, metrics and community detection.

Beginner-friendly glossary (n = number of nodes, m = number of edges)
---------------------------------------------------------------------
degree                 number of collaborators a researcher has.
average degree         2m / n.  (Each edge adds 1 to the degree of two nodes.)
density                m / (n(n-1)/2): the fraction of all possible edges that exist.
connected component    a group of nodes that can reach each other through edges.
clustering coefficient for a node: (edges among its neighbours) / (possible edges among
                       them). 1 = all its collaborators also collaborate with each other.
degree centrality      degree / (n - 1).
betweenness centrality fraction of shortest paths between OTHER pairs that pass through
                       the node (normalised to 0..1). High = potential "bridge".
closeness centrality   (reachable nodes) / (sum of distances to them), scaled for
                       disconnected graphs (Wasserman-Faust). High = short paths to others.
sampling               betweenness and closeness need a BFS from every node; on large
                       graphs we run it from k random "pivot" nodes instead (an estimate).
modularity             quality score of a partition into communities (higher = denser
                       inside groups than expected by chance; roughly -0.5 .. 1).
Louvain                greedy algorithm that moves nodes between groups to increase
                       modularity, then merges groups and repeats.

Nothing here measures research quality or importance; these are structural measures only.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

import community as community_louvain  # the python-louvain package
import networkx as nx
import numpy as np
import pandas as pd

DEFAULT_SEED = 42
EXACT_METRICS_MAX_NODES = 2000  # up to this size betweenness/closeness are computed exactly
DEFAULT_SAMPLE_SIZE = 500  # above it, estimate them from this many sampled source nodes

METRIC_COLUMNS = [
    "researcher_id",
    "degree",
    "degree_centrality",
    "betweenness_centrality",
    "closeness_centrality",
    "clustering_coefficient",
    "community_id",
]


# --------------------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------------------
def build_graph(edges: Optional[pd.DataFrame]) -> nx.Graph:
    """Build an undirected simple graph from a DataFrame with ``source``/``target``.

    ``nx.Graph`` already merges duplicate edges; self-loops are removed explicitly here
    as a second line of defence (the loader removes them too).
    """
    graph = nx.Graph()
    if edges is None or len(edges) == 0:
        return graph
    sources = edges["source"].astype("int64").tolist()
    targets = edges["target"].astype("int64").tolist()
    graph.add_edges_from((int(a), int(b)) for a, b in zip(sources, targets) if a != b)
    return graph


# --------------------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------------------
def network_summary(graph: nx.Graph, communities: Optional[Dict[int, int]] = None) -> dict:
    """Headline statistics, all computed from ``graph`` (nothing hard-coded)."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    degrees = np.array([d for _, d in graph.degree()], dtype=float)

    component_sizes = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
    largest = component_sizes[0] if component_sizes else 0

    summary = {
        "nodes": int(n),
        "edges": int(m),
        "average_degree": float(2 * m / n) if n else 0.0,  # 2m / n
        "median_degree": float(np.median(degrees)) if n else 0.0,
        "max_degree": int(degrees.max()) if n else 0,
        "density": float(nx.density(graph)) if n > 1 else 0.0,
        "connected_components": int(len(component_sizes)),
        "largest_component_size": int(largest),
        "largest_component_fraction": float(largest / n) if n else 0.0,
        "isolated_nodes": int(sum(1 for _ in nx.isolates(graph))),
        "average_clustering": float(nx.average_clustering(graph)) if n else 0.0,
        "transitivity": float(nx.transitivity(graph)) if n else 0.0,
        "communities": None,
        "modularity": None,
    }

    if communities:
        summary["communities"] = int(len(set(communities.values())))
        if m > 0 and set(communities) == set(graph.nodes()):
            summary["modularity"] = float(community_louvain.modularity(communities, graph))
    return summary


def degree_distribution(graph: nx.Graph) -> pd.DataFrame:
    """How many researchers have each degree: columns degree, count, fraction."""
    degrees = [d for _, d in graph.degree()]
    if not degrees:
        return pd.DataFrame({"degree": pd.Series(dtype="int64"),
                             "count": pd.Series(dtype="int64"),
                             "fraction": pd.Series(dtype="float64")})
    counts = pd.Series(degrees).value_counts().sort_index()
    out = pd.DataFrame({"degree": counts.index.astype("int64"),
                        "count": counts.values.astype("int64")})
    out["fraction"] = out["count"] / out["count"].sum()
    return out


# --------------------------------------------------------------------------------------
# Community detection
# --------------------------------------------------------------------------------------
def relabel_communities(partition: Dict[int, int]) -> Dict[int, int]:
    """Renumber communities 0, 1, 2... by size (largest first); ties by smallest member.

    Louvain's own labels are arbitrary. A deterministic numbering makes results stable,
    easy to read ("community 0 is the biggest") and testable.
    """
    groups: Dict[int, list] = {}
    for node, label in partition.items():
        groups.setdefault(label, []).append(node)
    ordered = sorted(groups.values(), key=lambda members: (-len(members), min(members)))
    relabelled: Dict[int, int] = {}
    for new_id, members in enumerate(ordered):
        for node in members:
            relabelled[int(node)] = new_id
    return relabelled


def detect_communities(
    graph: nx.Graph, seed: int = DEFAULT_SEED, resolution: float = 1.0
) -> Dict[int, int]:
    """Louvain community detection. Returns ``{researcher_id: community_id}``.

    Louvain has a random element (order in which nodes are visited). A fixed ``seed``
    makes the result reproducible for the same input.
    """
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_edges() == 0:  # nothing to optimise: every node is alone
        return {int(node): i for i, node in enumerate(sorted(graph.nodes()))}
    raw = community_louvain.best_partition(graph, resolution=resolution, random_state=seed)
    return relabel_communities(raw)


def community_summary(communities: Dict[int, int]) -> pd.DataFrame:
    """One row per community: community_id, size, share of all researchers."""
    if not communities:
        return pd.DataFrame({"community_id": pd.Series(dtype="int64"),
                             "size": pd.Series(dtype="int64"),
                             "share": pd.Series(dtype="float64")})
    sizes = pd.Series(list(communities.values())).value_counts()
    out = pd.DataFrame({"community_id": sizes.index.astype("int64"),
                        "size": sizes.values.astype("int64")})
    out = out.sort_values(["size", "community_id"], ascending=[False, True]).reset_index(drop=True)
    out["share"] = out["size"] / out["size"].sum()
    return out


# --------------------------------------------------------------------------------------
# Node metrics
# --------------------------------------------------------------------------------------
def default_sample_size(n_nodes: int) -> Optional[int]:
    """Exact betweenness/closeness for small graphs, a documented sample for large ones.

    Both measures need a shortest-path search (BFS) from EVERY node: O(n*m) in total, which
    takes minutes in pure Python for a few thousand nodes. Running the search from only
    ``k`` randomly sampled "pivot" nodes costs O(k*m) and gives an unbiased estimate.
    """
    if n_nodes <= EXACT_METRICS_MAX_NODES:
        return None  # None = exact
    return DEFAULT_SAMPLE_SIZE


def _resolve_k(n_nodes: int, k: Optional[int]) -> Optional[int]:
    """Turn 0 / None / too-large values into 'exact' (None); otherwise keep k."""
    if k is None or k <= 0 or k >= n_nodes:
        return None
    return int(k)


def _closeness_from_pivots(graph: nx.Graph, pivots) -> Dict[int, float]:
    """Estimate closeness centrality from BFS runs started at the given pivot nodes.

    Exact closeness with Wasserman-Faust scaling for a node v in a component of size s is

        closeness(v) = (r / (n - 1)) * (r / sum_of_distances)        with r = s - 1

    which equals ``(r / (n - 1)) / average_distance_to_the_r_reachable_nodes``.
    Here the average distance is estimated by the mean distance from v to the sampled pivots
    that lie in v's component; ``r`` is known exactly from the component sizes.
    A node whose component contains no pivot (a small, rarely sampled component) is simply
    computed exactly - its component is small, so this is cheap.
    """
    n = graph.number_of_nodes()
    if n <= 1:
        return {v: 0.0 for v in graph}

    component_size: Dict[int, int] = {}
    for comp in nx.connected_components(graph):
        for v in comp:
            component_size[v] = len(comp)

    dist_sum = dict.fromkeys(graph, 0)
    hits = dict.fromkeys(graph, 0)
    for pivot in pivots:
        for v, d in nx.single_source_shortest_path_length(graph, pivot).items():
            if v != pivot:
                dist_sum[v] += d
                hits[v] += 1

    closeness: Dict[int, float] = {}
    for v in graph:
        reachable = component_size[v] - 1
        if reachable == 0:
            closeness[v] = 0.0  # isolated node
            continue
        if hits[v] > 0:
            avg_distance = dist_sum[v] / hits[v]
        else:  # no pivot in this component: exact breadth-first search from v itself
            lengths = nx.single_source_shortest_path_length(graph, v)
            avg_distance = sum(lengths.values()) / reachable
        closeness[v] = (reachable / (n - 1)) / avg_distance
    return closeness


def approximate_closeness(graph: nx.Graph, k: int, seed: int = DEFAULT_SEED) -> Dict[int, float]:
    """Closeness estimated from ``k`` randomly sampled pivot nodes (reproducible via seed)."""
    nodes = sorted(graph.nodes())
    pivots = random.Random(seed).sample(nodes, min(k, len(nodes)))
    return _closeness_from_pivots(graph, pivots)


def compute_node_metrics(
    graph: nx.Graph,
    communities: Dict[int, int],
    sample_size: Optional[int] = None,
    seed: int = DEFAULT_SEED,
    compute_closeness: bool = True,
) -> pd.DataFrame:
    """Return one row per researcher with all centrality metrics and the community ID.

    ``sample_size``: None/0 (or >= number of nodes) = exact betweenness and closeness;
    otherwise both are estimated from that many sampled source nodes (``seed`` makes the
    sample reproducible). Degree, clustering and communities are always exact.
    """
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    n = graph.number_of_nodes()
    k = _resolve_k(n, sample_size)
    degree = dict(graph.degree())
    degree_c = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, k=k, normalized=True, seed=seed)
    if not compute_closeness:
        closeness = {}  # left empty -> NaN in the table
    elif k is None:
        closeness = nx.closeness_centrality(graph)  # exact; wf_improved handles disconnection
    else:
        closeness = approximate_closeness(graph, k, seed)
    clustering = nx.clustering(graph)

    ids = sorted(graph.nodes())
    table = pd.DataFrame(
        {
            "researcher_id": ids,
            "degree": [degree[i] for i in ids],
            "degree_centrality": [degree_c[i] for i in ids],
            "betweenness_centrality": [betweenness[i] for i in ids],
            "closeness_centrality": [closeness.get(i, np.nan) for i in ids],
            "clustering_coefficient": [clustering[i] for i in ids],
            "community_id": [communities.get(i, -1) for i in ids],
        }
    )
    return table[METRIC_COLUMNS]


def neighbors_table(graph: nx.Graph, metrics: pd.DataFrame, node: int) -> pd.DataFrame:
    """Immediate collaborators of ``node`` with their degree and community."""
    if node not in graph:
        return pd.DataFrame(columns=["researcher_id", "degree", "community_id"])
    neighbours = set(graph.neighbors(node))
    rows = metrics[metrics["researcher_id"].isin(neighbours)]
    rows = rows[["researcher_id", "degree", "community_id"]]
    return rows.sort_values(["degree", "researcher_id"], ascending=[False, True]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------------------
def filter_graph(
    graph: nx.Graph,
    communities: Dict[int, int],
    min_degree: int = 0,
    community_id: Optional[int] = None,
    top_n: Optional[int] = None,
    selected: Optional[int] = None,
    neighborhood_only: bool = False,
) -> nx.Graph:
    """Return a new subgraph after applying the user's filters.

    Order of operations
    1. neighbourhood: keep only ``selected`` and its immediate neighbours (if requested)
    2. community: keep only members of ``community_id``
    3. min_degree: keep nodes whose degree in the FULL graph is at least this value
    4. top_n: keep the N highest-degree nodes that remain
    The ``selected`` researcher (if present in the graph) is always kept so it can be
    highlighted.
    """
    if graph.number_of_nodes() == 0:
        return graph.copy()

    nodes = set(graph.nodes())
    keep = set()
    if selected is not None and selected in graph:
        keep.add(selected)
        if neighborhood_only:
            nodes = {selected} | set(graph.neighbors(selected))

    if community_id is not None:
        nodes = {v for v in nodes if communities.get(v) == community_id}
    if min_degree and min_degree > 0:
        nodes = {v for v in nodes if graph.degree(v) >= min_degree}

    if top_n is not None and top_n > 0 and len(nodes) > top_n:
        ranked = sorted(nodes, key=lambda v: (-graph.degree(v), v))
        top = ranked[:top_n]
        if selected in nodes and selected not in top:
            top = ranked[: max(top_n - 1, 0)] + [selected]
        nodes = set(top)

    nodes |= keep
    return graph.subgraph(nodes).copy()
