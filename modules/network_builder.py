"""
Network builder module for ResearchCollab Explorer.
Constructs undirected co-authorship graphs using NetworkX, calculates shared
publication edge weights, avoids duplicate edges & self-loops, and applies
interactive temporal and topology filters.
"""

from __future__ import annotations
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
import pandas as pd


def build_coauthorship_network(
    publications: List[Dict[str, Any]],
    authors_catalog: Dict[str, Dict[str, Any]]
) -> nx.Graph:
    """
    Construct an undirected co-authorship graph from publication records.
    
    Rules:
    - Undirected graph G = (V, E)
    - Nodes V represent researchers, keyed by stable author ID
    - Edges E represent co-authorship on at least one publication
    - Edge weight w(u, v) is the integer count of shared publications
    - Strict avoidance of duplicate edges and self-loops (u != v)
    """
    G = nx.Graph()

    # Pre-populate all nodes from authors catalog
    for auth_id, meta in authors_catalog.items():
        G.add_node(
            auth_id,
            id=auth_id,
            name=meta.get("name", "Unknown Researcher"),
            institution=meta.get("institution", "Unknown Institution"),
            country=meta.get("country", ""),
            works_count=meta.get("works_count", 0),
            citations=meta.get("total_citations", 0),
            publications_list=list(meta.get("publication_ids", []))
        )

    # Accumulate pairwise co-authorships per publication
    for pub in publications:
        pub_id = pub.get("id", "")
        pub_title = pub.get("title", "")
        pub_year = pub.get("publication_year", 0)
        
        # Deduplicate author IDs within the single publication
        work_author_ids = list(dict.fromkeys(
            [a["id"] for a in pub.get("authors", []) if a.get("id")]
        ))

        if len(work_author_ids) < 2:
            # Single-author paper, node already exists, no co-authorship edge
            continue

        # Generate every distinct pair (u, v) exactly once
        for u, v in combinations(work_author_ids, 2):
            if u == v:
                # Disallow self-loops
                continue

            # Ensure both nodes exist in the graph
            if not G.has_node(u):
                G.add_node(u, id=u, name=u, institution="Unknown", works_count=1, citations=0)
            if not G.has_node(v):
                G.add_node(v, id=v, name=v, institution="Unknown", works_count=1, citations=0)

            pub_record = {
                "id": pub_id,
                "title": pub_title,
                "year": pub_year
            }

            if G.has_edge(u, v):
                G[u][v]["weight"] += 1
                G[u][v]["shared_publications"].append(pub_record)
            else:
                G.add_edge(
                    u,
                    v,
                    weight=1,
                    shared_publications=[pub_record]
                )

    return G


def filter_network(
    G: nx.Graph,
    min_weight: int = 1,
    year_range: Optional[Tuple[int, int]] = None,
    target_researcher_id: Optional[str] = None,
    ego_radius: int = 1,
    remove_isolates: bool = False
) -> nx.Graph:
    """
    Produce a filtered sub-graph based on user controls:
    - year_range: (min_year, max_year) filters shared publications; unknown years are excluded
    - min_weight: minimum number of shared publications required
    - target_researcher_id: extracts ego network around researcher
    - remove_isolates: filters out disconnected nodes (degree = 0)
    """
    # Start with a shallow copy or newly constructed sub-graph
    filtered_G = nx.Graph()

    # Copy all nodes
    for node, data in G.nodes(data=True):
        filtered_G.add_node(node, **data)

    # Filter edges based on publication years and weight
    for u, v, data in G.edges(data=True):
        shared_pubs = data.get("shared_publications", [])

        if year_range is not None:
            min_yr, max_yr = year_range
            valid_pubs = [
                p for p in shared_pubs
                if isinstance(p.get("year"), int)
                and min_yr <= p["year"] <= max_yr
            ]
        else:
            valid_pubs = shared_pubs

        effective_weight = len(valid_pubs)

        if effective_weight >= min_weight and effective_weight > 0:
            filtered_G.add_edge(
                u,
                v,
                weight=effective_weight,
                shared_publications=valid_pubs
            )

    # Ego network filter if requested
    if target_researcher_id and filtered_G.has_node(target_researcher_id):
        ego_nodes = nx.single_source_shortest_path_length(
            filtered_G,
            target_researcher_id,
            cutoff=max(1, min(ego_radius, 3))
        ).keys()
        filtered_G = filtered_G.subgraph(ego_nodes).copy()

    # Remove isolated nodes if requested
    if remove_isolates:
        isolated = [node for node, deg in filtered_G.degree() if deg == 0]
        filtered_G.remove_nodes_from(isolated)

    return filtered_G


def get_edge_dataframe(G: nx.Graph) -> pd.DataFrame:
    """
    Convert graph edges into a structured pandas DataFrame.
    """
    rows = []
    for u, v, data in G.edges(data=True):
        u_name = G.nodes[u].get("name", u)
        v_name = G.nodes[v].get("name", v)
        weight = data.get("weight", 1)
        shared = data.get("shared_publications", [])
        shared_titles = "; ".join([p.get("title", "") for p in shared[:3]])
        if len(shared) > 3:
            shared_titles += f" (+{len(shared)-3} more)"

        rows.append({
            "source_id": u,
            "source_name": u_name,
            "target_id": v,
            "target_name": v_name,
            "weight": weight,
            "shared_papers_count": len(shared),
            "sample_shared_titles": shared_titles
        })

    if not rows:
        return pd.DataFrame(columns=[
            "source_id", "source_name", "target_id", "target_name",
            "weight", "shared_papers_count", "sample_shared_titles"
        ])

    df = pd.DataFrame(rows)
    return df.sort_values(by="weight", ascending=False).reset_index(drop=True)


def get_node_dataframe(G: nx.Graph) -> pd.DataFrame:
    """
    Convert graph nodes into a structured pandas DataFrame.
    """
    rows = []
    for node, data in G.nodes(data=True):
        deg = G.degree(node)
        weighted_deg = sum([edata.get("weight", 1) for _, _, edata in G.edges(node, data=True)])
        rows.append({
            "author_id": node,
            "name": data.get("name", "Unknown"),
            "institution": data.get("institution", "Unknown"),
            "country": data.get("country", ""),
            "publications_in_corpus": data.get("works_count", 1),
            "citations": data.get("citations", 0),
            "degree": deg,
            "weighted_degree": weighted_deg
        })

    if not rows:
        return pd.DataFrame(columns=[
            "author_id", "name", "institution", "country",
            "publications_in_corpus", "citations", "degree", "weighted_degree"
        ])

    df = pd.DataFrame(rows)
    return df.sort_values(by="degree", ascending=False).reset_index(drop=True)
