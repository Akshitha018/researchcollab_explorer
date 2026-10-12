"""
Export utilities module for ResearchCollab Explorer.
Generates structured CSV, JSON, and GraphML exports for academic reproducibility,
Gephi/Cytoscape integration, and thesis appendix data tables.
"""

from __future__ import annotations
import io
import json
from typing import Any, Dict, List
import networkx as nx
import pandas as pd


def generate_publications_dataframe(publications: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Format publication records into a clean tabular DataFrame.
    """
    rows = []
    for p in publications:
        author_names = ", ".join([a.get("name", "") for a in p.get("authors", [])])
        rows.append({
            "work_id": p.get("id", ""),
            "title": p.get("title", ""),
            "publication_year": p.get("publication_year", 0),
            "citations": p.get("cited_by_count", 0),
            "primary_topic": p.get("primary_topic", ""),
            "doi": p.get("doi", ""),
            "author_count": len(p.get("authors", [])),
            "authors_list": author_names
        })

    if not rows:
        return pd.DataFrame(columns=[
            "work_id", "title", "publication_year", "citations",
            "primary_topic", "doi", "author_count", "authors_list"
        ])

    return pd.DataFrame(rows).sort_values(by="citations", ascending=False).reset_index(drop=True)


def generate_authorships_dataframe(authorships: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Format authorships (paper-to-author bipartite links) into a clean DataFrame.
    """
    if not authorships:
        return pd.DataFrame(columns=[
            "publication_id", "author_id", "author_name", "position", "institution"
        ])
    return pd.DataFrame(authorships).reset_index(drop=True)


def generate_graphml_export(G: nx.Graph) -> str:
    """
    Serialize NetworkX graph to GraphML XML string for external tools (Gephi, Cytoscape).
    """
    # GraphML requires primitive attribute values (no nested lists/dicts)
    clean_g = nx.Graph()
    for n, d in G.nodes(data=True):
        clean_g.add_node(
            str(n),
            name=str(d.get("name", n)),
            institution=str(d.get("institution", "Unknown")),
            works_count=int(d.get("works_count", 1)),
            citations=int(d.get("citations", 0))
        )

    for u, v, d in G.edges(data=True):
        clean_g.add_edge(
            str(u),
            str(v),
            weight=int(d.get("weight", 1))
        )

    buffer = io.BytesIO()
    nx.write_graphml(clean_g, buffer)
    return buffer.getvalue().decode("utf-8")


def generate_summary_metrics_dataframe(summary_metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert summary graph metrics dictionary into a readable key-value DataFrame.
    """
    metric_labels = {
        "num_nodes": "Total Researchers (|V|)",
        "num_edges": "Total Unique Co-authorships (|E|)",
        "total_collaborations": "Total Weighted Collaborations",
        "density": "Network Density",
        "avg_degree": "Average Degree (Collaborators/Author)",
        "avg_clustering_coefficient": "Average Clustering Coefficient",
        "transitivity": "Network Transitivity",
        "num_components": "Connected Components Count",
        "largest_component_size": "Largest Component Size",
        "largest_component_ratio": "Largest Component Coverage (%)",
        "avg_shortest_path_length": "Avg Shortest Path Length (on LCC)",
        "diameter": "Network Diameter (on LCC)",
        "is_connected": "Is Graph Fully Connected"
    }

    rows = []
    for k, label in metric_labels.items():
        if k in summary_metrics:
            rows.append({
                "Metric": label,
                "Key": k,
                "Value": str(summary_metrics[k])
            })

    return pd.DataFrame(rows)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert DataFrame to UTF-8 CSV bytes for Streamlit st.download_button.
    """
    return df.to_csv(index=False).encode("utf-8")
