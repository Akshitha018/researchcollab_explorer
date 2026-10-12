"""Turn analysis results into downloadable files (bytes) for ``st.download_button``.

Every function is pure (no Streamlit, no disk access) so it is easy to test.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Optional

import networkx as nx
import pandas as pd


def graph_edges_dataframe(graph: nx.Graph) -> pd.DataFrame:
    """Edges of ``graph`` as a DataFrame (source < target, sorted)."""
    rows = [(min(u, v), max(u, v)) for u, v in graph.edges() if u != v]
    df = pd.DataFrame(rows, columns=["source", "target"], dtype="int64")
    return df.sort_values(["source", "target"]).reset_index(drop=True)


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def edges_to_csv(edges: pd.DataFrame) -> bytes:
    """Cleaned edge list as CSV."""
    return _to_csv_bytes(edges[["source", "target"]])


def metrics_to_csv(metrics: pd.DataFrame) -> bytes:
    """Researcher metrics as CSV."""
    return _to_csv_bytes(metrics)


def communities_to_csv(communities: Dict[int, int], restrict_to: Optional[set] = None) -> bytes:
    """Community assignments (researcher_id, community_id) as CSV."""
    items = [(int(r), int(c)) for r, c in communities.items()
             if restrict_to is None or r in restrict_to]
    df = pd.DataFrame(items, columns=["researcher_id", "community_id"], dtype="int64")
    return _to_csv_bytes(df.sort_values("researcher_id").reset_index(drop=True))


def build_stats_report(
    summary: dict,
    cleaning: Optional[dict],
    settings: dict,
    scope: str,
    filters: Optional[dict],
    dataset_label: str,
) -> dict:
    """Assemble the network-statistics report as a plain dictionary."""
    return {
        "report": "ResearchCollab Explorer network statistics",
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset": dataset_label,
        "scope": scope,
        "filters": filters or {},
        "settings": settings,
        "cleaning": cleaning or {},
        "network": summary,
    }


def report_to_json(report: dict) -> bytes:
    return json.dumps(report, indent=2, default=str).encode("utf-8")


def report_to_csv(report: dict) -> bytes:
    """Flatten the report into ``section,key,value`` rows."""
    rows = []
    for section, content in report.items():
        if isinstance(content, dict):
            for key, value in content.items():
                rows.append((section, key, value))
        else:
            rows.append(("general", section, content))
    return _to_csv_bytes(pd.DataFrame(rows, columns=["section", "key", "value"]))
