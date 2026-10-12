"""Visualization: PyVis interactive network (as an HTML string) and Plotly charts.

The PyVis HTML is built entirely in memory (``Network.generate_html``) and the JavaScript
library is inlined (``cdn_resources="in_line"``), so no temp folder, no ``lib/`` folder and
no internet connection are needed to display the network. If ``generate_html`` is ever
unavailable, a ``tempfile.TemporaryDirectory`` is used as a fallback, so no particular
directory is assumed to exist.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network

# --------------------------------------------------------------------------------------
# Design tokens (kept in one place so the whole app looks consistent)
# --------------------------------------------------------------------------------------
PRIMARY = "#4F46E5"
TEXT = "#0F172A"
MUTED = "#64748B"
GRID = "#E2E8F0"
HIGHLIGHT = "#F59E0B"
FONT = "Inter, Segoe UI, Roboto, Arial, sans-serif"

COMMUNITY_PALETTE = [
    "#4F46E5", "#0EA5E9", "#10B981", "#F59E0B", "#EF4444",
    "#8B5CF6", "#EC4899", "#14B8A6", "#F97316", "#84CC16",
    "#6366F1", "#06B6D4", "#22C55E", "#EAB308", "#F43F5E",
    "#A855F7", "#0891B2", "#65A30D", "#DC2626", "#D946EF",
]


def community_color(community_id: int) -> str:
    """Colour for a community. Colours repeat after 20 communities (documented limit)."""
    return COMMUNITY_PALETTE[int(community_id) % len(COMMUNITY_PALETTE)]


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------------------------------------
# PyVis network
# --------------------------------------------------------------------------------------
def scale_sizes(degrees: Dict[int, int], min_size: float = 8.0, max_size: float = 40.0) -> Dict[int, float]:
    """Map degree -> node size. A square root keeps hubs from dwarfing everything else."""
    if not degrees:
        return {}
    lo, hi = min(degrees.values()), max(degrees.values())
    if hi == lo:
        return {n: (min_size + max_size) / 2 for n in degrees}
    return {
        n: min_size + (max_size - min_size) * math.sqrt((d - lo) / (hi - lo))
        for n, d in degrees.items()
    }


def _network_options(physics: bool) -> str:
    # NOTE: PyVis removes ALL spaces from this string before parsing it, so values must
    # not contain spaces (that is why font names below are written without spaces).
    options = {
        "nodes": {
            "shape": "dot",
            "borderWidth": 1.5,
            "borderWidthSelected": 4,
            "font": {"size": 14, "face": "arial", "color": TEXT,
                     "strokeWidth": 3, "strokeColor": "#FFFFFF"},
        },
        "edges": {"smooth": False, "selectionWidth": 2, "hoverWidth": 1.5,
                  "color": {"inherit": False}},
        "interaction": {
            "hover": True,
            "tooltipDelay": 100,
            "navigationButtons": True,
            "keyboard": True,
            "multiselect": True,
            "selectConnectedEdges": True,
            "hideEdgesOnDrag": True,
        },
        "physics": {
            "enabled": bool(physics),
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -45,
                "centralGravity": 0.012,
                "springLength": 90,
                "springConstant": 0.08,
                "damping": 0.5,
                "avoidOverlap": 0.2,
            },
            "stabilization": {"enabled": True, "iterations": 200, "updateInterval": 25},
            "minVelocity": 0.75,
        },
    }
    return json.dumps(options)


def build_network_html(
    graph: nx.Graph,
    communities: Dict[int, int],
    metrics: pd.DataFrame,
    selected: Optional[int] = None,
    height_px: int = 720,
    physics: bool = True,
) -> str:
    """Return a self-contained HTML page with an interactive force-directed network.

    * node colour  = Louvain community
    * node size    = degree (square-root scaled)
    * tooltip      = ID, degree, centralities, community (plain text, readable on hover)
    * ``selected`` = highlighted in amber; its neighbours get a bold outline and
                     everything else is faded so the neighbourhood stands out.
    """
    net = Network(
        height=f"{int(height_px)}px",
        width="100%",
        bgcolor="#FFFFFF",
        font_color=TEXT,
        notebook=False,
        cdn_resources="in_line",
        directed=False,
    )

    lookup = metrics.set_index("researcher_id").to_dict("index") if len(metrics) else {}
    degrees = {int(n): int(d) for n, d in graph.degree()}
    sizes = scale_sizes(degrees)

    has_selection = selected is not None and selected in graph
    neighbours = set(graph.neighbors(selected)) if has_selection else set()
    top_labelled = set(sorted(degrees, key=lambda v: (-degrees[v], v))[:20])
    label_everything = graph.number_of_nodes() <= 150

    for node in graph.nodes():
        node = int(node)
        info = lookup.get(node, {})
        comm = communities.get(node, info.get("community_id", -1))
        base = community_color(comm) if comm >= 0 else "#94A3B8"

        tooltip = (
            f"Researcher ID: {node}\n"
            f"Degree: {degrees[node]}\n"
            f"Degree centrality: {info.get('degree_centrality', float('nan')):.4f}\n"
            f"Betweenness: {info.get('betweenness_centrality', float('nan')):.4f}\n"
            f"Community: {comm}"
        )

        size = sizes[node]
        border, border_width = "#FFFFFF", 1.5
        background = base
        if has_selection:
            if node == selected:
                background, border, border_width, size = HIGHLIGHT, "#B45309", 5, size * 1.5 + 6
            elif node in neighbours:
                border, border_width = TEXT, 3
            else:
                background = _rgba(base, 0.25)

        show_label = label_everything or node == selected or node in neighbours or node in top_labelled
        net.add_node(
            node,
            label=str(node) if show_label else " ",
            title=tooltip,
            size=float(size),
            borderWidth=border_width,
            color={
                "background": background,
                "border": border,
                "highlight": {"background": background, "border": TEXT},
                "hover": {"background": background, "border": TEXT},
            },
        )

    for u, v in graph.edges():
        touches_selected = has_selection and (u == selected or v == selected)
        if touches_selected:
            color, width = HIGHLIGHT, 2.5
        elif has_selection:
            color, width = "rgba(148,163,184,0.15)", 1
        else:
            color, width = "rgba(148,163,184,0.55)", 1
        net.add_edge(int(u), int(v), color=color, width=width)

    net.set_options(_network_options(physics))
    return _render_html(net)


def _render_html(net: Network) -> str:
    """Get the HTML string from PyVis without assuming any directory exists."""
    try:
        return net.generate_html()
    except AttributeError:  # very old PyVis without generate_html
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "network.html"
            net.write_html(str(target))
            return target.read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------
# Plotly charts
# --------------------------------------------------------------------------------------
def _style(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.0, "font": {"size": 16, "color": TEXT}},
        template="plotly_white",
        height=height,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        font={"family": FONT, "color": TEXT},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel={"font": {"family": FONT}},
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _empty_figure(title: str, message: str = "No data to display") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font={"size": 14, "color": MUTED})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, title)


def degree_distribution_chart(dist: pd.DataFrame, log_log: bool = True) -> go.Figure:
    """Number of researchers (y) having each degree (x). Log-log shows the long tail."""
    title = "Degree distribution"
    if dist.empty:
        return _empty_figure(title)
    fig = go.Figure(
        go.Scatter(
            x=dist["degree"], y=dist["count"], mode="markers",
            marker={"color": PRIMARY, "size": 8, "opacity": 0.8},
            hovertemplate="Degree %{x}<br>%{y} researchers<extra></extra>",
        )
    ) if log_log else go.Figure(
        go.Bar(x=dist["degree"], y=dist["count"], marker_color=PRIMARY,
               hovertemplate="Degree %{x}<br>%{y} researchers<extra></extra>")
    )
    fig.update_xaxes(title="Degree (number of collaborators)", type="log" if log_log else "linear")
    fig.update_yaxes(title="Number of researchers", type="log" if log_log else "linear")
    return _style(fig, title)


def top_metric_chart(metrics: pd.DataFrame, metric: str = "degree", n: int = 10,
                     title: Optional[str] = None) -> go.Figure:
    """Horizontal bars for the top ``n`` researchers by ``metric``."""
    label = metric.replace("_", " ").capitalize()
    title = title or f"Top {n} researchers by {metric.replace('_', ' ')}"
    data = metrics.dropna(subset=[metric]) if len(metrics) else metrics
    if data.empty:
        return _empty_figure(title)
    top = data.sort_values([metric, "researcher_id"], ascending=[False, True]).head(n)
    top = top.assign(label="ID " + top["researcher_id"].astype(str))
    fig = go.Figure(
        go.Bar(
            x=top[metric], y=top["label"], orientation="h", marker_color=PRIMARY,
            customdata=top[["community_id"]].to_numpy(),
            hovertemplate="%{y}<br>" + label + ": %{x}<br>Community %{customdata[0]}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title=label)
    return _style(fig, title)


def community_size_chart(summary: pd.DataFrame, max_bars: int = 25) -> go.Figure:
    """Bar chart of the largest communities (size = number of researchers)."""
    title = "Community sizes"
    if summary.empty:
        return _empty_figure(title)
    top = summary.head(max_bars).copy()
    top["label"] = "C" + top["community_id"].astype(str)
    fig = go.Figure(
        go.Bar(
            x=top["label"], y=top["size"],
            marker_color=[community_color(c) for c in top["community_id"]],
            hovertemplate="Community %{customdata}<br>%{y} researchers<extra></extra>",
            customdata=top["community_id"],
        )
    )
    fig.update_xaxes(title=f"Community (largest {len(top)} shown)", type="category")
    fig.update_yaxes(title="Researchers")
    return _style(fig, title)


def community_size_histogram(summary: pd.DataFrame) -> go.Figure:
    """Distribution of community sizes: how many communities have each size."""
    title = "Community size distribution"
    if summary.empty:
        return _empty_figure(title)
    counts = summary["size"].value_counts().sort_index()
    fig = go.Figure(
        go.Bar(x=counts.index, y=counts.values, marker_color=PRIMARY,
               hovertemplate="Size %{x}<br>%{y} communities<extra></extra>")
    )
    fig.update_xaxes(title="Community size (researchers)", type="log")
    fig.update_yaxes(title="Number of communities")
    return _style(fig, title)


def degree_vs_betweenness_chart(metrics: pd.DataFrame, title: Optional[str] = None) -> go.Figure:
    """Scatter plot: each dot is a researcher; colour = community."""
    title = title or "Degree vs betweenness centrality"
    if metrics.empty:
        return _empty_figure(title)
    data = metrics.assign(community=metrics["community_id"].astype(str))
    fig = px.scatter(
        data, x="degree", y="betweenness_centrality", color="community",
        color_discrete_sequence=COMMUNITY_PALETTE,
        hover_data={"researcher_id": True, "degree": True, "community": True,
                    "betweenness_centrality": ":.5f"},
    )
    fig.update_traces(marker={"size": 8, "opacity": 0.8})
    fig.update_xaxes(title="Degree")
    fig.update_yaxes(title="Betweenness centrality")
    fig.update_layout(showlegend=False)
    return _style(fig, title)
