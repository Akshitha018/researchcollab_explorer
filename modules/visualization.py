"""
Visualization module for ResearchCollab Explorer.
Builds interactive Plotly network graphs with dark themes, zoom/pan controls,
ego-network highlights, community coloring, and statistical charts.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from modules.community_detection import get_community_color

# Theme Constants
BG_COLOR = "#0B0F19"
PANEL_COLOR = "#141B2D"
PRIMARY_TEAL = "#36D6C3"
SECONDARY_VIOLET = "#9B8AFB"
TEXT_COLOR = "#E2E8F0"
MUTED_TEXT = "#94A3B8"
DIMMED_EDGE = "rgba(40, 50, 75, 0.4)"
ACTIVE_EDGE = "rgba(155, 138, 251, 0.65)"
HIGHLIGHT_EDGE = "rgba(54, 214, 195, 0.9)"


def compute_graph_layout(G: nx.Graph, layout_type: str = "spring") -> Dict[str, np.ndarray]:
    """
    Compute 2D coordinates for graph nodes.
    Supports Spring (Fruchterman-Reingold), Kamada-Kawai, and Circular layouts.
    """
    n = G.number_of_nodes()
    if n == 0:
        return {}

    if layout_type == "kamada_kawai" and n < 500:
        try:
            return nx.kamada_kawai_layout(G)
        except Exception:
            pass
    elif layout_type == "circular":
        return nx.circular_layout(G)

    # Default: Optimized Spring layout
    k_dist = 0.85 / np.sqrt(max(1, n))
    return nx.spring_layout(G, k=k_dist, iterations=75, seed=42)


def create_network_figure(
    G: nx.Graph,
    metrics_df: pd.DataFrame,
    layout_type: str = "spring",
    size_by: str = "pagerank",
    color_by: str = "community",
    search_query: Optional[str] = None,
    highlight_author_id: Optional[str] = None
) -> go.Figure:
    """
    Generate an interactive Plotly 2D network visualization.
    """
    fig = go.Figure()

    if G.number_of_nodes() == 0:
        fig.add_annotation(
            text="No nodes or edges match the current filters.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color=TEXT_COLOR, size=16)
        )
        _apply_dark_theme(fig, title="Co-Authorship Network (Empty)")
        return fig

    pos = compute_graph_layout(G, layout_type=layout_type)
    metrics_lookup = metrics_df.set_index("author_id").to_dict(orient="index") if not metrics_df.empty else {}

    # Determine highlighted node set if search or selection is active
    highlighted_nodes = set()
    if highlight_author_id and highlight_author_id in G:
        highlighted_nodes.add(highlight_author_id)
        highlighted_nodes.update(G.neighbors(highlight_author_id))
    elif search_query:
        sq_lower = search_query.strip().lower()
        for node, data in G.nodes(data=True):
            if sq_lower in data.get("name", "").lower():
                highlighted_nodes.add(node)
                highlighted_nodes.update(G.neighbors(node))

    has_highlight = len(highlighted_nodes) > 0

    # Group edge segments by shared-publication count to encode tie strength.
    edge_groups = {}

    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        is_edge_highlighted = (u in highlighted_nodes and v in highlighted_nodes) if has_highlight else False
        weight = max(1, int(data.get("weight", 1)))
        group = edge_groups.setdefault((weight, is_edge_highlighted), ([], [], []))
        edge_x, edge_y, edge_text = group
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        source_name = G.nodes[u].get("name", u)
        target_name = G.nodes[v].get("name", v)
        shared = data.get("shared_publications", [])
        titles = [publication.get("title", "") for publication in shared[:3]]
        title_text = "<br>".join(titles)
        tooltip = (
            f"<b>{source_name} ↔ {target_name}</b><br>"
            f"Shared publications: {weight}"
            f"{'<br>' + title_text if title_text else ''}"
        )
        edge_text.extend([tooltip, tooltip, ""])

    for (weight, highlighted), (edge_x, edge_y, edge_text) in sorted(
        edge_groups.items(),
        key=lambda item: (item[0][1], item[0][0])
    ):
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            text=edge_text,
            line=dict(
                width=min(4.0, 1.2 + (weight - 1) * 0.7),
                color=PRIMARY_TEAL if highlighted else (
                    DIMMED_EDGE if has_highlight else ACTIVE_EDGE
                )
            ),
            hoverinfo="text",
            mode="lines",
            name=f"{weight} shared publication{'s' if weight != 1 else ''}",
            showlegend=False
        ))

    # 2. BUILD NODES
    node_x, node_y = [], []
    node_sizes = []
    node_colors = []
    node_line_widths = []
    node_line_colors = []
    custom_data = []
    hover_texts = []

    # Scaling factor for node sizes
    size_values = []
    for node in G.nodes():
        m = metrics_lookup.get(node, {})
        if size_by == "pagerank":
            val = m.get("pagerank", 0.01)
        elif size_by == "betweenness":
            val = m.get("betweenness_centrality", 0.0)
        elif size_by == "degree_centrality":
            val = m.get("degree_centrality", 0.0)
        else:
            val = G.degree(node)
        size_values.append(val)

    val_min = min(size_values) if size_values else 0
    val_max = max(size_values) if size_values else 1
    val_range = val_max - val_min if val_max > val_min else 1.0
    pagerank_values = [
        metrics_lookup.get(node, {}).get("pagerank", 0.0)
        for node in G.nodes()
    ]
    pagerank_min = min(pagerank_values) if pagerank_values else 0.0
    pagerank_range = max(pagerank_values) - pagerank_min if pagerank_values else 0.0

    for idx, node in enumerate(G.nodes()):
        if node not in pos:
            continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        node_data = G.nodes[node]
        m = metrics_lookup.get(node, {})
        name = node_data.get("name", node)
        inst = node_data.get("institution", "Unknown Institution")
        deg = G.degree(node)
        pr = m.get("pagerank", 0.0)
        bet = m.get("betweenness_centrality", 0.0)
        comm = m.get("community_id", 0)
        n_pubs = node_data.get("works_count", 1)

        # Normalized size calculation: 12px to 38px
        norm = (size_values[idx] - val_min) / val_range
        calc_size = 12 + int(norm * 26)

        # Highlight check
        is_node_highlighted = node in highlighted_nodes if has_highlight else True

        if not is_node_highlighted:
            # Dimmed background node
            node_sizes.append(max(8, calc_size - 4))
            node_colors.append("rgba(75, 85, 99, 0.4)")
            node_line_widths.append(0.5)
            node_line_colors.append("rgba(100, 116, 139, 0.3)")
        else:
            node_sizes.append(calc_size)
            if color_by == "community":
                node_colors.append(get_community_color(comm))
            elif color_by == "pagerank":
                pagerank_norm = (
                    (pr - pagerank_min) / pagerank_range
                    if pagerank_range > 0
                    else 0.5
                )
                node_colors.append(
                    sample_colorscale(
                        [SECONDARY_VIOLET, PRIMARY_TEAL],
                        [pagerank_norm]
                    )[0]
                )
            else:
                node_colors.append(PRIMARY_TEAL)

            if has_highlight and (node == highlight_author_id or (search_query and search_query.lower() in name.lower())):
                node_line_widths.append(3.0)
                node_line_colors.append("#FFFFFF")
            else:
                node_line_widths.append(1.2)
                node_line_colors.append(PANEL_COLOR)

        custom_data.append([
            name, inst, deg, pr, bet, comm, n_pubs, node
        ])

        hover_text = (
            f"<b>{name}</b><br>"
            f"🏛 {inst}<br>"
            f"🤝 Degree (Co-authors): {deg}<br>"
            f"⭐ PageRank: {pr:.4f}<br>"
            f"🌉 Betweenness: {bet:.4f}<br>"
            f"👥 Community: #{comm}<br>"
            f"📄 Corpus Publications: {n_pubs}"
        )
        hover_texts.append(hover_text)

    # Node Scatter Trace
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=node_line_widths, color=node_line_colors),
            opacity=0.95
        ),
        text=hover_texts,
        hoverinfo="text",
        customdata=custom_data,
        name="Researchers",
        showlegend=False
    ))

    # Clean layout styling
    title_suffix = f" ({len(G.nodes)} researchers, {len(G.edges)} collaborations)"
    _apply_dark_theme(fig, title=f"Researcher Collaboration Graph{title_suffix}")
    return fig


def create_degree_distribution_chart(metrics_df: pd.DataFrame) -> go.Figure:
    """
    Plot degree distribution histogram with power-law scale-free inspection.
    """
    fig = go.Figure()
    if metrics_df.empty:
        return fig

    degrees = metrics_df["degree"].values
    fig.add_trace(go.Histogram(
        x=degrees,
        nbinsx=max(10, int(degrees.max()) if len(degrees) > 0 and degrees.max() > 0 else 10),
        marker=dict(
            color=PRIMARY_TEAL,
            line=dict(color=PANEL_COLOR, width=1.5)
        ),
        opacity=0.85,
        name="Researchers Count"
    ))

    _apply_dark_theme(
        fig,
        title="Collaboration Degree Distribution (Descriptive)",
        xaxis_title="Degree (Number of Direct Co-Authors)",
        yaxis_title="Frequency (Researcher Count)"
    )
    return fig


def create_centrality_scatter_chart(metrics_df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot comparing PageRank (Authority) vs Betweenness Centrality (Brokerage),
    with bubble size proportional to Degree.
    """
    fig = go.Figure()
    if metrics_df.empty:
        return fig

    colors = [get_community_color(c) for c in metrics_df["community_id"]]
    sizes = np.clip(metrics_df["degree"].values * 2.5 + 8, 8, 32)

    hover_texts = [
        f"<b>{row['name']}</b><br>"
        f"🏛 {row['institution']}<br>"
        f"⭐ PageRank: {row['pagerank']:.5f}<br>"
        f"🌉 Betweenness: {row['betweenness_centrality']:.4f}<br>"
        f"🤝 Degree: {row['degree']}<br>"
        f"👥 Community: #{row['community_id']}"
        for _, row in metrics_df.iterrows()
    ]

    fig.add_trace(go.Scatter(
        x=metrics_df["betweenness_centrality"],
        y=metrics_df["pagerank"],
        mode="markers",
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(color=PANEL_COLOR, width=1),
            opacity=0.88
        ),
        text=hover_texts,
        hoverinfo="text",
        name="Researchers"
    ))

    _apply_dark_theme(
        fig,
        title="Centrality Analysis: PageRank (Prestige) vs Betweenness (Bridge Brokerage)",
        xaxis_title="Betweenness Centrality (Brokerage Role)",
        yaxis_title="PageRank Centrality (Network Authority)"
    )
    return fig


def create_publication_trends_chart(publications: List[Dict[str, Any]]) -> go.Figure:
    """
    Plot publication count trends by year with dark aesthetic.
    """
    fig = go.Figure()
    if not publications:
        return fig

    years = [p.get("publication_year", 0) for p in publications if p.get("publication_year", 0) > 0]
    if not years:
        return fig

    year_counts = pd.Series(years).value_counts().sort_index()

    fig.add_trace(go.Bar(
        x=year_counts.index.astype(str),
        y=year_counts.values,
        marker=dict(
            color=SECONDARY_VIOLET,
            line=dict(color=PANEL_COLOR, width=1.5)
        ),
        name="Publications per Year"
    ))

    _apply_dark_theme(
        fig,
        title="Corpus Publication Output Over Time",
        xaxis_title="Publication Year",
        yaxis_title="Number of Publications"
    )
    return fig


def create_community_breakdown_chart(community_summaries: List[Dict[str, Any]]) -> go.Figure:
    """
    Plot community member count donut chart.
    """
    fig = go.Figure()
    if not community_summaries:
        return fig

    labels = [f"Community #{c['community_id']}" for c in community_summaries]
    values = [c["size"] for c in community_summaries]
    colors = [c["color"] for c in community_summaries]

    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        hole=0.48,
        marker=dict(colors=colors, line=dict(color=PANEL_COLOR, width=2)),
        textinfo="label+percent",
        hoverinfo="label+value+percent",
        textfont=dict(color="#FFFFFF", size=12)
    ))

    _apply_dark_theme(fig, title="Community Membership Breakdown")
    return fig


def _apply_dark_theme(
    fig: go.Figure,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = ""
) -> None:
    """
    Apply unified premium dark theme styling (#0B0F19 background, #141B2D panel, teal/violet accents).
    """
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(color=TEXT_COLOR, size=15),
            x=0.02,
            y=0.96
        ),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin=dict(l=40, r=40, t=55, b=40),
        font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        showlegend=False,
        hoverlabel=dict(
            bgcolor=PANEL_COLOR,
            bordercolor=PRIMARY_TEAL,
            font=dict(color="#FFFFFF", size=13, family="Inter, system-ui, sans-serif")
        )
    )

    if xaxis_title or yaxis_title:
        fig.update_xaxes(
            title=dict(text=xaxis_title, font=dict(color=MUTED_TEXT, size=12)),
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
            zeroline=False,
            tickfont=dict(color=MUTED_TEXT, size=11)
        )
        fig.update_yaxes(
            title=dict(text=yaxis_title, font=dict(color=MUTED_TEXT, size=12)),
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
            zeroline=False,
            tickfont=dict(color=MUTED_TEXT, size=11)
        )
    else:
        # Hide axes for network view
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)


def create_top_collaborators_chart(
    metrics_df: pd.DataFrame,
    metric_col: str = "degree",
    top_n: int = 10
) -> go.Figure:
    """
    Horizontal bar chart of top researchers ranked by degree or PageRank.
    """
    fig = go.Figure()
    if metrics_df.empty or metric_col not in metrics_df.columns:
        return fig

    top_df = metrics_df.sort_values(by=metric_col, ascending=True).tail(top_n)

    labels = top_df["name"].tolist()
    values = top_df[metric_col].tolist()

    metric_name_display = "Direct Co-Authors (Degree)" if metric_col == "degree" else "PageRank Centrality"

    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker=dict(
            color=PRIMARY_TEAL,
            line=dict(color=PANEL_COLOR, width=1.5)
        ),
        text=[f"{v:.4f}" if isinstance(v, float) else str(v) for v in values],
        textposition="inside",
        textfont=dict(color="#0B0F19", size=11, family="Inter, sans-serif"),
        hoverinfo="y+x"
    ))

    _apply_dark_theme(
        fig,
        title=f"Top {top_n} Researchers by {metric_name_display}",
        xaxis_title=metric_name_display,
        yaxis_title=""
    )
    fig.update_layout(margin=dict(l=150, r=40, t=50, b=40))
    return fig


def create_ego_figure(
    G: nx.Graph,
    center_node: str,
    metrics_df: pd.DataFrame
) -> go.Figure:
    """
    Build an interactive Ego Network subgraph centered around a single researcher.
    """
    fig = go.Figure()
    if not G.has_node(center_node):
        fig.add_annotation(
            text="Selected researcher not found in graph.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(color=TEXT_COLOR, size=14)
        )
        _apply_dark_theme(fig, title="Ego Network (Unavailable)")
        return fig

    ego_g = nx.ego_graph(G, center_node, radius=1)
    if ego_g.number_of_nodes() <= 1:
        fig.add_annotation(
            text="Researcher has no co-authors in this active filtered corpus (Solo Author).",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(color=TEXT_COLOR, size=14)
        )
        _apply_dark_theme(fig, title=f"Ego Network for {G.nodes[center_node].get('name', center_node)}")
        return fig

    pos = nx.spring_layout(ego_g, seed=42, k=1.0)
    center_name = G.nodes[center_node].get("name", center_node)

    # Edge traces
    edge_x, edge_y = [], []
    for u, v in ego_g.edges():
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=2.0, color="rgba(54, 214, 195, 0.65)"),
            hoverinfo="none",
            showlegend=False
        ))

    # Node coordinates
    node_x, node_y = [], []
    node_colors = []
    node_sizes = []
    node_texts = []
    node_borders = []
    node_border_colors = []

    for n in ego_g.nodes():
        if n not in pos:
            continue
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)

        name = ego_g.nodes[n].get("name", n)
        inst = ego_g.nodes[n].get("institution", "Unknown")
        deg = G.degree(n)

        if n == center_node:
            node_colors.append(PRIMARY_TEAL)
            node_sizes.append(34)
            node_borders.append(3.0)
            node_border_colors.append("#FFFFFF")
            node_texts.append(f"<b>⭐ CENTER: {name}</b><br>🏛 {inst}<br>🤝 Total Degree: {deg}")
        else:
            shared = G[center_node][n].get("weight", 1) if G.has_edge(center_node, n) else 1
            node_colors.append(SECONDARY_VIOLET)
            node_sizes.append(18 + min(16, shared * 4))
            node_borders.append(1.5)
            node_border_colors.append(PANEL_COLOR)
            node_texts.append(f"<b>{name}</b><br>🏛 {inst}<br>🤝 Shared Papers: {shared}<br>Total Degree: {deg}")

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=node_borders, color=node_border_colors),
            opacity=0.95
        ),
        text=[ego_g.nodes[n].get("name", "") if n == center_node or len(ego_g) < 12 else "" for n in ego_g.nodes()],
        textposition="top center",
        textfont=dict(color="#FFFFFF", size=11, family="Inter, sans-serif"),
        hovertext=node_texts,
        hoverinfo="text",
        showlegend=False
    ))

    _apply_dark_theme(fig, title=f"1-Hop Ego Network: {center_name} ({len(ego_g)-1} Direct Collaborators)")
    return fig
