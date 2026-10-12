"""
ResearchCollab Explorer
Research Collaboration Network Analysis and Visualization Using Graph Theory

Main Streamlit Application
"""

from __future__ import annotations
import os
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

from modules.data_loader import (
    load_offline_dataset,
    fetch_openalex_works,
    OFFLINE_DATASET_MAP
)
from modules.network_builder import (
    build_coauthorship_network,
    filter_network,
    get_edge_dataframe,
    get_node_dataframe
)
from modules.graph_metrics import (
    compute_network_summary_metrics,
    compute_node_centrality_metrics,
    get_top_researchers
)
from modules.community_detection import (
    detect_communities,
    get_community_summary,
    get_community_color
)
from modules.visualization import (
    create_network_figure,
    create_degree_distribution_chart,
    create_centrality_scatter_chart,
    create_publication_trends_chart,
    create_community_breakdown_chart,
    create_top_collaborators_chart,
    create_ego_figure
)
from modules.export_utils import (
    generate_publications_dataframe,
    generate_authorships_dataframe,
    generate_graphml_export,
    generate_summary_metrics_dataframe,
    dataframe_to_csv_bytes
)

# Set page configuration first
st.set_page_config(
    page_title="ResearchCollab Explorer",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Dark Theme CSS
CUSTOM_CSS = """
<style>
/* Base Dark Theme Tokens */
:root {
    --bg-main: #0B0F19;
    --panel-main: #141B2D;
    --panel-border: rgba(54, 214, 195, 0.2);
    --teal-accent: #36D6C3;
    --violet-accent: #9B8AFB;
    --text-primary: #F1F5F9;
    --text-secondary: #94A3B8;
}

/* App Background & Typography */
.stApp {
    background-color: var(--bg-main);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: var(--panel-main);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Top App Header Banner */
.app-header {
    background: linear-gradient(135deg, rgba(20, 27, 45, 0.95) 0%, rgba(11, 15, 25, 0.95) 100%);
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}
.app-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.app-subtitle {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-top: 6px;
    margin-bottom: 0;
}

/* Metric Cards */
.metric-card {
    background: var(--panel-main);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 16px 18px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    border-color: var(--teal-accent);
}
.metric-label {
    color: var(--text-secondary);
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.metric-value {
    color: var(--teal-accent);
    font-size: 1.85rem;
    font-weight: 700;
    line-height: 1.2;
}
.metric-sub {
    color: var(--text-secondary);
    font-size: 0.75rem;
    margin-top: 4px;
}

/* Mode Badges */
.badge-live {
    background: rgba(54, 214, 195, 0.15);
    color: var(--teal-accent);
    border: 1px solid rgba(54, 214, 195, 0.35);
    padding: 3px 10px;
    border-radius: 16px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
}
.badge-sample {
    background: rgba(155, 138, 251, 0.15);
    color: var(--violet-accent);
    border: 1px solid rgba(155, 138, 251, 0.35);
    padding: 3px 10px;
    border-radius: 16px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
}

/* Info and Alert boxes */
.info-card {
    background: rgba(20, 27, 45, 0.85);
    border-left: 4px solid var(--teal-accent);
    padding: 14px 18px;
    border-radius: 0 8px 8px 0;
    margin: 12px 0;
    color: var(--text-primary);
    font-size: 0.9rem;
}

/* Dataframe & Table Overrides */
div[data-testid="stDataFrame"] {
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

/* Hide Streamlit default decoration */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Session State
if "dataset" not in st.session_state:
    st.session_state["dataset"] = load_offline_dataset("gnn")
if "data_source_mode" not in st.session_state:
    st.session_state["data_source_mode"] = "sample"  # 'sample' or 'live'
if "active_dataset_key" not in st.session_state:
    st.session_state["active_dataset_key"] = "gnn"


# ==========================================
# SIDEBAR CONTROLS & DATA INGESTION
# ==========================================
with st.sidebar:
    st.markdown("### 🕸️ ResearchCollab")
    st.markdown("<p style='color: #94A3B8; font-size: 0.85rem; margin-top:-10px;'>Network Analysis & Graph Theory</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Data Source Selection
    st.markdown("#### 📡 Ingestion Source")
    data_source_option = st.radio(
        "Ingestion Source",
        options=["Offline Benchmark Sample", "Live OpenAlex API"],
        index=0 if st.session_state["data_source_mode"] == "sample" else 1,
        label_visibility="collapsed"
    )

    if data_source_option == "Offline Benchmark Sample":
        if st.session_state["dataset"].get("is_live", False):
            st.session_state["dataset"] = load_offline_dataset(
                st.session_state.get("active_dataset_key", "gnn")
            )
        st.session_state["data_source_mode"] = "sample"
        chosen_sample = st.selectbox(
            "Select Benchmark Topic",
            options=list(OFFLINE_DATASET_MAP.keys()),
            format_func=lambda k: OFFLINE_DATASET_MAP[k]["name"],
            index=list(OFFLINE_DATASET_MAP.keys()).index(st.session_state.get("active_dataset_key", "gnn"))
        )
        if st.button("Load Benchmark Dataset", use_container_width=True):
            with st.spinner("Loading offline dataset..."):
                st.session_state["dataset"] = load_offline_dataset(chosen_sample)
                st.session_state["active_dataset_key"] = chosen_sample
                st.rerun()

    else:
        st.session_state["data_source_mode"] = "live"
        query_input = st.text_input("OpenAlex Research Query", value="Graph Neural Networks", placeholder="e.g. Quantum Computing, CRISPR...")
        paper_limit = st.slider("Max Works to Retrieve", min_value=15, max_value=80, value=35, step=5)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            fetch_btn = st.button("Query API", type="primary", use_container_width=True)
        with col_btn2:
            reset_btn = st.button("Reset", use_container_width=True)

        if reset_btn:
            st.session_state["dataset"] = load_offline_dataset("gnn")
            st.session_state["active_dataset_key"] = "gnn"
            st.session_state["data_source_mode"] = "sample"
            st.rerun()

        if fetch_btn:
            with st.spinner(f"Ingesting live metadata from OpenAlex for '{query_input}'..."):
                live_data, err = fetch_openalex_works(query_input, limit=paper_limit)
                if err:
                    st.error(f"⚠️ {err}")
                    st.info(
                        "The loaded dataset was not changed. Its source badge below "
                        "shows which data the dashboard is currently analyzing."
                    )
                elif live_data:
                    st.session_state["dataset"] = live_data
                    st.session_state["data_source_mode"] = "live"
                    st.success(f"Successfully ingested {len(live_data['publications'])} works!")
                    st.rerun()

    st.markdown("---")

    # Current Dataset Status Indicator
    is_live_current = st.session_state["dataset"].get("is_live", False)
    if is_live_current:
        st.markdown(
            f"<div class='badge-live'>● LIVE OPENALEX API DATA</div>"
            f"<div style='font-size:0.75rem; color:#94A3B8; margin-top:4px;'>{st.session_state['dataset'].get('source_note', '')}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='badge-sample'>● OFFLINE SAMPLE DATASET</div>"
            f"<div style='font-size:0.75rem; color:#94A3B8; margin-top:4px;'>{st.session_state['dataset'].get('source_note', '')}</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("#### 🧭 Analytical Views")
    page_selection = st.radio(
        "Navigation",
        [
            "Overview Dashboard",
            "Interactive Network Graph",
            "Graph & Centrality Analysis",
            "Community Detection",
            "Researcher Profiles",
            "Data Tables & Exports",
            "Methodology & Viva Guide"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<p style='font-size: 0.72rem; color: #64748B;'>University Capstone Project • Graph Theory & Complex Networks Analysis</p>", unsafe_allow_html=True)


# ==========================================
# PREPARE RAW NETWORK & STATS
# ==========================================
dataset = st.session_state["dataset"]
publications_raw = dataset.get("publications", [])
authors_catalog = dataset.get("authors", {})
authorships_raw = dataset.get("authorships", [])

# Construct Full Base Graph
base_G = build_coauthorship_network(publications_raw, authors_catalog)

# Detect Communities on Base Graph
node_to_comm, comm_sets, algo_name, modularity_q = detect_communities(base_G)
community_summaries = get_community_summary(base_G, comm_sets, node_to_comm)

# Calculate Node Metrics on Base Graph
node_metrics_df = compute_node_centrality_metrics(base_G, node_to_comm)
global_metrics = compute_network_summary_metrics(base_G)


# ==========================================
# APP HEADER
# ==========================================
st.markdown(
    f"""
    <div class="app-header">
        <h1 class="app-title">
            <span style="color: #36D6C3;">ResearchCollab</span> Explorer
            <span style="font-size:0.8rem; font-weight:400; background:rgba(255,255,255,0.08); padding:4px 10px; border-radius:20px; color:#E2E8F0;">Graph Theory</span>
        </h1>
        <p class="app-subtitle">
            Research Collaboration Network Analysis and Co-authorship Topology Using Graph Theory • 
            Dataset: <b style="color:#FFFFFF;">{dataset.get('dataset_name', 'Research Corpus')}</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# PAGE 1: OVERVIEW DASHBOARD
# ==============================================================================
if page_selection == "Overview Dashboard":
    st.markdown("### 📊 Executive Overview Dashboard")
    st.markdown("<p style='color: #94A3B8;'>High-level topological, bibliometric, and structural properties of the co-authorship network.</p>", unsafe_allow_html=True)

    # Top Metric KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Researchers (|V|)</div>
                <div class="metric-value">{global_metrics['num_nodes']}</div>
                <div class="metric-sub">Network vertices</div>
            </div>""", unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Collaborations (|E|)</div>
                <div class="metric-value">{global_metrics['num_edges']}</div>
                <div class="metric-sub">Undirected edges</div>
            </div>""", unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Publications</div>
                <div class="metric-value">{len(publications_raw)}</div>
                <div class="metric-sub">In corpus</div>
            </div>""", unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Network Density</div>
                <div class="metric-value">{global_metrics['density']:.4f}</div>
                <div class="metric-sub">Completeness ratio</div>
            </div>""", unsafe_allow_html=True
        )
    with col5:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Modularity (Q)</div>
                <div class="metric-value" style="color: #9B8AFB;">{modularity_q:.3f}</div>
                <div class="metric-sub">{len(comm_sets)} communities</div>
            </div>""", unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Two Column Layout: Secondary Stats & Publication Trends
    row2_col1, row2_col2 = st.columns([1, 1])

    with row2_col1:
        st.markdown("#### 🔬 Network Macro Topology")
        
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            st.metric("Avg Degree (Co-authors)", f"{global_metrics['avg_degree']}")
            st.metric("Clustering Coefficient", f"{global_metrics['avg_clustering_coefficient']:.4f}")
            st.metric("Connected Components", f"{global_metrics['num_components']}")
        with c_sub2:
            st.metric("Largest Component Size", f"{global_metrics['largest_component_size']} ({global_metrics['largest_component_ratio']}%)")
            st.metric("Diameter (on LCC)", f"{global_metrics['diameter']}")
            st.metric("Avg Path Length (on LCC)", f"{global_metrics['avg_shortest_path_length']}")

        st.markdown(
            """
            <div class="info-card">
                <b>Topological Interpretation:</b> Real-world academic co-authorship networks typically exhibit 
                <i>Small-World</i> properties: relatively high clustering coefficient combined with low average shortest 
                path length, alongside a heavy-tailed degree distribution.
            </div>
            """, unsafe_allow_html=True
        )

    with row2_col2:
        st.markdown("#### 📅 Temporal Publication Distribution")
        fig_trends = create_publication_trends_chart(publications_raw)
        st.plotly_chart(fig_trends, use_container_width=True)

    st.markdown("---")

    # Top Central Researchers Preview & Horizontal Chart
    row3_col1, row3_col2 = st.columns([1, 1])
    with row3_col1:
        st.markdown("#### 🏆 Top Collaborators Ranking")
        fig_top_bar = create_top_collaborators_chart(node_metrics_df, metric_col="degree", top_n=8)
        st.plotly_chart(fig_top_bar, use_container_width=True)

    with row3_col2:
        st.markdown("#### ⭐ High-Prestige Scholars (PageRank Leaders)")
        top_pr = get_top_researchers(node_metrics_df, metric_col="pagerank", top_n=4)
        for _, r in top_pr.iterrows():
            comm_color = get_community_color(int(r['community_id']))
            st.markdown(
                f"""
                <div style="background:#141B2D; border:1px solid rgba(255,255,255,0.08); border-left:4px solid {comm_color}; border-radius:8px; padding:12px 14px; margin-bottom:10px;">
                    <div style="font-weight:700; color:#FFFFFF; font-size:1.05rem;">{r['name']}</div>
                    <div style="color:#94A3B8; font-size:0.8rem; margin-bottom:6px;">🏛 {r['institution']}</div>
                    <div style="display:flex; justify-content:space-between; font-size:0.82rem; border-top:1px solid rgba(255,255,255,0.05); padding-top:6px;">
                        <span>🤝 Degree: <b style="color:#36D6C3;">{r['degree']}</b></span>
                        <span>⭐ PageRank: <b style="color:#9B8AFB;">{r['pagerank']:.5f}</b></span>
                        <span>👥 Community #{int(r['community_id'])}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )


# ==============================================================================
# PAGE 2: INTERACTIVE NETWORK GRAPH
# ==============================================================================
elif page_selection == "Interactive Network Graph":
    st.markdown("### 🕸️ Interactive Co-Authorship Network Visualization")
    st.markdown("<p style='color: #94A3B8;'>Explore nodes, weighted collaboration edges, community structures, and ego-networks in 2D space.</p>", unsafe_allow_html=True)

    # Filter Controls Expander
    with st.expander("🛠️ Interactive Graph Filters & Controls", expanded=True):
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)

        with fcol1:
            years_in_data = [p["publication_year"] for p in publications_raw if p.get("publication_year", 0) > 0]
            min_y = min(years_in_data) if years_in_data else 2010
            max_y = max(years_in_data) if years_in_data else 2024
            if min_y == max_y:
                min_y = max_y - 1
            selected_years = st.slider(
                "Publication Year Range",
                min_value=min_y,
                max_value=max_y,
                value=(min_y, max_y),
                help="Publications without a known year are excluded by this filter."
            )

        with fcol2:
            max_edge_weight = max(
                (int(data.get("weight", 1)) for _, _, data in base_G.edges(data=True)),
                default=1
            )
            weight_limit = max(1, min(5, max_edge_weight))
            min_weight = st.slider(
                "Min Collaboration Strength (Shared Papers)",
                min_value=1,
                max_value=weight_limit,
                value=1
            )

        with fcol3:
            layout_type = st.selectbox("Force-Directed Layout", ["spring", "kamada_kawai", "circular"])
            size_by = st.selectbox("Node Size Metric", ["pagerank", "degree", "betweenness", "degree_centrality"])

        with fcol4:
            color_by = st.selectbox("Node Color By", ["community", "pagerank", "uniform"])
            remove_isolates = st.checkbox("Hide Isolated Researchers (Degree = 0)", value=False)

    # Search and Ego-Network Subgraph Selection
    scol1, scol2 = st.columns([2, 1])
    with scol1:
        researcher_ids = sorted(
            base_G.nodes(),
            key=lambda author_id: (
                str(base_G.nodes[author_id].get("name", author_id)).casefold(),
                str(author_id)
            )
        )
        target_author_id = st.selectbox(
            "🎯 Focus on Specific Researcher (Ego-Network)",
            options=[None] + researcher_ids,
            format_func=lambda author_id: (
                "None (View Full Network)"
                if author_id is None
                else f"{base_G.nodes[author_id].get('name', author_id)} ({author_id})"
            )
        )
    with scol2:
        search_query_term = st.text_input("🔍 Quick Search Name Filter", placeholder="Type name to highlight...")

    # Apply Filters to produce active graph
    active_G = filter_network(
        base_G,
        min_weight=min_weight,
        year_range=selected_years,
        target_researcher_id=target_author_id,
        ego_radius=1,
        remove_isolates=remove_isolates
    )

    # Calculate metrics for active graph
    active_node_to_comm, _, _, _ = detect_communities(active_G)
    active_metrics_df = compute_node_centrality_metrics(active_G, active_node_to_comm)

    # Render Graph Figure
    fig_network = create_network_figure(
        G=active_G,
        metrics_df=active_metrics_df,
        layout_type=layout_type,
        size_by=size_by,
        color_by=color_by,
        search_query=search_query_term,
        highlight_author_id=target_author_id
    )
    st.plotly_chart(fig_network, use_container_width=True, height=620)

    # Bottom Inspection Bar
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; background:#141B2D; padding:10px 16px; border-radius:8px; border:1px solid rgba(255,255,255,0.06); font-size:0.85rem; color:#94A3B8;">
            <span>Displaying <b>{active_G.number_of_nodes()}</b> researchers and <b>{active_G.number_of_edges()}</b> collaborations</span>
            <span>Zoom & Pan: Use mouse wheel or Plotly toolbar • Hover over any node for profile details</span>
        </div>
        """, unsafe_allow_html=True
    )


# ==============================================================================
# PAGE 3: GRAPH ANALYSIS & CENTRALITY
# ==============================================================================
elif page_selection == "Graph & Centrality Analysis":
    st.markdown("### 📈 Comprehensive Graph Analysis & Centrality Metrics")
    st.markdown("<p style='color: #94A3B8;'>Mathematical ranking of researchers across Degree, Betweenness, and PageRank.</p>", unsafe_allow_html=True)

    # Analytics Charts
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        fig_deg_dist = create_degree_distribution_chart(node_metrics_df)
        st.plotly_chart(fig_deg_dist, use_container_width=True)
    with gcol2:
        fig_scatter = create_centrality_scatter_chart(node_metrics_df)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # Centrality Rankings Table
    st.markdown("#### 🏆 Researcher Centrality Rankings")
    
    tcol1, tcol2 = st.columns([1, 2])
    with tcol1:
        rank_metric = st.selectbox(
            "Sort Ranking Table By",
            options=["pagerank", "betweenness_centrality", "degree", "degree_centrality", "clustering_coefficient"],
            format_func=lambda s: {
                "pagerank": "PageRank (Network Authority)",
                "betweenness_centrality": "Betweenness Centrality (Brokerage)",
                "degree": "Degree (Raw Co-authors)",
                "degree_centrality": "Normalized Degree Centrality",
                "clustering_coefficient": "Clustering Coefficient"
            }.get(s, s)
        )
    with tcol2:
        institution_filter = st.selectbox(
            "Filter by Institution",
            options=["All Institutions"] + sorted(list(set([r['institution'] for _, r in node_metrics_df.iterrows()])))
        )

    # Filtered DF for display
    display_df = node_metrics_df.copy()
    if institution_filter != "All Institutions":
        display_df = display_df[display_df["institution"] == institution_filter]

    display_df = display_df.sort_values(by=rank_metric, ascending=False).reset_index(drop=True)
    display_df.index = display_df.index + 1  # 1-indexed ranks

    columns_to_show = [
        "name", "institution", "degree", "weighted_degree",
        "degree_centrality", "betweenness_centrality", "pagerank",
        "clustering_coefficient", "community_id"
    ]

    st.dataframe(
        display_df[columns_to_show].rename(columns={
            "name": "Researcher",
            "institution": "Institution",
            "degree": "Degree",
            "weighted_degree": "Weighted Degree",
            "degree_centrality": "Deg Centrality",
            "betweenness_centrality": "Betweenness",
            "pagerank": "PageRank",
            "clustering_coefficient": "Clustering Coeff",
            "community_id": "Community #"
        }),
        use_container_width=True,
        height=380
    )


# ==============================================================================
# PAGE 4: COMMUNITY DETECTION
# ==============================================================================
elif page_selection == "Community Detection":
    st.markdown("### 👥 Community Detection & Research Clusters")
    st.markdown("<p style='color: #94A3B8;'>Algorithmic partitioning of the co-authorship graph into collaboration communities.</p>", unsafe_allow_html=True)

    # Algorithm Pipeline Banner
    st.markdown(
        f"""
        <div class="info-card">
            <b>Active Algorithm:</b> <span style="color:#36D6C3; font-weight:700;">{algo_name}</span><br>
            <b>Modularity Score Q:</b> <span style="color:#9B8AFB; font-weight:700;">{modularity_q:.4f}</span> 
            (Q &gt; 0.3 indicates strong community structure; theoretical max is 1.0).<br>
            <b>Detected Clusters:</b> {len(comm_sets)} distinct research communities.
        </div>
        """, unsafe_allow_html=True
    )

    ccol1, ccol2 = st.columns([1, 1])
    with ccol1:
        st.markdown("#### 🍩 Community Membership Distribution")
        fig_comm_donut = create_community_breakdown_chart(community_summaries)
        st.plotly_chart(fig_comm_donut, use_container_width=True)

    with ccol2:
        st.markdown("#### 🔬 Detailed Community Roster")
        for c in community_summaries[:8]:
            top_scholars = ", ".join(c["top_researchers"]) if c["top_researchers"] else "N/A"
            st.markdown(
                f"""
                <div style="background:#141B2D; border-left:4px solid {c['color']}; border-radius:6px; padding:10px 14px; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; color:#FFFFFF;">Community #{c['community_id']}</span>
                        <span style="color:{c['color']}; font-weight:700;">{c['size']} members ({c['percentage']}%)</span>
                    </div>
                    <div style="font-size:0.82rem; color:#94A3B8; margin-top:4px;">
                        <b>Key Scholars:</b> {top_scholars}
                    </div>
                    <div style="font-size:0.8rem; color:#64748B; margin-top:2px;">
                        <b>Affiliations:</b> {c['key_institutions']} • <b>Internal Collaborations:</b> {c['internal_edges']}
                    </div>
                </div>
                """, unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown("#### 🔍 Community Member Inspector")
    comm_ids = [c["community_id"] for c in community_summaries]
    if comm_ids:
        selected_comm_id = st.selectbox(
            "Inspect Community Members",
            options=comm_ids,
            format_func=lambda x: f"Community #{x} ({community_summaries[x]['size']} members)"
        )

        comm_members_df = node_metrics_df[node_metrics_df["community_id"] == selected_comm_id][
            ["name", "institution", "degree", "weighted_degree", "pagerank", "betweenness_centrality"]
        ].sort_values(by="pagerank", ascending=False).reset_index(drop=True)
        comm_members_df.index = comm_members_df.index + 1
        st.dataframe(comm_members_df, use_container_width=True)
    else:
        st.info("No researchers are available for community analysis in this dataset.")


# ==============================================================================
# PAGE 5: RESEARCHER PROFILES & EGO NETWORKS
# ==============================================================================
elif page_selection == "Researcher Profiles":
    st.markdown("### 🔬 Researcher Profiles & Direct Collaborators")
    st.markdown("<p style='color: #94A3B8;'>Detailed individual dossiers, collaborator breakdown, ego subgraph, and co-authorship history.</p>", unsafe_allow_html=True)

    profile_author_ids = sorted(
        base_G.nodes(),
        key=lambda author_id: (
            str(base_G.nodes[author_id].get("name", author_id)).casefold(),
            str(author_id)
        )
    )
    target_id = st.selectbox(
        "Select Researcher Dossier",
        options=[None] + profile_author_ids,
        format_func=lambda author_id: (
            "No researchers available"
            if author_id is None
            else f"{base_G.nodes[author_id].get('name', author_id)} ({author_id})"
        )
    )
    if not profile_author_ids:
        st.info("No researcher records are available in this dataset.")

    if target_id and base_G.has_node(target_id):
        node_attr = base_G.nodes[target_id]
        selected_name = node_attr.get("name", target_id)
        m_row = node_metrics_df[node_metrics_df["author_id"] == target_id].iloc[0]

        # Profile KPIs
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        with pcol1:
            st.metric("Total Direct Co-Authors", f"{base_G.degree(target_id)}")
        with pcol2:
            st.metric("PageRank Authority", f"{m_row['pagerank']:.5f}")
        with pcol3:
            st.metric("Betweenness Centrality", f"{m_row['betweenness_centrality']:.4f}")
        with pcol4:
            st.metric("Assigned Community", f"#{int(m_row['community_id'])}")

        st.markdown(
            f"""
            <div style="background:#141B2D; border:1px solid rgba(54,214,195,0.25); border-radius:10px; padding:16px; margin: 16px 0;">
                <h3 style="margin:0 0 6px 0; color:#FFFFFF;">{selected_name}</h3>
                <p style="margin:0; color:#36D6C3; font-size:0.95rem;">🏛 {node_attr.get('institution', 'Unknown Institution')}</p>
                <p style="margin:4px 0 0 0; color:#94A3B8; font-size:0.85rem;">Author ID: <code>{target_id}</code></p>
            </div>
            """, unsafe_allow_html=True
        )

        # Two columns: Ego graph and Collaborators table
        ecol1, ecol2 = st.columns([1, 1])

        with ecol1:
            st.markdown("#### 🕸️ 1-Hop Ego Subgraph")
            fig_ego = create_ego_figure(base_G, target_id, node_metrics_df)
            st.plotly_chart(fig_ego, use_container_width=True)

        with ecol2:
            st.markdown("#### 🤝 Direct Collaborators List")
            neighbors = list(base_G.neighbors(target_id))
            if neighbors:
                collab_records = []
                for nb in neighbors:
                    nb_data = base_G.nodes[nb]
                    edge_data = base_G[target_id][nb]
                    shared = edge_data.get("shared_publications", [])
                    sample_titles = "; ".join([p.get("title", "") for p in shared[:2]])
                    collab_records.append({
                        "Collaborator": nb_data.get("name", nb),
                        "Institution": nb_data.get("institution", "Unknown"),
                        "Shared Papers": edge_data.get("weight", 1),
                        "Sample Joint Papers": sample_titles
                    })
                
                collab_df = pd.DataFrame(collab_records).sort_values(by="Shared Papers", ascending=False).reset_index(drop=True)
                collab_df.index = collab_df.index + 1
                st.dataframe(collab_df, use_container_width=True, height=400)
            else:
                st.info("This researcher currently has no co-authorship edges in the active corpus (Solo author).")

        # Publications History in Corpus
        st.markdown("---")
        st.markdown(f"#### 📄 Publications by {selected_name} in Corpus")
        author_pubs = [
            p for p in publications_raw
            if any(a.get("id") == target_id for a in p.get("authors", []))
        ]
        if author_pubs:
            pub_items = []
            for p in author_pubs:
                pub_items.append({
                    "Title": p.get("title", "Untitled"),
                    "Year": p.get("publication_year", "N/A"),
                    "Citations": p.get("cited_by_count", 0),
                    "Topic": p.get("primary_topic", "General"),
                    "Co-Authors": ", ".join([a.get("name", "") for a in p.get("authors", []) if a.get("name") != selected_name])
                })
            st.dataframe(pd.DataFrame(pub_items), use_container_width=True)
        else:
            st.write("No direct publication details matched for this researcher.")


# ==============================================================================
# PAGE 6: DATA TABLES & EXPORTS
# ==============================================================================
elif page_selection == "Data Tables & Exports":
    st.markdown("### 📥 Corpus Tables & CSV Data Exports")
    st.markdown("<p style='color: #94A3B8;'>Download verified research data in CSV, JSON, and GraphML formats for viva assessment and external software.</p>", unsafe_allow_html=True)

    tab_nodes, tab_edges, tab_pubs, tab_authorships, tab_metrics = st.tabs([
        "👤 Researchers (Nodes)",
        "🔗 Collaborations (Edges)",
        "📄 Publications",
        "🧾 Authorships",
        "📊 Summary Metrics & GraphML"
    ])

    with tab_nodes:
        st.markdown("#### Researchers Data Table")
        st.dataframe(node_metrics_df, use_container_width=True, height=350)
        st.download_button(
            label="⬇️ Download Researchers CSV",
            data=dataframe_to_csv_bytes(node_metrics_df),
            file_name="researchers_metrics.csv",
            mime="text/csv",
            use_container_width=True
        )

    with tab_edges:
        st.markdown("#### Co-authorship Edges Table")
        edges_df = get_edge_dataframe(base_G)
        st.dataframe(edges_df, use_container_width=True, height=350)
        st.download_button(
            label="⬇️ Download Edges CSV",
            data=dataframe_to_csv_bytes(edges_df),
            file_name="coauthorship_edges.csv",
            mime="text/csv",
            use_container_width=True
        )

    with tab_pubs:
        st.markdown("#### Publications Table")
        pubs_df = generate_publications_dataframe(publications_raw)
        st.dataframe(pubs_df, use_container_width=True, height=350)
        st.download_button(
            label="⬇️ Download Publications CSV",
            data=dataframe_to_csv_bytes(pubs_df),
            file_name="publications_corpus.csv",
            mime="text/csv",
            use_container_width=True
        )

    with tab_authorships:
        st.markdown("#### Publication-to-Researcher Authorships")
        authorships_df = generate_authorships_dataframe(authorships_raw)
        st.dataframe(authorships_df, use_container_width=True, height=350)
        st.download_button(
            label="⬇️ Download Authorships CSV",
            data=dataframe_to_csv_bytes(authorships_df),
            file_name="authorships.csv",
            mime="text/csv",
            use_container_width=True
        )

    with tab_metrics:
        st.markdown("#### Network Global Metrics Table")
        summary_df = generate_summary_metrics_dataframe(global_metrics)
        st.dataframe(summary_df, use_container_width=True, height=280)
        
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.download_button(
                label="⬇️ Download Summary Metrics CSV",
                data=dataframe_to_csv_bytes(summary_df),
                file_name="network_summary_metrics.csv",
                mime="text/csv",
                use_container_width=True
            )
        with dcol2:
            graphml_str = generate_graphml_export(base_G)
            st.download_button(
                label="⬇️ Download GraphML (Gephi / Cytoscape)",
                data=graphml_str.encode("utf-8"),
                file_name="collaboration_network.graphml",
                mime="application/xml",
                use_container_width=True
            )


# ==============================================================================
# PAGE 7: METHODOLOGY, LIMITATIONS & VIVA GUIDE
# ==============================================================================
elif page_selection == "Methodology & Viva Guide":
    st.markdown("### 📚 Methodology, Theoretical Foundations & Viva Defense Guide")
    st.markdown("<p style='color: #94A3B8;'>Detailed technical documentation, mathematical definitions, algorithms, and examiner viva Q&A.</p>", unsafe_allow_html=True)

    with st.expander("📐 Mathematical Definitions & Graph Theory Formulations", expanded=True):
        st.markdown(
            r"""
            #### 1. Graph Model Representation
            Let the collaboration network be defined as an undirected weighted simple graph:
            $$G = (V, E, W)$$
            Where:
            - $V$: Set of unique researchers / authors $|V| = n$.
            - $E$: Set of co-authorship relationships $|E| = m$, where $(u, v) \in E \iff$ authors $u$ and $v$ co-authored $\ge 1$ publication.
            - $W$: Weight function $w(u, v) \in \mathbb{N}_{\ge 1}$ denoting the exact count of shared publications.

            #### 2. Network Density
            Measures the proportion of possible connections that actually exist:
            $$\text{Density}(G) = \frac{2 \cdot |E|}{|V|(|V| - 1)}$$

            #### 3. Degree Centrality
            Normalized degree indicating immediate connectivity:
            $$C_D(v) = \frac{\text{deg}(v)}{|V| - 1}$$

            #### 4. Betweenness Centrality
            Quantifies the extent to which researcher $v$ acts as an intermediary / bridge along shortest paths:
            $$C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$
            Where $\sigma_{st}$ is the total number of shortest paths from node $s$ to $t$, and $\sigma_{st}(v)$ is the number of those paths passing through $v$.

            #### 5. PageRank Centrality
            Measures recursive network authority derived from collaborating with other well-connected researchers:
            $$PR(v) = \frac{1 - d}{|V|} + d \sum_{u \in N(v)} \frac{w(u, v) \cdot PR(u)}{\sum_{z \in N(u)} w(u, z)}$$
            Where $d = 0.85$ is the damping factor (representing teleportation probability).

            #### 6. Newman-Girvan Modularity ($Q$)
            Measures the quality of partition into communities compared to a random null-model:
            $$Q = \frac{1}{2m} \sum_{i, j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$
            """
        )

    with st.expander("🔄 4-Stage Community Detection Fallback Architecture", expanded=True):
        st.markdown(
            """
            To guarantee zero unhandled runtime crashes across heterogeneous graphs, ResearchCollab Explorer 
            employs a sequential 4-tier fallback pipeline:
            1. **Primary: Louvain Modularity Optimization (`nx.community.louvain_communities`)**:
               Fast heuristic modularity maximization with iterative local edge aggregations.
            2. **Fallback 1: Greedy Modularity (`nx.community.greedy_modularity_communities`)**:
               Hierarchical agglomeration (Clauset-Newman-Moore) if Louvain encounters optimization hurdles.
            3. **Fallback 2: Asynchronous Label Propagation (`nx.community.label_propagation_communities`)**:
               Near-linear $O(m)$ dissemination suitable for massive or sparse components.
            4. **Fallback 3: Connected Components (`nx.connected_components`)**:
               Deterministic topological reachability fallback guaranteeing every node belongs to a defined partition.
            """
        )

    with st.expander("🎓 University Viva / Defense Examination Questions & Model Answers", expanded=True):
        st.markdown(
            """
            **Q1: Why is an undirected graph chosen for co-authorship rather than a directed graph?**  
            *Answer:* Co-authorship is intrinsically symmetric. If Researcher A co-authors a paper with Researcher B, Researcher B by definition co-authors the paper with Researcher A. While citation networks are naturally directed (Paper A cites Paper B $\\neq$ Paper B cites Paper A), co-authorship represents mutual participation in a joint scientific endeavor.

            **Q2: How does edge weighting alter the interpretation of the network?**  
            *Answer:* Unweighted graphs treat a one-time casual co-authorship identically to a decade-long partnership of 20 joint papers. By assigning $w(u, v) = \\text{count}(\\text{shared papers})$, we account for tie strength (Granovetter's theory). Highly weighted edges signify institutional synergy, shared laboratories, or long-standing advisor-advisee relationships.

            **Q3: What are the main limitations of co-authorship network analysis?**  
            *Answer:*
            1. *Author Name Ambiguity:* Different authors with common names (e.g., 'J. Smith') or single authors with name variations require canonical author IDs (which OpenAlex resolves via stable OpenAlex IDs).
            2. *Author Contribution Disparity:* Co-authorship graphs treat all co-authors on a paper equally unless fractional counting is used; first authors and senior PI authors are connected with identical edge weights to middle authors.
            3. *Disciplinary Publishing Conventions:* High-energy physics or clinical trials routinely feature 50+ or 1,000+ authors on a single paper (hyper-authorship), which creates large artificial dense cliques.

            **Q4: Why does Betweenness Centrality differ from PageRank in interpreting researcher prominence?**  
            *Answer:* PageRank identifies *authoritative hubs*—researchers who collaborate with other highly-cited, central figures. Betweenness Centrality identifies *brokers and boundary spanners*—researchers who connect two otherwise isolated scientific disciplines or research groups. A researcher can have moderate degree but very high betweenness if they form the sole bridge between two large departments.
            """
        )

    with st.expander("⚠️ System Limitations & Ethical Considerations", expanded=False):
        st.markdown(
            """
            - **API Rate Limits:** OpenAlex limits anonymous requests; the polite pool (`mailto:` header) is used to maintain polite throughput.
            - **Corpus Coverage:** Analysis reflects the ingested set of publications (retrieved via query or offline benchmark sample) and does not represent an author's complete lifelong bibliography unless a dedicated author entity query is executed.
            - **Offline Mode Integrity:** The application explicitly labels offline benchmark data as sample datasets, maintaining research rigor without confusing demo data with live queries.
            """
        )
