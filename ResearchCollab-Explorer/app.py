"""ResearchCollab Explorer - Streamlit application.

Run with:   streamlit run app.py

How the app is organised
------------------------
1. Sidebar  : navigation, data source, advanced settings, search & filters.
2. Loading  : the edge list is read and cleaned (src/data_loader.py) and cached.
3. Analysis : graph, metrics and Louvain communities are computed ONCE per dataset/setting
              and cached (st.cache_data), so moving a slider never recomputes them.
4. Pages    : each page is a small function that only displays cached results.
"""

from __future__ import annotations

import html
import re
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import networkx as nx
import pandas as pd
import streamlit as st

from src import data_loader as dl
from src import exports as ex
from src import graph_analysis as ga
from src import visualization as viz

APP_TITLE = "ResearchCollab Explorer"
APP_SUBTITLE = "Explore the connections that shape scientific discovery."

PAGE_OVERVIEW = "📊 Overview"
PAGE_NETWORK = "🕸️ Collaboration network"
PAGE_COMMUNITIES = "🧩 Communities"
PAGE_ANALYTICS = "👤 Researcher analytics"
PAGE_EXPORT = "⬇️ Export & download"
PAGE_ABOUT = "📘 About & methodology"
PAGES = [PAGE_OVERVIEW, PAGE_NETWORK, PAGE_COMMUNITIES, PAGE_ANALYTICS, PAGE_EXPORT, PAGE_ABOUT]

SOURCE_LOCAL = "Local file"
SOURCE_UPLOAD = "Upload file"
SOURCE_SAMPLE = "Bundled sample"

MAX_NETWORK_NODES = 2000  # hard cap so the browser never has to draw a huge graph

st.set_page_config(
    page_title=APP_TITLE, page_icon="🕸️", layout="wide", initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.rc-hero {background: linear-gradient(120deg,#4F46E5 0%,#7C3AED 55%,#0EA5E9 100%);
          color:#fff; padding:1.5rem 2rem; border-radius:18px; margin-bottom:1rem;}
.rc-hero h1 {margin:0; font-size:2rem; color:#fff; letter-spacing:-0.01em;}
.rc-hero p {margin:.3rem 0 0; opacity:.93; font-size:1.05rem;}
.rc-pill {display:inline-block; background:#EEF2FF; color:#3730A3; border-radius:999px;
          padding:.2rem .8rem; font-size:.82rem; margin:0 .4rem .6rem 0; font-weight:600;}
.rc-card {background:#fff; border:1px solid #E2E8F0; border-radius:14px;
          padding:.9rem 1.1rem; box-shadow:0 1px 2px rgba(15,23,42,.05); margin-bottom:.6rem;}
.rc-card .label {color:#64748B; font-size:.74rem; text-transform:uppercase; letter-spacing:.06em;}
.rc-card .value {color:#0F172A; font-size:1.8rem; font-weight:700; line-height:1.25;}
.rc-card .hint {color:#94A3B8; font-size:.75rem;}
.rc-note {background:#F1F5F9; border-left:4px solid #4F46E5; border-radius:8px;
          padding:.7rem 1rem; color:#334155; font-size:.92rem; margin:.4rem 0 .8rem 0;}
section[data-testid="stSidebar"] {border-right:1px solid #E2E8F0;}
h2, h3 {letter-spacing:-0.01em;}
</style>
"""


# --------------------------------------------------------------------------------------
# Small UI helpers
# --------------------------------------------------------------------------------------
def fmt_int(value) -> str:
    return f"{int(value):,}"


def fmt_float(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def metric_card(label: str, value: str, hint: str = "") -> None:
    st.markdown(
        f'<div class="rc-card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>'
        f'<div class="hint">{html.escape(hint)}&nbsp;</div></div>',
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f'<div class="rc-note">{text}</div>', unsafe_allow_html=True)


def render_html(page_html: str, height: int) -> None:
    """Embed a self-contained HTML page (the PyVis network) in the app.

    ``st.iframe`` is the current API; older Streamlit versions only have
    ``components.html``. The HTML only contains numbers produced by this app (validated
    integer IDs and metrics), never free text typed by a user.
    """
    if hasattr(st, "iframe"):
        st.iframe(page_html, height=height)
    else:  # pragma: no cover - older Streamlit
        import streamlit.components.v1 as components

        components.html(page_html, height=height, scrolling=False)


def hero() -> None:
    st.markdown(
        f'<div class="rc-hero"><h1>🕸️ {APP_TITLE}</h1><p>{APP_SUBTITLE}</p></div>',
        unsafe_allow_html=True,
    )


METRIC_COLUMN_CONFIG = {
    "researcher_id": st.column_config.NumberColumn("Researcher ID", format="%d"),
    "degree": st.column_config.NumberColumn("Degree", format="%d"),
    "degree_centrality": st.column_config.NumberColumn("Degree centrality", format="%.5f"),
    "betweenness_centrality": st.column_config.NumberColumn("Betweenness", format="%.5f"),
    "closeness_centrality": st.column_config.NumberColumn("Closeness", format="%.4f"),
    "clustering_coefficient": st.column_config.NumberColumn("Clustering", format="%.3f"),
    "community_id": st.column_config.NumberColumn("Community", format="%d"),
}

METRIC_LABELS = {
    "degree": "Degree",
    "degree_centrality": "Degree centrality",
    "betweenness_centrality": "Betweenness centrality",
    "closeness_centrality": "Closeness centrality",
    "clustering_coefficient": "Clustering coefficient",
}

METRIC_GUIDE = [
    ("Degree",
     "The number of distinct co-authors a researcher has in this dataset.",
     "It counts collaborators, not papers or impact. One large multi-author paper can give "
     "a researcher many collaborators at once."),
    ("Degree centrality",
     "degree / (n − 1), i.e. degree rescaled to the range 0-1.",
     "Lets you compare networks of different sizes. It carries every limitation of degree."),
    ("Betweenness centrality",
     "The share of shortest paths between other pairs of researchers that pass through this "
     "researcher (normalised to 0-1).",
     "Highlights structural 'bridges' between groups. On large networks it is estimated from a "
     "sample of source nodes (see Advanced settings), so small values are noisy. A bridge "
     "position is not evidence of better research."),
    ("Closeness centrality",
     "How short the average shortest path is to the researchers they can reach "
     "(Wasserman-Faust scaling, so disconnected graphs are handled).",
     "Depends strongly on the size of the component a researcher belongs to. On large networks "
     "it is estimated from the same sample of pivot nodes as betweenness."),
    ("Clustering coefficient",
     "Among a researcher's collaborators, the fraction of pairs that also collaborated.",
     "Multi-author papers create fully connected groups, which inflates clustering. "
     "Researchers with fewer than two collaborators score 0 by definition."),
    ("Community ID",
     "The Louvain community the researcher was assigned to (0 = the largest community).",
     "A community is a structurally close group in the collaboration network, not necessarily "
     "an official research group. Results can change with the seed or resolution."),
]


# --------------------------------------------------------------------------------------
# Cached computations
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading and cleaning the edge list…")
def cached_load_path(path_str: str, mtime: float):
    """Load + clean a file. ``mtime`` is part of the cache key so edits to the file reload."""
    edges, report = dl.load_edges_from_path(path_str)
    return edges, report.to_dict()


@st.cache_data(show_spinner="Reading and cleaning the uploaded file…")
def cached_load_bytes(data: bytes, filename: str):
    edges, report = dl.load_edges_from_bytes(data, filename)
    return edges, report.to_dict()


@st.cache_resource(show_spinner=False)
def cached_graph(edges: pd.DataFrame) -> nx.Graph:
    """The NetworkX graph, shared between reruns. It is treated as READ-ONLY."""
    return ga.build_graph(edges)


@st.cache_data(show_spinner="Computing network metrics and communities (cached afterwards)…")
def cached_analysis(edges: pd.DataFrame, sample_size: int, seed: int, closeness: bool):
    """All the expensive work. Runs once per (dataset, settings) combination."""
    graph = ga.build_graph(edges)
    communities = ga.detect_communities(graph, seed=seed)
    metrics = ga.compute_node_metrics(
        graph, communities, sample_size=sample_size or None, seed=seed,
        compute_closeness=closeness,
    )
    return {
        "communities": communities,
        "metrics": metrics,
        "summary": ga.network_summary(graph, communities),
        "degree_dist": ga.degree_distribution(graph),
        "comm_summary": ga.community_summary(communities),
    }


# --------------------------------------------------------------------------------------
# State containers
# --------------------------------------------------------------------------------------
@dataclass
class Filters:
    search_text: str = ""
    selected: Optional[int] = None
    community_id: Optional[int] = None
    min_degree: int = 1
    top_n: int = MAX_NETWORK_NODES
    neighbourhood_only: bool = False


@dataclass
class Context:
    label: str
    edges: pd.DataFrame
    cleaning: dict
    graph: nx.Graph
    communities: Dict[int, int]
    metrics: pd.DataFrame
    summary: dict
    degree_dist: pd.DataFrame
    comm_summary: pd.DataFrame
    settings: dict
    dkey: str
    filters: Filters

    def view(self) -> nx.Graph:
        """The graph after applying the sidebar filters."""
        f = self.filters
        return ga.filter_graph(
            self.graph, self.communities,
            min_degree=f.min_degree, community_id=f.community_id, top_n=f.top_n,
            selected=f.selected, neighborhood_only=f.neighbourhood_only and f.selected is not None,
        )


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
def sidebar_navigation() -> str:
    st.sidebar.markdown("## 🕸️ ResearchCollab")
    st.sidebar.caption("Research collaboration network analytics")
    return st.sidebar.radio("Navigate", PAGES, key="page", label_visibility="collapsed")


def _use_sample() -> None:
    st.session_state["source_mode"] = SOURCE_SAMPLE


def _download_snap() -> None:
    """Button callback (widget state may only be changed in callbacks, not after creation)."""
    try:
        dl.download_snap_dataset(dl.DATA_DIR)
        st.session_state["dataset_path"] = f"data/{dl.SNAP_DATASET_FILENAME}"
        st.session_state["dl_error"] = ""
    except dl.DatasetError as exc:
        st.session_state["dl_error"] = str(exc)


def sidebar_data_source():
    """Choose and load the dataset. Returns (edges, cleaning, label) or None."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Data source")
    if "source_mode" not in st.session_state:
        st.session_state["source_mode"] = (
            SOURCE_LOCAL if dl.locate_default_dataset() else SOURCE_SAMPLE
        )
    mode = st.sidebar.radio(
        "Data source", [SOURCE_LOCAL, SOURCE_UPLOAD, SOURCE_SAMPLE],
        key="source_mode", label_visibility="collapsed",
    )

    try:
        if mode == SOURCE_SAMPLE:
            st.sidebar.caption("Small synthetic network for testing (30 researchers).")
            edges, cleaning = cached_load_path(str(dl.SAMPLE_PATH), dl.SAMPLE_PATH.stat().st_mtime)
            return edges, cleaning, dl.SAMPLE_PATH.name

        if mode == SOURCE_UPLOAD:
            uploaded = st.sidebar.file_uploader(
                "Edge list (.txt, .csv or .gz)", type=["txt", "csv", "tsv", "gz", "edges"],
                help="One edge per line: id1 id2 (space, tab or comma separated).",
            )
            if uploaded is None:
                st.sidebar.info("Upload an edge-list file to begin.")
                return None
            edges, cleaning = cached_load_bytes(uploaded.getvalue(), uploaded.name)
            return edges, cleaning, uploaded.name

        # --- local file ---------------------------------------------------------------
        st.session_state.setdefault("dataset_path", dl.DEFAULT_DATASET_PATH)
        path_text = st.sidebar.text_input(
            "Dataset path", key="dataset_path",
            help="Relative to the project folder. Accepts .txt.gz, .txt or .csv.",
        )
        path = dl.resolve_dataset_path(path_text)
        if (path is None or not path.is_file()) and path_text.strip() == dl.DEFAULT_DATASET_PATH:
            path = dl.find_default_dataset() or path  # e.g. an extracted ca-GrQc.txt
        st.sidebar.button("⬇ Download SNAP dataset", on_click=_download_snap,
                          help="Fetches ca-GrQc.txt.gz from snap.stanford.edu into the data folder.")
        if st.session_state.get("dl_error"):
            st.sidebar.error(st.session_state["dl_error"])
        if path is None or not path.is_file():
            st.sidebar.warning("Dataset file not found.")
            return None
        edges, cleaning = cached_load_path(str(path), path.stat().st_mtime)
        return edges, cleaning, path.name
    except dl.DatasetError as exc:
        st.sidebar.error(str(exc))
        return None


def empty_state(page: str) -> None:
    st.markdown("### No dataset loaded yet")
    note(
        "ResearchCollab Explorer needs an edge list to analyse. You have three options:<br>"
        "1. Put <code>ca-GrQc.txt.gz</code> in the project's <code>data</code> folder "
        f"(download: {dl.SNAP_DATASET_URL}) or click <b>Download SNAP dataset</b> in the sidebar.<br>"
        "2. Choose <b>Upload file</b> in the sidebar and upload any edge list.<br>"
        "3. Choose <b>Bundled sample</b> to try the app straight away."
    )
    st.button("Use the bundled sample now", on_click=_use_sample, type="primary")


def sidebar_advanced(n_nodes: int, dkey: str):
    with st.sidebar.expander("Advanced settings"):
        default_k = ga.default_sample_size(n_nodes) or 0
        k = st.number_input(
            "Sample size for betweenness & closeness (0 = exact)", min_value=0,
            max_value=max(n_nodes, 1), value=int(min(default_k, n_nodes)),
            step=100 if n_nodes >= 200 else 1, key=f"sample_k::{dkey}",
            help="Both measures need a shortest-path search from every node (O(n·m)). With k "
                 "sampled source nodes they become fast, unbiased estimates. "
                 "0 = exact, which can take minutes above a few thousand researchers.",
        )
        if (k == 0 or k >= n_nodes) and n_nodes > ga.EXACT_METRICS_MAX_NODES:
            st.caption("⚠️ Exact mode on a network this size can take several minutes.")
        seed = st.number_input("Random seed", min_value=0, max_value=10**6,
                               value=ga.DEFAULT_SEED, step=1, key="seed",
                               help="Makes Louvain communities and sampling reproducible.")
        closeness = st.checkbox(
            "Compute closeness centrality", value=True, key="closeness",
            help="Untick to skip closeness entirely (the column is then left empty).",
        )
    return int(k), int(seed), bool(closeness)


def parse_researcher_id(text: str, graph: nx.Graph):
    """Return (node or None, status) where status is empty / invalid / missing / ok."""
    text = (text or "").strip()
    if not text:
        return None, "empty"
    if not re.fullmatch(r"[0-9]{1,18}", text):
        return None, "invalid"
    node = int(text)
    return (node, "ok") if node in graph else (None, "missing")


def sidebar_filters(summary: dict, comm_summary: pd.DataFrame, graph: nx.Graph, dkey: str) -> Filters:
    n_total = summary["nodes"]
    top_upper = min(n_total, MAX_NETWORK_NODES)
    top_default = min(300, top_upper)
    sizes = dict(zip(comm_summary["community_id"], comm_summary["size"]))
    k_search, k_comm, k_deg, k_top, k_neigh = (
        f"f_search::{dkey}", f"f_comm::{dkey}", f"f_deg::{dkey}", f"f_top::{dkey}", f"f_neigh::{dkey}"
    )

    def reset() -> None:
        st.session_state[k_search] = ""
        st.session_state[k_comm] = None
        st.session_state[k_deg] = 1
        st.session_state[k_top] = top_default
        st.session_state[k_neigh] = False

    # Initial values live in session_state (set once). The widgets below therefore get no
    # separate default, which avoids Streamlit's "default value + Session State" warning
    # and lets the Reset button simply write the defaults back.
    st.session_state.setdefault(k_search, "")
    st.session_state.setdefault(k_comm, None)
    st.session_state.setdefault(k_deg, 1)
    st.session_state.setdefault(k_top, top_default)
    st.session_state.setdefault(k_neigh, False)

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Search & filters")
    st.sidebar.caption("Applied to the network view, the optional analytics filter and exports.")

    search = st.sidebar.text_input("Search researcher ID", key=k_search, placeholder="e.g. 3466")
    selected, status = parse_researcher_id(search, graph)
    if status == "invalid":
        st.sidebar.warning("Researcher IDs are whole numbers.")
    elif status == "missing":
        st.sidebar.warning("That researcher ID is not in the loaded network.")
    elif status == "ok":
        st.sidebar.success(f"Selected researcher {selected}")

    community_options = [None] + comm_summary["community_id"].tolist()
    community = st.sidebar.selectbox(
        "Community", community_options, key=k_comm,  # state starts as None = "All communities"
        format_func=lambda c: "All communities" if c is None else f"Community {c} ({sizes[c]} researchers)",
    )

    max_degree = summary["max_degree"]
    if max_degree >= 2:
        min_degree = st.sidebar.slider("Minimum degree", 1, int(max_degree), key=k_deg)
    else:
        min_degree = 1

    if top_upper >= 10:
        top_n = st.sidebar.slider(
            "Top N nodes (by degree)", 10, int(top_upper), step=5, key=k_top,
            help=f"At most {MAX_NETWORK_NODES:,} nodes are ever drawn, to keep the browser fast.",
        )
    else:
        top_n = max(top_upper, 1)

    neighbourhood = st.sidebar.checkbox(
        "Show only selected researcher's neighbourhood", key=k_neigh, disabled=selected is None,
    )
    st.sidebar.button("Reset filters", on_click=reset)

    return Filters(
        search_text=search.strip(), selected=selected, community_id=community,
        min_degree=int(min_degree), top_n=int(top_n),
        neighbourhood_only=bool(neighbourhood and selected is not None),
    )


# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------
def dataset_banner(ctx: Context) -> None:
    s = ctx.summary
    st.markdown(
        f'<span class="rc-pill">Dataset: {html.escape(ctx.label)}</span>'
        f'<span class="rc-pill">{fmt_int(s["nodes"])} researchers</span>'
        f'<span class="rc-pill">{fmt_int(s["edges"])} collaborations</span>',
        unsafe_allow_html=True,
    )


def page_overview(ctx: Context) -> None:
    s = ctx.summary
    dataset_banner(ctx)
    st.subheader("Network overview")

    row1 = st.columns(3)
    with row1[0]:
        metric_card("Total researchers (nodes)", fmt_int(s["nodes"]), "unique researcher IDs")
    with row1[1]:
        metric_card("Total collaborations (edges)", fmt_int(s["edges"]), "unique undirected pairs")
    with row1[2]:
        metric_card("Average degree", fmt_float(s["average_degree"], 2), "2 × edges ÷ nodes")
    row2 = st.columns(3)
    with row2[0]:
        metric_card("Connected components", fmt_int(s["connected_components"]),
                    f"largest holds {s['largest_component_fraction']:.1%} of researchers")
    with row2[1]:
        metric_card("Average clustering coefficient", fmt_float(s["average_clustering"], 4),
                    "mean over all researchers")
    with row2[2]:
        metric_card("Network density", fmt_float(s["density"], 6), "edges ÷ possible edges")

    if s["communities"] is not None:
        mod = fmt_float(s["modularity"], 3) if s["modularity"] is not None else "n/a"
        st.caption(f"Louvain found **{s['communities']:,}** communities (modularity {mod}). "
                   f"Median degree {s['median_degree']:.0f}, maximum degree {s['max_degree']:,}.")

    with st.expander("Data cleaning report"):
        c = ctx.cleaning
        report = pd.DataFrame(
            [
                ("Lines read", c["total_lines"]),
                ("Comment / blank lines ignored", c["comment_or_blank_lines"]),
                ("Header lines skipped", c["header_lines"]),
                ("Malformed rows removed", c["malformed_rows"]),
                ("Rows with invalid IDs removed", c["invalid_id_rows"]),
                ("Valid edge rows", c["valid_rows"]),
                ("Self-loops removed", c["self_loops_removed"]),
                ("Duplicate edges removed", c["duplicate_edges_removed"]),
                ("Unique edges kept", c["unique_edges"]),
            ],
            columns=["Step", "Count"],
        )
        st.dataframe(report, hide_index=True)
        st.caption("The SNAP file stores every collaboration twice (once per direction); "
                   "those mirrored rows are what the duplicate count mostly reflects.")

    st.markdown("### Charts")
    left, right = st.columns(2)
    with left:
        log_log = st.toggle("Log-log scale", value=True, key="deg_loglog")
        st.plotly_chart(viz.degree_distribution_chart(ctx.degree_dist, log_log), key="ov_degdist")
    with right:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.plotly_chart(viz.top_metric_chart(ctx.metrics, "degree", 10), key="ov_top10")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(viz.community_size_histogram(ctx.comm_summary), key="ov_commhist")
    with right:
        n = s["nodes"]
        upper = min(n, 1000)
        if upper >= 20:
            count = st.slider("Researchers to plot (highest degree first)", 10, upper,
                              min(200, upper), key="ov_scatter_n")
        else:
            count = upper
        subset = ctx.metrics.sort_values(["degree", "researcher_id"], ascending=[False, True]).head(count)
        st.plotly_chart(
            viz.degree_vs_betweenness_chart(
                subset, f"Degree vs betweenness (top {len(subset):,} by degree)"),
            key="ov_scatter",
        )


def researcher_panel(ctx: Context, node: int) -> None:
    row = ctx.metrics[ctx.metrics["researcher_id"] == node].iloc[0]
    st.markdown(f"#### Researcher {node}")
    cols = st.columns(6)
    cols[0].metric("Degree", fmt_int(row["degree"]))
    cols[1].metric("Degree centrality", fmt_float(row["degree_centrality"], 5))
    cols[2].metric("Betweenness", fmt_float(row["betweenness_centrality"], 5))
    cols[3].metric("Closeness", fmt_float(row["closeness_centrality"], 4))
    cols[4].metric("Clustering", fmt_float(row["clustering_coefficient"], 3))
    cols[5].metric("Community", fmt_int(row["community_id"]))
    neighbours = ga.neighbors_table(ctx.graph, ctx.metrics, node)
    st.markdown(f"**Immediate collaborators ({len(neighbours):,})**")
    st.dataframe(neighbours, hide_index=True, column_config=METRIC_COLUMN_CONFIG)


def page_network(ctx: Context) -> None:
    f = ctx.filters
    dataset_banner(ctx)
    st.subheader("Interactive collaboration network")
    note("<b>Scroll</b> to zoom · <b>drag the background</b> to pan · <b>drag a node</b> to move it · "
         "<b>hover</b> for details · <b>click</b> a node to highlight its connections. "
         "Colour = Louvain community, size = degree.")

    view = ctx.view()
    if view.number_of_nodes() == 0:
        st.warning("No researchers match the current filters. Lower the minimum degree, pick "
                   "another community or press **Reset filters** in the sidebar.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Researchers shown",
              f"{fmt_int(view.number_of_nodes())} of {fmt_int(ctx.summary['nodes'])}")
    c2.metric("Collaborations shown",
              f"{fmt_int(view.number_of_edges())} of {fmt_int(ctx.summary['edges'])}")
    with c3:
        physics = st.toggle("Physics layout", value=True, key="net_physics",
                            help="Turn off to freeze the layout after it settles.")
    with c4:
        height = st.slider("Height (px)", 500, 1000, 720, step=20, key="net_height")

    if view.number_of_nodes() > 800:
        st.info("Large views can feel slow. Reduce *Top N nodes* or raise *Minimum degree* for a smoother graph.")
    if f.selected is None and view.number_of_nodes() < ctx.summary["nodes"]:
        st.caption("Showing the highest-degree researchers that match your filters. "
                   "Adjust *Top N nodes* in the sidebar to show more or fewer.")

    html_page = viz.build_network_html(view, ctx.communities, ctx.metrics,
                                       selected=f.selected, height_px=height, physics=physics)
    render_html(html_page, height + 20)

    if f.selected is not None:
        st.markdown("---")
        researcher_panel(ctx, f.selected)
    else:
        st.caption("Tip: type a researcher ID in the sidebar to highlight that researcher and their collaborators.")


def page_communities(ctx: Context) -> None:
    dataset_banner(ctx)
    st.subheader("Community detection (Louvain)")
    cs = ctx.comm_summary
    if cs.empty:
        st.warning("No communities to show because the network is empty.")
        return
    s = ctx.summary

    cols = st.columns(4)
    with cols[0]:
        metric_card("Communities detected", fmt_int(s["communities"]), "Louvain, fixed seed")
    with cols[1]:
        metric_card("Largest community", fmt_int(cs["size"].max()), "researchers")
    with cols[2]:
        metric_card("Median community size", f"{cs['size'].median():.0f}", "researchers")
    with cols[3]:
        metric_card("Modularity", fmt_float(s["modularity"], 3) if s["modularity"] is not None else "n/a",
                    "higher = clearer structure")

    note("A detected community is a group of researchers who are <b>structurally close</b> in this "
         "collaboration network (they collaborate with each other more than expected by chance). "
         "It is <b>not</b> necessarily an officially defined research group, lab or topic.")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(viz.community_size_chart(cs), key="cm_sizes")
    with right:
        st.plotly_chart(viz.community_size_histogram(cs), key="cm_hist")

    st.markdown("### Explore one community")
    sizes = dict(zip(cs["community_id"], cs["size"]))
    options = cs["community_id"].tolist()
    wanted = ctx.filters.community_id
    index = options.index(wanted) if wanted in options else 0
    cid = st.selectbox("Select a community", options, index=index,
                       format_func=lambda c: f"Community {c} · {sizes[c]:,} researchers")

    members = ctx.metrics[ctx.metrics["community_id"] == cid].sort_values(
        ["degree", "researcher_id"], ascending=[False, True])
    member_ids = set(members["researcher_id"])
    internal = ctx.graph.subgraph(member_ids)
    c1, c2, c3 = st.columns(3)
    c1.metric("Members", fmt_int(len(members)))
    c2.metric("Collaborations inside", fmt_int(internal.number_of_edges()))
    c3.metric("Internal density", fmt_float(nx.density(internal) if len(members) > 1 else 0.0, 4))

    st.markdown("**Members (sorted by degree)**")
    st.dataframe(members, hide_index=True, column_config=METRIC_COLUMN_CONFIG)
    st.download_button("Download this community's members (CSV)", ex.metrics_to_csv(members),
                       file_name=f"community_{cid}_members.csv", mime="text/csv")

    st.markdown("**Community subgraph**")
    if len(members) > 20:
        limit = st.slider("Max researchers to draw", 10, int(min(len(members), MAX_NETWORK_NODES)),
                          int(min(len(members), 300)), key="cm_limit")
    else:
        limit = len(members)
    selected = ctx.filters.selected if ctx.filters.selected in member_ids else None
    sub = ga.filter_graph(ctx.graph, ctx.communities, community_id=cid, top_n=limit, selected=selected)
    if len(members) > limit:
        st.caption(f"Drawing the {limit:,} highest-degree members of {len(members):,}.")
    render_html(viz.build_network_html(sub, ctx.communities, ctx.metrics, selected=selected,
                                       height_px=600, physics=True), 620)

    with st.expander("Community membership of every researcher"):
        st.dataframe(ctx.metrics[["researcher_id", "community_id"]], hide_index=True,
                     column_config=METRIC_COLUMN_CONFIG)


def page_analytics(ctx: Context) -> None:
    dataset_banner(ctx)
    st.subheader("Researcher analytics")
    note("These are <b>structural</b> measures of position in the collaboration network. A high value "
         "does <b>not</b> prove research quality, importance or productivity.")

    data = ctx.metrics
    apply_filters = st.checkbox("Apply sidebar filters to this table", value=False,
                                help="Off = every researcher in the dataset.")
    if apply_filters:
        data = data[data["researcher_id"].isin(set(ctx.view().nodes()))]

    c1, c2, c3 = st.columns([2, 2, 1])
    query = c1.text_input("Search researcher ID (partial match)", placeholder="e.g. 346")
    sort_col = c2.selectbox("Sort by", list(METRIC_LABELS) + ["researcher_id", "community_id"],
                            format_func=lambda c: METRIC_LABELS.get(c, c.replace("_", " ").capitalize()))
    descending = c3.toggle("Descending", value=True)

    if query.strip():
        data = data[data["researcher_id"].astype(str).str.contains(re.escape(query.strip()))]
    data = data.sort_values([sort_col, "researcher_id"], ascending=[not descending, True])

    st.caption(f"Showing {len(data):,} of {len(ctx.metrics):,} researchers. "
               "Click a column header to re-sort.")
    if data.empty:
        st.info("No researchers match the search. Clear the search box or the filters.")
    else:
        st.dataframe(data, hide_index=True, column_config=METRIC_COLUMN_CONFIG, height=420)

    st.markdown("### Top 10 researchers")
    metric = st.selectbox("Rank by", list(METRIC_LABELS), format_func=METRIC_LABELS.get, key="top_metric")
    ranked = ctx.metrics.dropna(subset=[metric]).sort_values(
        [metric, "researcher_id"], ascending=[False, True]).head(10)
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(viz.top_metric_chart(ctx.metrics, metric, 10), key="an_top10")
    with right:
        st.dataframe(ranked, hide_index=True, column_config=METRIC_COLUMN_CONFIG)
    if metric == "clustering_coefficient":
        st.caption("Many low-degree researchers have a clustering coefficient of exactly 1.0, so "
                   "read this ranking together with degree.")

    st.markdown("### Inspect a researcher")
    ids = ctx.metrics["researcher_id"].tolist()
    default_id = ctx.filters.selected if ctx.filters.selected in set(ids) else int(
        ctx.metrics.sort_values(["degree", "researcher_id"], ascending=[False, True]).iloc[0]["researcher_id"])
    chosen = st.selectbox("Researcher", ids, index=ids.index(default_id), key=f"inspect::{ctx.dkey}::{default_id}")
    researcher_panel(ctx, int(chosen))

    with st.expander("How to interpret each metric (and its limits)"):
        for name, meaning, limit in METRIC_GUIDE:
            st.markdown(f"**{name}** — {meaning}  \n*Limitation:* {limit}")


def page_export(ctx: Context) -> None:
    dataset_banner(ctx)
    st.subheader("Export & download")
    scope = st.radio("What should be exported?", ["Whole network", "Current filtered view"],
                     horizontal=True,
                     help="'Current filtered view' uses the search and filters in the sidebar.")
    whole = scope == "Whole network"
    tag = "full" if whole else "filtered"

    if whole:
        edges, graph_scope = ctx.edges, ctx.graph
        metrics, comms, summary = ctx.metrics, ctx.communities, ctx.summary
    else:
        graph_scope = ctx.view()
        if graph_scope.number_of_nodes() == 0:
            st.warning("The current filters match no researchers, so there is nothing to export. "
                       "Press **Reset filters** in the sidebar.")
            return
        nodes = set(graph_scope.nodes())
        edges = ex.graph_edges_dataframe(graph_scope)
        metrics = ctx.metrics[ctx.metrics["researcher_id"].isin(nodes)]
        comms = {n: c for n, c in ctx.communities.items() if n in nodes}
        summary = ga.network_summary(graph_scope, comms)

    c1, c2 = st.columns(2)
    c1.metric("Researchers in export", fmt_int(len(metrics)))
    c2.metric("Collaborations in export", fmt_int(len(edges)))

    report = ex.build_stats_report(
        summary=summary, cleaning=ctx.cleaning, settings=ctx.settings,
        scope=scope, filters=None if whole else asdict(ctx.filters), dataset_label=ctx.label,
    )

    b1, b2 = st.columns(2)
    with b1:
        st.download_button("📄 Cleaned edge list (CSV)", ex.edges_to_csv(edges),
                           file_name=f"researchcollab_{tag}_edges.csv", mime="text/csv")
        st.download_button("👥 Community assignments (CSV)", ex.communities_to_csv(comms),
                           file_name=f"researchcollab_{tag}_communities.csv", mime="text/csv")
    with b2:
        st.download_button("📈 Researcher metrics (CSV)", ex.metrics_to_csv(metrics),
                           file_name=f"researchcollab_{tag}_metrics.csv", mime="text/csv")
        st.download_button("🧾 Network statistics report (JSON)", ex.report_to_json(report),
                           file_name=f"researchcollab_{tag}_report.json", mime="application/json")
        st.download_button("🧾 Network statistics report (CSV)", ex.report_to_csv(report),
                           file_name=f"researchcollab_{tag}_report.csv", mime="text/csv")

    if not whole:
        st.caption("Metrics and communities are those computed on the FULL network, restricted to the "
                   "researchers in view. The statistics report is computed on the filtered subgraph.")
    with st.expander("Preview the statistics report"):
        st.json(report)


def page_about(label: Optional[str] = None) -> None:
    st.subheader("About & methodology")
    st.markdown(
        """
### Project objectives
ResearchCollab Explorer turns a real scientific collaboration network into something you can *see* and
*measure*. It builds a graph, computes standard network metrics, detects communities and lets you
explore the result interactively. Everything shown is calculated from the data you load.

### Dataset and citation
The default dataset is the **Stanford SNAP** *General Relativity and Quantum Cosmology* (ca-GrQc)
collaboration network, built from the arXiv GR-QC category.
According to SNAP, an undirected edge joins authors *i* and *j* if they co-authored at least one paper,
and the data cover papers from January 1993 to April 2003.

* File: <https://snap.stanford.edu/data/ca-GrQc.txt.gz>
* Dataset page: <https://snap.stanford.edu/data/ca-GrQc.html>
* Collection: <https://snap.stanford.edu/data/>

**Citation:** J. Leskovec, J. Kleinberg and C. Faloutsos. *Graph Evolution: Densification and Shrinking
Diameters.* ACM Transactions on Knowledge Discovery from Data (ACM TKDD), 1(1), 2007.

> **Researcher IDs are numbers.** The SNAP file contains no author names, institutions, paper titles or
> dates, and this app never invents any. "Researcher 3466" is only an identifier in the file.

### Nodes and edges
* **Node** = one researcher (author), identified by a numeric ID.
* **Edge** = an undirected link between two researchers who co-authored at least one paper.
* A paper with *k* authors produces a fully connected group of *k* researchers.
* Edges are unweighted: the number of joint papers and their dates are not in the file.

### Data preprocessing
1. Lines starting with `#` (and blank lines) are ignored; an optional text header is skipped.
2. The first two columns are read as researcher IDs (tab, space, comma or semicolon separated).
3. Malformed rows and invalid IDs (not whole numbers) are removed.
4. Self-loops are removed.
5. Duplicate edges are removed; `(1, 2)` and `(2, 1)` are the same undirected edge.
6. The *Overview* page shows how many rows each step removed.

### Graph algorithms
* **Graph construction** - NetworkX `Graph` (undirected, simple).
* **Degree, density, components, clustering** - standard NetworkX routines. Average degree = 2m / n;
  density = m / (n(n−1)/2).
* **Betweenness centrality** - Brandes' algorithm. Above 2,000 researchers the app *samples* k source nodes
  (default k = 500; configurable under *Advanced settings*), giving an approximation that is
  reproducible through the random seed. Up to 2,000 researchers it is exact.
* **Closeness centrality** - Wasserman-Faust scaling, so disconnected graphs are handled. Exact up to
  2,000 researchers; above that, estimated from the same k sampled pivot nodes (components that no pivot
  reaches are computed exactly).
* **Community detection** - Louvain method (python-louvain): nodes are moved between groups to increase
  *modularity*, groups are merged, and the process repeats. A fixed seed makes it reproducible.
  Community numbers are assigned by size (0 = largest).
* **Caching** - expensive results are cached with Streamlit, so moving a slider does not recompute them.

### Meaning of the metrics
"""
    )
    for name, meaning, limit in METRIC_GUIDE:
        st.markdown(f"* **{name}** — {meaning} *Limitation:* {limit}")
    st.markdown(
        """
### Known limitations
* Co-authorship shows that people appear on the same paper, not how much they worked together.
* Author-name disambiguation is not verifiable here: one real person might appear as several IDs, or
  different people as one.
* The dataset covers one arXiv category and a fixed period; it is not all of science.
* Large collaborations create cliques, which inflates clustering and degree for their authors.
* Centrality and community results describe **network structure only** - never research quality.
* Louvain is heuristic: different seeds or resolutions can give somewhat different communities.
* Betweenness and closeness are sampling-based estimates on large networks; set the sample size to 0 for
  exact values (slow).
* The interactive view draws at most 2,000 nodes so that the browser stays responsive.
"""
    )
    st.caption("Built with Streamlit, NetworkX, PyVis, Plotly, pandas and python-louvain.")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    page = sidebar_navigation()
    data = sidebar_data_source()
    hero()

    if page == PAGE_ABOUT:
        page_about()
        return
    if data is None:
        empty_state(page)
        return

    edges, cleaning, label = data
    n_nodes = int(pd.unique(edges[["source", "target"]].to_numpy().ravel()).size)
    dkey = f"{label}:{len(edges)}:{n_nodes}"

    k, seed, closeness = sidebar_advanced(n_nodes, dkey)
    analysis = cached_analysis(edges, k, seed, closeness)
    graph = cached_graph(edges)

    exact = k <= 0 or k >= n_nodes
    method = "exact" if exact else f"approximate (k={k} sampled source nodes)"
    settings = {
        "louvain_seed": seed,
        "louvain_resolution": 1.0,
        "betweenness": method,
        "closeness": method if closeness else "skipped",
    }
    filters = sidebar_filters(analysis["summary"], analysis["comm_summary"], graph, dkey)

    ctx = Context(
        label=label, edges=edges, cleaning=cleaning, graph=graph,
        communities=analysis["communities"], metrics=analysis["metrics"],
        summary=analysis["summary"], degree_dist=analysis["degree_dist"],
        comm_summary=analysis["comm_summary"], settings=settings, dkey=dkey, filters=filters,
    )

    {
        PAGE_OVERVIEW: page_overview,
        PAGE_NETWORK: page_network,
        PAGE_COMMUNITIES: page_communities,
        PAGE_ANALYTICS: page_analytics,
        PAGE_EXPORT: page_export,
    }[page](ctx)


main()
