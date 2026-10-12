# ResearchCollab Explorer

ResearchCollab Explorer is a Streamlit dashboard for exploring academic co-authorship as an undirected, weighted graph. It can analyze local benchmark samples or works returned by an explicit OpenAlex query. Sample data is always identified as sample data; it is not presented as live API data.

## Quick start (Windows / VS Code)

Use Python 3.10 or later. From the project directory in the VS Code PowerShell terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

If PowerShell prevents virtual-environment activation, run this for the current terminal only:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

Streamlit normally serves the application at `http://localhost:8501`. The supported app entry point is `app.py`. The optional `run_app.py` wrapper defaults to port 3000; use the Streamlit command above unless that wrapper is specifically needed.

### OpenAlex data

The app starts with the local Graph Neural Networks sample. To use real OpenAlex works, choose **Live OpenAlex API**, enter a research query, and click **Query API**. A successful response replaces the current corpus; a failed request leaves the current corpus intact and displays an error.

Set `OPENALEX_EMAIL` to a contact address for the OpenAlex polite-pool User-Agent, if desired. In PowerShell, set it before launching Streamlit:

```powershell
$env:OPENALEX_EMAIL = "you@example.org"
```

No API key is required. Requests use a bounded response size (10–100 works), a 3.05-second connection timeout and a 12-second read timeout. Successful query results are cached in Streamlit for one hour; request failures are not cached. Network availability and OpenAlex service status are not guaranteed.

## What the application does

The Streamlit sidebar identifies the currently loaded corpus as either **OFFLINE SAMPLE DATASET** or **LIVE OPENALEX API DATA**. The label is based on the loaded dataset, not just on the selected ingestion option.

The dashboard includes:

- **Overview Dashboard:** corpus size, graph summary statistics, publication-year distribution, top collaborators and central researchers.
- **Interactive Network Graph:** name search/highlighting, researcher-focused one-hop views, year and minimum shared-publication filters, isolate removal, layout choice, node sizing and coloring. Edge width and edge hover details show shared-publication strength.
- **Graph & Centrality Analysis:** degree distribution, PageRank-versus-betweenness chart, centrality ranking and institution filter.
- **Community Detection:** weighted Louvain detection with Greedy Modularity, Label Propagation and Connected Components fallbacks.
- **Researcher Profiles:** researcher details, direct collaborators, shared-work counts, a one-hop ego graph and corpus publications.
- **Data Tables & Exports:** researcher, collaboration, publication and authorship CSV files; summary-metrics CSV; and GraphML for external graph tools.
- **Methodology & Viva Guide:** definitions, caveats and oral-defense questions.

## Architecture

`app.py` is the Streamlit UI and orchestration entry point. It loads a dataset, builds the graph and metrics, and renders the selected page.

| Path | Responsibility |
| --- | --- |
| `modules/data_loader.py` | Loads local JSON samples; calls OpenAlex `/works`; parses publications, authors and authorships; caches successful queries; reports request/response errors. |
| `modules/network_builder.py` | Constructs the NetworkX graph, filters it, and creates node/edge tables. |
| `modules/graph_metrics.py` | Computes network-level and researcher-level metrics. |
| `modules/community_detection.py` | Detects and summarizes communities. |
| `modules/visualization.py` | Generates Plotly graph views and charts using the dark palette. |
| `modules/export_utils.py` | Converts corpus tables to CSV and graphs to GraphML. |
| `sample_data/*.json` | Three local, static benchmark/demo corpora: GNN, Quantum Computing and CRISPR. |
| `tests/` | Pytest checks for graph semantics, API behavior, visualization output and Streamlit pages. |
| `.streamlit/config.toml` | Streamlit theme and server settings. |

`src/App.tsx`, `src/main.tsx`, `src/index.css`, `index.html`, `vite.config.ts`, `tsconfig.json`, `package.json` and `metadata.json` are a separate, currently unused Vite/React / AI Studio scaffold. Its React root currently renders an empty app; its Gemini capability metadata and the placeholder Gemini values in `.env.example` are not used by the Streamlit code. Python is the application runtime; no Gemini key or Node/npm installation is required for the supported setup. `.vscode/launch.json` and `.vscode/settings.json` provide Python/pytest launch and discovery settings.

## Data and analysis methodology

### Input handling

Offline files are static sample datasets and are loaded without network access. The three samples are useful for demonstrations and repeatable local analysis; they should not be treated as a complete or freshly retrieved research corpus.

Live mode searches the OpenAlex Works endpoint. The app requests the highest-cited matching works within its result limit (`per_page`), excludes paratext, then keeps the returned corpus and its authorship metadata in memory. It reads the Works response's `id`, `display_name`/`title`, `publication_year`, `cited_by_count`, `doi`, `primary_topic`, `topics`, and `authorships` fields. OpenAlex author IDs are used as researcher keys where present. The dashboard analyzes only the works returned for that query—not every work associated with each researcher.

### Graph construction

The corpus becomes an undirected simple graph \(G=(V,E,W)\):

- Each researcher ID is a node; catalog researchers with no co-authorship remain as isolated nodes.
- Two different researchers share an edge if they appear together in a returned work.
- Each edge weight is the number of shared works in the corpus. An author's repeated occurrence inside one work is deduplicated before pairs are generated, preventing self-loops and duplicate contributions from that record.
- All authors on a multi-author work are connected pairwise. Thus a work with \(k\) distinct authors contributes up to \(k(k-1)/2\) edges.

The year filter applies to each shared work; works with no valid year are excluded when a year range is active. A minimum-strength filter removes edges below the selected number of shared works. The optional isolate filter removes nodes with no remaining links.

### Metrics and communities

- Degree counts directly connected researchers; weighted degree sums edge weights.
- PageRank uses edge weight. Degree centrality, betweenness, closeness and clustering describe the graph topology; betweenness/closeness use unweighted shortest paths.
- Density, average degree, clustering and connected components describe the selected graph. Average path length and diameter are calculated on its largest connected component, so disconnected pairs outside that component do not produce infinite distances.
- Louvain maximizes weighted modularity where supported. The dashboard reports the algorithm used and the modularity score; a detected community is an algorithmic cluster, not necessarily a complete clique or a validated scientific specialty.
- Plotly graph coordinates are for visualization only and have no geographic or causal meaning. Edge widths are scaled for display and do not change graph calculations.

## Exports

CSV exports contain the currently loaded corpus and graph-derived tables. The collaboration CSV includes each edge's integer shared-publication weight and sample titles. GraphML contains node attributes and edge weights for Gephi/Cytoscape; nested publication details are not embedded in GraphML.

## Limitations and responsible interpretation

- Sample datasets are curated examples, not live results. Always check the sidebar data-source badge before using a chart or export.
- Live analyses are limited by the text query and result cap and are citation-sorted. Results can omit relevant work, and citation counts change over time.
- OpenAlex author disambiguation is useful but imperfect. Missing IDs fall back to normalized names, which can merge distinct researchers with identical names.
- This is a corpus-level collaboration model, not an author's full publication history. Per-researcher publication counts in the graph represent works in this corpus, not OpenAlex lifetime totals.
- Every co-author pair receives equal credit for a shared work. Hyper-authored papers produce dense cliques and can inflate degree, density and clustering.
- Citation counts are metadata on publications and are not a measure of author quality. Network centrality is not equivalent to research impact.
- Layouts, color bins and chart scales are visual aids. A degree histogram alone does not establish a power-law or scale-free distribution.
- The app has no authentication or authorization layer. Keep it on localhost or a trusted network; do not expose the development server publicly without appropriate deployment security.

## Tests and verification

Run the local suite with:

```powershell
python -m pytest -q
```

The tests exercise graph edges and weights, temporal/ego filters, metrics, community output, CSV/GraphML serialization, chart data, mocked OpenAlex cache/error/missing-metadata paths and navigation through Streamlit pages. By default the opt-in live integration test is skipped. To issue one real, bounded request and verify its parsed records through graph metrics and a Plotly figure:

```powershell
$env:RUN_OPENALEX_INTEGRATION = "1"
python -m pytest -q tests/test_openalex_integration.py
Remove-Item Env:RUN_OPENALEX_INTEGRATION
```

This requires network access and makes one OpenAlex Works search request for at most 10 works (8-second read timeout). It does not print researcher names, contact information or API keys. A live test can fail because of network or service availability; mocked tests remain offline and deterministic.

## Viva / oral-defense questions

1. **Why is co-authorship modeled as undirected?**  
   Authorship of the same work is mutual: if A and B co-author it, each is a co-author of the other. Citation links, unlike co-authorship, are directed.

2. **What does an edge weight mean?**  
   It is the number of returned corpus works shared by the two endpoint researchers. It measures tie strength within this corpus, not the total collaboration history of either author.

3. **How do degree and weighted degree differ?**  
   Degree counts distinct direct collaborators. Weighted degree sums the shared-work counts across those ties.

4. **How do PageRank and betweenness differ?**  
   Weighted PageRank reflects connections to other well-connected researchers. Betweenness measures how often a node lies on shortest paths between other nodes; in this implementation those paths are unweighted.

5. **Why calculate path metrics on the largest connected component?**  
   Shortest paths between disconnected components do not exist. Restricting these metrics to the largest connected component gives finite distances and makes the scope explicit.

6. **What does modularity describe?**  
   It compares within-community edge concentration with a degree-based random-network null model. A high value can indicate structure in this graph, but does not prove that the groups are disciplines or causal collaborations.

7. **What are the main co-authorship network limitations?**  
   Author disambiguation errors, incomplete query samples, equal co-author credit and hyper-authorship can all materially affect graph statistics.

8. **How does the app distinguish sample and live data?**  
   Local JSON files set `is_live=False` and carry an offline-sample source note; successful OpenAlex responses set `is_live=True`. The sidebar badge reads that loaded-dataset flag.
