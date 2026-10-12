# 🕸️ ResearchCollab Explorer

**Explore the connections that shape scientific discovery.**

ResearchCollab Explorer is a Streamlit web app that builds, analyses and visualises a real scientific
collaboration network: **researchers are nodes**, and an **undirected edge** joins two researchers who
co-authored at least one paper. It uses the Stanford SNAP *General Relativity and Quantum Cosmology*
(ca-GrQc) dataset, and everything on screen is calculated from the loaded data - there are no hard-coded
statistics.

| Feature | What it does |
|---|---|
| Overview | Nodes, edges, average degree, components, clustering, density + 4 interactive charts |
| Collaboration network | PyVis force-directed graph: zoom, pan, drag, hover, search, neighbour highlighting |
| Communities | Louvain communities, size charts, member tables, community sub-graph |
| Researcher analytics | Searchable/sortable metrics table, top-10 by any metric, per-researcher inspector |
| Search & filters | ID search, community, minimum degree, Top-N, neighbourhood view, reset |
| Export | Edge list, metrics, communities (CSV) and a statistics report (JSON/CSV) |
| About & methodology | Dataset citation, definitions, algorithms, limitations |

> **Researcher IDs are numbers.** The SNAP file contains no names, institutions, titles or dates, and this
> project never invents any.

---

## Project structure

```
ResearchCollab-Explorer/
├── app.py                      # Streamlit app (UI + pages)
├── requirements.txt            # runtime dependencies
├── requirements-dev.txt        # + pytest
├── pytest.ini
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml             # theme + upload limit
├── data/
│   ├── README.md
│   └── sample_edges.csv        # tiny synthetic sample (30 researchers)
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # read + clean edge lists (.txt.gz / .txt / .csv)
│   ├── graph_analysis.py       # NetworkX graph, metrics, Louvain, filtering
│   ├── visualization.py        # PyVis HTML + Plotly charts
│   └── exports.py              # CSV / JSON download bytes
└── tests/
    ├── test_data_loader.py
    ├── test_graph_analysis.py
    ├── test_exports.py
    └── test_app_smoke.py       # runs the real app headlessly
```

---

## Windows setup (PowerShell + VS Code)

Prerequisite: Python 3.11 or newer (3.13 also works) and VS Code. Check with `python --version`.

**1. Create the project folder** (or extract the provided zip here instead):

```powershell
cd $HOME\Documents
mkdir ResearchCollab-Explorer
cd ResearchCollab-Explorer
```

**2. Open it in VS Code:**

```powershell
code .
```

(If `code` is not recognised, open VS Code, choose *File → Open Folder…* and pick the folder.)
Open a terminal inside VS Code with *Terminal → New Terminal* (it starts in the project folder).

**3. Create a virtual environment:**

```powershell
python -m venv venv
```

**4. Activate it:**

```powershell
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` at the start of the prompt. If PowerShell says *"running scripts is disabled"*, run
this once and activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

In VS Code, press `Ctrl+Shift+P` → *Python: Select Interpreter* → choose the one inside `venv`.

**5. Install the requirements:**

```powershell
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

**6. Get the dataset (optional - the app also works with the bundled sample):**
download <https://snap.stanford.edu/data/ca-GrQc.txt.gz> and put it at `data\ca-GrQc.txt.gz`
(do **not** extract it). Or skip this step and click **⬇ Download SNAP dataset** in the app sidebar.

**7. Run the app:**

```powershell
streamlit run app.py
```

Your browser opens at <http://localhost:8501>. Stop the app with `Ctrl+C`.

**8. Run the tests:**

```powershell
pytest
```

**Faster first look:** with no dataset file the app starts on the bundled sample automatically.

### How to test each feature

| Feature | Try this |
|---|---|
| Cleaning | Overview → *Data cleaning report*. With the sample you should see 1 self-loop and 2 duplicates removed |
| Overview metrics/charts | Overview page; toggle *Log-log scale*; move the "researchers to plot" slider |
| Interactive network | *Collaboration network*: scroll to zoom, drag nodes, hover for tooltips, click a node |
| Search | Sidebar → *Search researcher ID* (e.g. `4` in the sample). Try `abc` and `99999999` to see the messages |
| Neighbourhood view | After selecting a researcher, tick *Show only selected researcher's neighbourhood* |
| Filters | *Minimum degree*, *Top N nodes*, *Community*; **Reset filters** restores defaults |
| Communities | *Communities* page: pick a community, see members and its sub-graph |
| Analytics | *Researcher analytics*: search, sort, top-10 by metric, inspect a researcher |
| Export | *Export & download*: switch between *Whole network* and *Current filtered view* |
| Upload | Sidebar → *Upload file* → any `.txt`, `.csv` or `.gz` edge list |

---

## Dataset

* **Source:** Stanford Network Analysis Project (SNAP), *ca-GrQc* - arXiv General Relativity and Quantum
  Cosmology collaboration network. <https://snap.stanford.edu/data/ca-GrQc.html>
* **Direct file:** <https://snap.stanford.edu/data/ca-GrQc.txt.gz>
* **Format:** compressed text edge list; lines starting with `#` are comments; each line is
  `id1<TAB>id2`.
* **Supported inputs:** `.txt.gz` (read directly), extracted `.txt`, or an uploaded CSV/TXT edge list.
* **Cleaning:** malformed rows, invalid IDs, self-loops and duplicate edges are removed and counted.
* **Configurable path:** edit the *Dataset path* box in the sidebar, or set an environment variable:

  ```powershell
  $env:RESEARCHCOLLAB_DATA_PATH = "data\my_edges.txt"
  streamlit run app.py
  ```

**Citation**

> J. Leskovec, J. Kleinberg and C. Faloutsos. *Graph Evolution: Densification and Shrinking Diameters.*
> ACM Transactions on Knowledge Discovery from Data (ACM TKDD), 1(1), 2007.

```bibtex
@article{leskovec2007graph,
  title   = {Graph Evolution: Densification and Shrinking Diameters},
  author  = {Leskovec, Jure and Kleinberg, Jon and Faloutsos, Christos},
  journal = {ACM Transactions on Knowledge Discovery from Data (TKDD)},
  volume  = {1},
  number  = {1},
  year    = {2007}
}
```

---

## How the code works (for your viva)

1. **`data_loader.py`** reads each line, ignores comments/blank lines, splits on tab/space/comma/semicolon,
   checks that the first two tokens are whole numbers, drops self-loops, orders each pair as
   `(smaller, larger)` so that `(2,1)` and `(1,2)` are the same undirected edge, and removes duplicates.
   It returns a clean DataFrame plus a `CleaningReport` of counts. Gzip is detected from the file's first
   two bytes (`1f 8b`).
2. **`graph_analysis.py`** builds an `nx.Graph` and computes:
   * average degree = `2m / n`; density = `m / (n(n-1)/2)`;
   * connected components (groups reachable through edges);
   * clustering coefficient of a node = (edges among its neighbours) / (possible edges among them);
   * degree centrality = `degree / (n-1)`;
   * betweenness = fraction of shortest paths between other pairs that go through the node (Brandes);
   * closeness = (reachable nodes) / (sum of distances), Wasserman-Faust scaled for disconnected graphs;
   * **Louvain** communities: repeatedly move nodes to the neighbouring group that most increases
     *modularity*, then merge groups and repeat. A fixed seed makes it reproducible; communities are
     numbered by size (0 = largest).
3. **Sampling for speed:** betweenness and closeness need a breadth-first search from *every* node
   (`O(n·m)`). Above 2,000 researchers the app runs the search from **k = 500 random pivot nodes** instead
   (an unbiased estimate; change *Sample size* under *Advanced settings*, `0` = exact).
4. **`visualization.py`** creates the PyVis network as an in-memory HTML string (JavaScript inlined, so no
   temp folder or internet is needed) and the Plotly charts.
5. **`app.py`** uses `st.cache_data` / `st.cache_resource`, so loading and the expensive metrics run once
   per dataset/setting; moving a slider only re-filters cached results.

### What the metrics do and do not mean

High degree/centrality means a researcher occupies a well-connected or bridging **position in this
co-authorship graph**. It is **not** evidence of research quality, importance or productivity. A
detected community is a **structurally close group**, not necessarily an official research group.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognised | Reinstall Python and tick *Add python.exe to PATH*; or try `py -3` |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `streamlit` not recognised | Activate the venv first (`(venv)` visible in the prompt) and re-run `pip install -r requirements.txt` |
| `ModuleNotFoundError: community` | Install the right package: `pip install python-louvain` (not `community`) |
| `ModuleNotFoundError: src` | Run commands from the project root (the folder containing `app.py`) |
| "Dataset file … was not found" | Put `ca-GrQc.txt.gz` in `data\`, use *Download SNAP dataset*, upload a file, or pick *Bundled sample* |
| First load takes ~20-60 s on the full dataset | Metrics are being computed once; later interactions are instant (cached) |
| Network area is blank | Reduce *Top N nodes*; make sure your browser allows JavaScript; refresh the page |
| Port 8501 busy | `streamlit run app.py --server.port 8502` |
| Stale results after editing code | In the app menu choose *Clear cache*, or restart Streamlit |

---

## Deployment on Streamlit Community Cloud

1. Create a **public GitHub repository** and push the project:

   ```powershell
   git init
   git add .
   git commit -m "ResearchCollab Explorer"
   git branch -M main
   git remote add origin https://github.com/<your-username>/ResearchCollab-Explorer.git
   git push -u origin main
   ```

   `.gitignore` already keeps `venv/`, caches, generated HTML and large dataset files out of Git
   (only `data/README.md` and `data/sample_edges.csv` are committed).
2. Go to <https://share.streamlit.io>, sign in with GitHub, choose **Create app**, select the repository,
   branch `main` and main file `app.py`, then **Deploy**. Dependencies come from `requirements.txt`.
3. **Getting the dataset on the deployed app** - the file is not in Git, so users can either:
   * click **⬇ Download SNAP dataset** in the sidebar (the server downloads it from SNAP once), or
   * choose **Upload file** and upload `ca-GrQc.txt.gz`, or
   * use the **Bundled sample**.

   (The 100 KB real file *could* be committed if you prefer: remove the `data/*` rules in `.gitignore`
   and add `!data/ca-GrQc.txt.gz`. This is optional.)
4. Community Cloud apps have limited memory/CPU: keep the default sampling for large graphs.

---

## Limitations and future enhancements

**Limitations**
* Co-authorship does not measure how closely people worked together; edges are unweighted.
* Author-name disambiguation cannot be checked: one person may appear as several IDs or vice versa.
* One arXiv category and a fixed period only; not all of science.
* Large collaborations create cliques that inflate degree and clustering for their authors.
* Louvain is a heuristic; other seeds/resolutions may give somewhat different communities.
* Betweenness/closeness are sampling-based estimates on large graphs.
* At most 2,000 nodes are drawn at once to keep the browser responsive.

**Future work**
* Weighted edges (number of joint papers) and time-sliced networks, if a documented dataset provides them.
* Other community algorithms (Leiden, label propagation) with side-by-side comparison.
* Real author metadata from a documented source (e.g. OpenAlex) joined by ID, with clear attribution.
* Link prediction ("who might collaborate next") and network-evolution animation.

## License / attribution

Code: for educational use. Data: © the SNAP authors; please cite the paper above when using the dataset.
