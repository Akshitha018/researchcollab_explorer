"""ResearchCollab Explorer - source package.

Modules
-------
data_loader     : read, parse and clean edge-list files (SNAP .txt.gz, .txt, .csv)
graph_analysis  : build the NetworkX graph, compute metrics, detect communities
visualization   : PyVis network HTML and Plotly charts
exports         : turn results into downloadable CSV / JSON bytes
"""

__all__ = ["data_loader", "graph_analysis", "visualization", "exports"]
