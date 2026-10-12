# Data folder

## Files

| File | Purpose | Committed to Git? |
|------|---------|-------------------|
| `sample_edges.csv` | Tiny synthetic network (30 researchers) for first-time testing | Yes |
| `ca-GrQc.txt.gz` | The real SNAP dataset (you download it) | **No** (ignored by `.gitignore`) |

## Getting the real dataset

Official file: <https://snap.stanford.edu/data/ca-GrQc.txt.gz>
Dataset page: <https://snap.stanford.edu/data/ca-GrQc.html>

1. Download `ca-GrQc.txt.gz`.
2. Put it in this folder, so the path is `data/ca-GrQc.txt.gz`.
3. **Do not extract it** - the app reads the compressed file directly. (An extracted
   `ca-GrQc.txt` also works.)

Alternatively, use the **"Download SNAP dataset"** button in the app sidebar, or upload
any edge-list file with the **Upload file** option.

## Accepted formats

* Gzip-compressed or plain text.
* One edge per line: `id1<separator>id2`, separator = tab, space, comma or semicolon.
* Lines beginning with `#` (or `%`) and blank lines are ignored.
* An optional header row such as `source,target` is detected and skipped.
* IDs must be non-negative integers. Other rows are counted as *invalid* and dropped.
* Self-loops (`5,5`) and duplicate edges (including reversed ones, `2,1` after `1,2`) are
  removed. The app shows exactly how many rows were removed for each reason.

## Important

The SNAP file contains **numeric IDs only**. It has no author names, paper titles,
institutions or dates, and this project never invents them.

## Citation

J. Leskovec, J. Kleinberg and C. Faloutsos. *Graph Evolution: Densification and Shrinking
Diameters.* ACM Transactions on Knowledge Discovery from Data (ACM TKDD), 1(1), 2007.
