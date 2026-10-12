"""Data loading and cleaning for ResearchCollab Explorer.

What this module does
---------------------
1. Opens an edge-list file (gzip-compressed ``.txt.gz``, plain ``.txt`` or ``.csv``)
   or raw uploaded bytes.
2. Parses every line, ignoring comments (``#``), blank lines and an optional header.
3. Cleans the data:
      * malformed rows (fewer than two columns)         -> dropped
      * invalid IDs (not a non-negative integer)         -> dropped
      * self-loops (a researcher linked to themselves)   -> dropped
      * duplicate edges (``1,2`` and ``2,1`` are the SAME undirected edge) -> collapsed
4. Returns a clean ``pandas.DataFrame`` with columns ``source`` and ``target``
   (always ``source < target``, sorted) plus a ``CleaningReport`` describing what happened.

The SNAP dataset only contains numeric IDs. This module never invents names, titles,
institutions or dates.
"""

from __future__ import annotations

import gzip
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Tuple, Union

import pandas as pd

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
SNAP_DATASET_URL = "https://snap.stanford.edu/data/ca-GrQc.txt.gz"
SNAP_DATASET_FILENAME = "ca-GrQc.txt.gz"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_PATH = DATA_DIR / "sample_edges.csv"

# The dataset path can be changed without editing code:  set RESEARCHCOLLAB_DATA_PATH.
# The default is RELATIVE to the project folder so no absolute server path is displayed.
DEFAULT_DATASET_PATH = os.environ.get(
    "RESEARCHCOLLAB_DATA_PATH", f"data/{SNAP_DATASET_FILENAME}"
)

GZIP_MAGIC = b"\x1f\x8b"  # the first two bytes of every gzip file
MAX_ID_DIGITS = 18  # keeps IDs inside the int64 range

_SPLIT_RE = re.compile(r"[,\s;]+")  # tab, space, comma or semicolon separated
_ID_RE = re.compile(r"^\d+$")  # a non-negative integer written with digits only
_HEADER_RE = re.compile(r"^[A-Za-z_][A-Za-z_ ]*$")  # e.g. "source", "from_id"


class DatasetError(Exception):
    """A problem with the dataset that can be shown to the user as-is."""


@dataclass
class CleaningReport:
    """Counts describing how the raw file was cleaned."""

    total_lines: int = 0
    comment_or_blank_lines: int = 0
    header_lines: int = 0
    malformed_rows: int = 0
    invalid_id_rows: int = 0
    valid_rows: int = 0  # rows with two valid IDs, before removing loops/duplicates
    self_loops_removed: int = 0
    duplicate_edges_removed: int = 0
    unique_edges: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Parsing and cleaning
# --------------------------------------------------------------------------------------
def parse_edge_lines(lines: Iterable[str]) -> Tuple[pd.DataFrame, CleaningReport]:
    """Parse text lines into a clean edge DataFrame.

    Returns ``(edges, report)``. ``edges`` may be empty; use :func:`_require_edges`
    (done automatically by the ``load_*`` functions) to turn that into an error.
    """
    report = CleaningReport()
    sources: list[int] = []
    targets: list[int] = []
    first_data_row = True

    for raw in lines:
        report.total_lines += 1
        line = raw.strip().lstrip("﻿")  # also drop a UTF-8 byte-order mark
        if not line or line.startswith("#") or line.startswith("%"):
            report.comment_or_blank_lines += 1
            continue

        tokens = [t for t in _SPLIT_RE.split(line) if t]
        if len(tokens) < 2:
            report.malformed_rows += 1
            first_data_row = False
            continue

        a, b = tokens[0], tokens[1]

        # A text header such as "source,target" is allowed only as the first data row.
        if first_data_row and _HEADER_RE.match(a) and _HEADER_RE.match(b):
            report.header_lines += 1
            first_data_row = False
            continue
        first_data_row = False

        if not (_ID_RE.match(a) and _ID_RE.match(b)):
            report.invalid_id_rows += 1
            continue
        if len(a) > MAX_ID_DIGITS or len(b) > MAX_ID_DIGITS:
            report.invalid_id_rows += 1
            continue

        sources.append(int(a))
        targets.append(int(b))

    edges = _clean_edges(sources, targets, report)
    return edges, report


def _clean_edges(sources: list, targets: list, report: CleaningReport) -> pd.DataFrame:
    """Remove self-loops and duplicates; orient every edge as (smaller ID, larger ID)."""
    df = pd.DataFrame(
        {
            "source": pd.Series(sources, dtype="int64"),
            "target": pd.Series(targets, dtype="int64"),
        }
    )
    report.valid_rows = int(len(df))

    loops = df["source"] == df["target"]
    report.self_loops_removed = int(loops.sum())
    df = df.loc[~loops]

    # In an UNDIRECTED graph, (2, 1) is the same collaboration as (1, 2).
    # Orienting every pair as (min, max) lets drop_duplicates catch both.
    low = df[["source", "target"]].min(axis=1)
    high = df[["source", "target"]].max(axis=1)
    df = pd.DataFrame({"source": low, "target": high})

    before = len(df)
    df = df.drop_duplicates().sort_values(["source", "target"]).reset_index(drop=True)
    report.duplicate_edges_removed = int(before - len(df))
    report.unique_edges = int(len(df))
    return df


def _require_edges(edges: pd.DataFrame, report: CleaningReport, label: str):
    if edges.empty:
        raise DatasetError(
            f"No valid edges were found in '{label}'. Expected one edge per line, e.g. "
            f"'123 456' or '123,456' (numeric researcher IDs). "
            f"Lines read: {report.total_lines}, malformed: {report.malformed_rows}, "
            f"invalid IDs: {report.invalid_id_rows}, self-loops: {report.self_loops_removed}."
        )
    return edges, report


# --------------------------------------------------------------------------------------
# Public loaders
# --------------------------------------------------------------------------------------
def load_edges_from_path(path: Union[str, Path]) -> Tuple[pd.DataFrame, CleaningReport]:
    """Load a ``.txt.gz``, ``.txt`` or ``.csv`` edge list from disk.

    Compression is detected from the file's first two bytes, not from its extension,
    so a gzip file renamed to ``.txt`` still works.
    Error messages show only the file name, never the full folder path.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists() or not p.is_file():
        raise DatasetError(
            f"Dataset file '{p.name}' was not found. Place it in the project's 'data' "
            f"folder, upload a file, use the bundled sample, or download it from "
            f"{SNAP_DATASET_URL}"
        )
    try:
        with p.open("rb") as fh:
            magic = fh.read(2)
        opener = gzip.open if magic == GZIP_MAGIC else open
        with opener(p, "rt", encoding="utf-8", errors="replace") as fh:
            edges, report = parse_edge_lines(fh)
    except DatasetError:
        raise
    except (OSError, EOFError) as exc:  # includes gzip.BadGzipFile
        raise DatasetError(
            f"Could not read '{p.name}'. The file may be corrupt or incomplete "
            f"({type(exc).__name__}). Try downloading it again."
        ) from exc
    return _require_edges(edges, report, p.name)


def load_edges_from_bytes(
    data: bytes, filename: str = "uploaded file"
) -> Tuple[pd.DataFrame, CleaningReport]:
    """Load an edge list from raw bytes (used for browser uploads)."""
    label = Path(filename).name or "uploaded file"
    if not data:
        raise DatasetError(f"The uploaded file '{label}' is empty.")
    try:
        if data[:2] == GZIP_MAGIC:
            data = gzip.decompress(data)
    except (OSError, EOFError) as exc:
        raise DatasetError(
            f"The uploaded file '{label}' looks gzip-compressed but could not be "
            f"decompressed ({type(exc).__name__})."
        ) from exc
    text = data.decode("utf-8", errors="replace")
    edges, report = parse_edge_lines(text.splitlines())
    return _require_edges(edges, report, label)


def resolve_dataset_path(path_text: Union[str, Path, None]) -> Union[Path, None]:
    """Turn user-typed text into a Path. Relative paths are relative to the project folder."""
    text = str(path_text or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def locate_default_dataset() -> Union[Path, None]:
    """The configured default dataset if it exists, else any SNAP file in ``data/``."""
    configured = resolve_dataset_path(DEFAULT_DATASET_PATH)
    if configured is not None and configured.is_file():
        return configured
    return find_default_dataset()


def find_default_dataset(data_dir: Union[str, Path] = DATA_DIR) -> Union[Path, None]:
    """Return the first SNAP file found in ``data_dir`` (compressed first), else None."""
    folder = Path(data_dir)
    for name in (SNAP_DATASET_FILENAME, "ca-GrQc.txt"):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def download_snap_dataset(
    dest_dir: Union[str, Path] = DATA_DIR, url: str = SNAP_DATASET_URL, timeout: int = 60
) -> Path:
    """Download the official SNAP file into ``dest_dir`` and return its path.

    Used by the optional "Download SNAP dataset" button, mainly for deployments where
    the (ignored-by-Git) dataset file is not present. Needs internet access only for
    this one call; no API key is involved.
    """
    if not url.lower().startswith("https://"):
        raise DatasetError("Only https:// download URLs are allowed.")
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / SNAP_DATASET_FILENAME
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ResearchCollab-Explorer"})
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as out:
            out.write(response.read())
        with tmp.open("rb") as fh:
            if fh.read(2) != GZIP_MAGIC:
                raise DatasetError("The downloaded file is not a gzip file; it was discarded.")
        tmp.replace(target)
    except DatasetError:
        tmp.unlink(missing_ok=True)
        raise
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        raise DatasetError(
            f"Download failed ({type(exc).__name__}). Check your internet connection, "
            f"or download the file manually from {SNAP_DATASET_URL}"
        ) from exc
    return target
