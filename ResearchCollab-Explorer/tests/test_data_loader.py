"""Tests for src/data_loader.py (parsing, cleaning, file formats, errors)."""

import gzip

import pytest

from src import data_loader as dl


# --------------------------------------------------------------------------------------
# parse_edge_lines: cleaning rules
# --------------------------------------------------------------------------------------
def test_comments_and_blank_lines_are_ignored():
    lines = ["# Directed graph", "# FromNodeId\tToNodeId", "", "1\t2", "2\t3"]
    edges, report = dl.parse_edge_lines(lines)
    assert len(edges) == 2
    assert report.comment_or_blank_lines == 3
    assert report.malformed_rows == 0


def test_self_loops_are_removed():
    edges, report = dl.parse_edge_lines(["1 1", "1 2", "3 3"])
    assert edges.to_dict("records") == [{"source": 1, "target": 2}]
    assert report.self_loops_removed == 2


def test_duplicates_and_reversed_duplicates_are_removed():
    edges, report = dl.parse_edge_lines(["1 2", "2 1", "1 2", "3 4"])
    assert len(edges) == 2
    assert report.duplicate_edges_removed == 2
    assert report.unique_edges == 2
    assert (edges["source"] < edges["target"]).all()  # always oriented low -> high


def test_malformed_and_invalid_rows_are_counted_and_dropped():
    lines = ["1 2", "lonely", "abc def", "-5 3", "4.5 6", "7 8"]
    edges, report = dl.parse_edge_lines(lines)
    assert len(edges) == 2
    assert report.malformed_rows == 1
    assert report.invalid_id_rows == 3


def test_oversized_ids_are_invalid():
    edges, report = dl.parse_edge_lines(["1 2", "9" * 40 + " 3"])
    assert len(edges) == 1
    assert report.invalid_id_rows == 1


@pytest.mark.parametrize("separator", ["\t", " ", ",", ";", "  "])
def test_all_separators_work(separator):
    edges, _ = dl.parse_edge_lines([f"10{separator}20"])
    assert edges.to_dict("records") == [{"source": 10, "target": 20}]


def test_header_row_is_skipped_but_counted():
    edges, report = dl.parse_edge_lines(["source,target", "1,2"])
    assert len(edges) == 1
    assert report.header_lines == 1
    assert report.invalid_id_rows == 0


def test_extra_columns_are_ignored():
    edges, _ = dl.parse_edge_lines(["1,2,0.5", "2,3,0.9"])
    assert len(edges) == 2


def test_utf8_bom_is_handled():
    edges, _ = dl.parse_edge_lines(["﻿1 2"])
    assert len(edges) == 1


# --------------------------------------------------------------------------------------
# Loading from disk / bytes
# --------------------------------------------------------------------------------------
SNAP_STYLE = (
    "# Directed graph (each unordered pair of nodes is saved twice)\n"
    "# Collaboration network of Arxiv General Relativity\n"
    "# FromNodeId\tToNodeId\n"
    "1\t2\n2\t1\n1\t3\n3\t1\n2\t3\n3\t2\n3\t3\n"
)


def test_load_gzip_file(tmp_path):
    path = tmp_path / "ca-test.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(SNAP_STYLE)
    edges, report = dl.load_edges_from_path(path)
    assert len(edges) == 3  # a triangle: 1-2, 1-3, 2-3
    assert report.self_loops_removed == 1
    assert report.duplicate_edges_removed == 3


def test_load_plain_text_file(tmp_path):
    path = tmp_path / "ca-test.txt"
    path.write_text(SNAP_STYLE, encoding="utf-8")
    edges, _ = dl.load_edges_from_path(path)
    assert len(edges) == 3


def test_gzip_is_detected_by_content_not_extension(tmp_path):
    path = tmp_path / "actually_gzip.txt"  # wrong extension on purpose
    path.write_bytes(gzip.compress(SNAP_STYLE.encode()))
    edges, _ = dl.load_edges_from_path(path)
    assert len(edges) == 3


def test_load_csv_file(tmp_path):
    path = tmp_path / "edges.csv"
    path.write_text("source,target\n1,2\n2,3\n", encoding="utf-8")
    edges, report = dl.load_edges_from_path(path)
    assert len(edges) == 2
    assert report.header_lines == 1


def test_missing_file_gives_helpful_error_without_leaking_path(tmp_path):
    missing = tmp_path / "private_folder" / "nope.txt.gz"
    with pytest.raises(dl.DatasetError) as info:
        dl.load_edges_from_path(missing)
    message = str(info.value)
    assert "nope.txt.gz" in message
    assert "private_folder" not in message
    assert str(tmp_path) not in message


def test_corrupt_gzip_raises_dataset_error(tmp_path):
    path = tmp_path / "broken.txt.gz"
    path.write_bytes(dl.GZIP_MAGIC + b"this is not a valid gzip stream")
    with pytest.raises(dl.DatasetError):
        dl.load_edges_from_path(path)


def test_file_without_valid_edges_raises(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("# only comments\n\n", encoding="utf-8")
    with pytest.raises(dl.DatasetError, match="No valid edges"):
        dl.load_edges_from_path(path)


def test_load_from_bytes_plain_and_gzip():
    plain = b"1 2\n2 3\n"
    assert len(dl.load_edges_from_bytes(plain, "up.txt")[0]) == 2
    assert len(dl.load_edges_from_bytes(gzip.compress(plain), "up.txt.gz")[0]) == 2


def test_load_from_bytes_errors():
    with pytest.raises(dl.DatasetError):
        dl.load_edges_from_bytes(b"", "empty.txt")
    with pytest.raises(dl.DatasetError):
        dl.load_edges_from_bytes(dl.GZIP_MAGIC + b"garbage", "bad.gz")
    with pytest.raises(dl.DatasetError):
        dl.load_edges_from_bytes(b"hello world\n", "text.txt")


def test_download_rejects_non_https_url(tmp_path):
    with pytest.raises(dl.DatasetError):
        dl.download_snap_dataset(tmp_path, url="http://example.com/x.gz")


def test_find_default_dataset(tmp_path):
    assert dl.find_default_dataset(tmp_path) is None
    (tmp_path / "ca-GrQc.txt").write_text("1 2\n")
    assert dl.find_default_dataset(tmp_path).name == "ca-GrQc.txt"
    (tmp_path / "ca-GrQc.txt.gz").write_bytes(b"x")
    assert dl.find_default_dataset(tmp_path).name == "ca-GrQc.txt.gz"  # compressed preferred


# --------------------------------------------------------------------------------------
# The bundled sample file
# --------------------------------------------------------------------------------------
def test_bundled_sample_loads_and_reports_cleaning():
    edges, report = dl.load_edges_from_path(dl.SAMPLE_PATH)
    assert report.unique_edges == 55
    assert report.self_loops_removed == 1  # the deliberate "5,5"
    assert report.duplicate_edges_removed == 2  # the deliberate "2,1" and "12,11"
    assert report.header_lines == 1
    assert set(edges["source"]).union(edges["target"]) >= {1, 35}
