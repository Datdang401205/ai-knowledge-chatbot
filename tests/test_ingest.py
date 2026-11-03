"""Kiểm thử đọc tài liệu và chuẩn hoá văn bản."""

from __future__ import annotations

from pathlib import Path

import pytest

from chatbot.ingest import (
    IngestError,
    clean_text,
    discover_documents,
    load_pages,
)


def test_clean_text_normalizes_pdf_artifacts() -> None:
    raw = "Top\xadk\xa0mặc định\r\nlà 4\r"
    assert clean_text(raw) == "Top-k mặc định\nlà 4\n"


def test_load_pages_reads_utf8_text(tmp_path: Path) -> None:
    path = tmp_path / "ghi_chu.txt"
    path.write_text("Xin chào, đây là tài liệu.", encoding="utf-8")

    pages = load_pages(path)

    assert len(pages) == 1
    assert pages[0].page == 1
    assert pages[0].source == "ghi_chu.txt"
    assert pages[0].has_text is True


def test_load_pages_reads_markdown(tmp_path: Path) -> None:
    path = tmp_path / "huong_dan.md"
    path.write_text("# Tiêu đề\n\nNội dung.", encoding="utf-8")

    pages = load_pages(path)

    assert "# Tiêu đề" in pages[0].text


def test_load_pages_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a,b", encoding="utf-8")

    with pytest.raises(IngestError, match="chưa hỗ trợ"):
        load_pages(path)


def test_load_pages_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="Không tìm thấy"):
        load_pages(tmp_path / "khong_ton_tai.txt")


def test_discover_documents_sorts_supported_files(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("x", encoding="utf-8")

    names = [path.name for path in discover_documents(tmp_path)]

    assert names == ["a.md", "b.txt"]


def test_discover_documents_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert discover_documents(tmp_path / "khong_co") == []
