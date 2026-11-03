"""Kiểm thử chia văn bản thành chunk."""

from __future__ import annotations

from chatbot.chunking import PageText, chunk_page, chunk_pages


def test_chunk_page_keeps_source_and_page() -> None:
    page = PageText(source="a.pdf", page=3, text="Câu một. " * 200)
    chunks = chunk_page(page, chunk_size=200, overlap=40)

    assert len(chunks) > 1
    assert all(chunk.source == "a.pdf" for chunk in chunks)
    assert all(chunk.page == 3 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_stable_between_runs() -> None:
    page = PageText(source="a.pdf", page=1, text="Nội dung ổn định. " * 100)
    first = chunk_page(page, chunk_size=150, overlap=30)
    second = chunk_page(page, chunk_size=150, overlap=30)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]


def test_chunk_page_ignores_blank_page() -> None:
    page = PageText(source="scan.pdf", page=1, text="   \n  ", has_text=False)
    assert chunk_page(page, chunk_size=200, overlap=20) == []


def test_chunk_pages_covers_every_non_empty_page() -> None:
    pages = [
        PageText(source="a.pdf", page=1, text="quá ngắn"),
        PageText(source="a.pdf", page=2, text="Nội dung đủ dài để tạo chunk. " * 40),
    ]
    chunks = chunk_pages(pages, chunk_size=200, overlap=20)

    assert {chunk.page for chunk in chunks} == {1, 2}


def test_chunk_overlap_brings_neighbour_text() -> None:
    text = "A" * 300 + " " + "B" * 300
    chunks = chunk_page(PageText(source="a.txt", page=1, text=text), chunk_size=200, overlap=50)

    assert len(chunks) >= 3
    assert chunks[1].start_char < chunks[0].end_char
