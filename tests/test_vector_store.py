"""Kiểm thử vector store tối giản."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chatbot.chunking import Chunk
from chatbot.vector_store import SearchResult, VectorStore, VectorStoreError


def make_chunk(identifier: str, text: str = "nội dung") -> Chunk:
    return Chunk(
        id=identifier,
        text=text,
        source="a.pdf",
        page=1,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
    )


def test_add_and_search_returns_best_match_first() -> None:
    store = VectorStore(dimension=3)
    chunks = [make_chunk("a"), make_chunk("b")]
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    assert store.add(chunks, vectors) == 2
    results = store.search(np.array([0.9, 0.1, 0.0]), top_k=2)

    assert isinstance(results[0], SearchResult)
    assert results[0].chunk.id == "a"
    assert results[0].score > results[1].score


def test_add_ignores_duplicate_ids() -> None:
    store = VectorStore(dimension=2)
    chunk = make_chunk("a")

    assert store.add([chunk], np.array([[1.0, 0.0]])) == 1
    assert store.add([chunk], np.array([[1.0, 0.0]])) == 0
    assert len(store) == 1


def test_add_rejects_length_mismatch() -> None:
    store = VectorStore(dimension=2)
    with pytest.raises(VectorStoreError, match="Số chunk"):
        store.add([make_chunk("a")], np.array([[1.0, 0.0], [0.0, 1.0]]))


def test_add_rejects_wrong_dimension() -> None:
    store = VectorStore(dimension=3)
    with pytest.raises(VectorStoreError, match="chiều"):
        store.add([make_chunk("a")], np.array([[1.0, 0.0]]))


def test_search_filters_by_min_score() -> None:
    store = VectorStore(dimension=2)
    store.add([make_chunk("a"), make_chunk("b")], np.array([[1.0, 0.0], [0.0, 1.0]]))

    results = store.search(np.array([1.0, 0.0]), top_k=2, min_score=0.5)

    assert [result.chunk.id for result in results] == ["a"]


def test_search_handles_empty_store_and_zero_vector() -> None:
    store = VectorStore(dimension=2)
    assert store.search(np.array([1.0, 0.0])) == []

    store.add([make_chunk("a")], np.array([[1.0, 0.0]]))
    assert store.search(np.array([0.0, 0.0])) == []
    assert store.search(np.array([1.0, 0.0]), top_k=0) == []


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "store.json"
    store = VectorStore(path=path, dimension=2)
    store.add([make_chunk("a", "xin chào")], np.array([[1.0, 0.0]]), "hashing")
    store.save()

    loaded = VectorStore.load(path)

    assert len(loaded) == 1
    assert loaded.chunks[0].text == "xin chào"
    assert loaded.embedding_provider == "hashing"
    assert loaded.stats()["dimension"] == 2


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(VectorStoreError, match="Chưa có vector store"):
        VectorStore.load(tmp_path / "khong_co.json")


def test_save_without_path_raises() -> None:
    with pytest.raises(VectorStoreError, match="Chưa cấu hình"):
        VectorStore(dimension=2).save()
