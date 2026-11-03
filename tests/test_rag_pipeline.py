"""Kiểm thử luồng RAG end-to-end trên tài liệu tạm."""

from __future__ import annotations

from pathlib import Path

import pytest

from chatbot.embeddings import HashingEmbedder
from chatbot.rag_pipeline import RAGPipeline
from chatbot.retrieval import RetrievalConfig
from chatbot.vector_store import VectorStore


def make_pipeline(tmp_path: Path) -> RAGPipeline:
    return RAGPipeline(
        VectorStore(),
        embedder=HashingEmbedder(dimension=512),
        config=RetrievalConfig(top_k=3, min_score=0.2),
        index_path=tmp_path / "index" / "store.json",
        manifest_path=tmp_path / "index" / "manifest.json",
    )


def write_docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rag.txt").write_text(
        "RAG là kiến trúc kết hợp truy xuất tài liệu với mô hình ngôn ngữ. "
        "RAG giúp trả lời dựa trên tài liệu và có dẫn nguồn.",
        encoding="utf-8",
    )
    (docs / "kho.md").write_text(
        "# Quy trình kho\n\nQuy trình kho gồm receiving, putaway, storage, "
        "picking, packing và shipping.",
        encoding="utf-8",
    )
    return docs


def test_build_index_reports_documents_and_chunks(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path)
    report = pipeline.build_index(write_docs(tmp_path), chunk_size=200, overlap=20, persist=False)

    assert report.documents == 2
    assert report.chunks >= 2
    assert report.warnings == ()
    assert pipeline.stats()["source_count"] == 2


def test_ask_returns_answer_with_sources(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path)
    pipeline.build_index(write_docs(tmp_path), chunk_size=200, overlap=20, persist=False)

    answer = pipeline.ask("RAG là gì?")

    assert answer.evidence_found is True
    assert "RAG" in answer.text
    assert answer.sources
    assert answer.sources[0].source == "rag.txt"


def test_ask_abstains_for_unknown_topic(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path)
    pipeline.build_index(write_docs(tmp_path), chunk_size=200, overlap=20, persist=False)

    answer = pipeline.ask("Giá cổ phiếu hôm nay là bao nhiêu?")

    assert answer.evidence_found is False
    assert answer.sources == ()


def test_ask_on_empty_index_abstains(tmp_path: Path) -> None:
    answer = make_pipeline(tmp_path).ask("RAG là gì?")

    assert answer.evidence_found is False


def test_ask_rejects_empty_question(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="không được rỗng"):
        make_pipeline(tmp_path).ask("   ")


def test_persist_and_reload_index(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path)
    pipeline.build_index(write_docs(tmp_path), chunk_size=200, overlap=20, persist=True)

    reloaded = RAGPipeline.from_index(
        index_path=tmp_path / "index" / "store.json",
        manifest_path=tmp_path / "index" / "manifest.json",
        embedder=HashingEmbedder(dimension=512),
        config=RetrievalConfig(top_k=3, min_score=0.2),
    )

    assert len(reloaded.store) == len(pipeline.store)
    assert reloaded.ask("Quy trình kho gồm gì?").evidence_found is True


def test_add_document_is_idempotent(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rag.txt").write_text("RAG là kiến trúc kết hợp truy xuất tài liệu.", encoding="utf-8")
    pipeline.build_index(docs, chunk_size=700, overlap=120, persist=False)

    extra = tmp_path / "kho.txt"
    extra.write_text(
        "Quy trình kho gồm receiving, putaway, storage, picking, packing và shipping.",
        encoding="utf-8",
    )
    first = pipeline.add_document(extra)
    second = pipeline.add_document(extra)

    assert first.chunks > 0
    assert second.chunks == 0
