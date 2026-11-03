"""Kiểm thử truy xuất hybrid (vector + từ khóa)."""

from __future__ import annotations

from chatbot.chunking import Chunk
from chatbot.embeddings import HashingEmbedder
from chatbot.retrieval import RetrievalConfig, Retriever
from chatbot.vector_store import VectorStore


def make_chunk(identifier: str, text: str, page: int = 1) -> Chunk:
    return Chunk(
        id=identifier,
        text=text,
        source="tai_lieu.pdf",
        page=page,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
    )


def build_retriever() -> Retriever:
    embedder = HashingEmbedder(dimension=512)
    store = VectorStore(dimension=512)
    chunks = [
        make_chunk(
            "otd",
            "OTD là tỷ lệ đơn giao đúng hạn. Công thức chia số đơn đúng hạn cho tổng đơn.",
            1,
        ),
        make_chunk(
            "kho",
            "Quy trình kho gồm receiving, putaway, storage, picking, packing và shipping.",
            2,
        ),
        make_chunk(
            "rag",
            "RAG kết hợp truy xuất tài liệu với mô hình ngôn ngữ để trả lời có dẫn nguồn.",
            3,
        ),
    ]
    store.add(chunks, embedder.embed_texts([chunk.text for chunk in chunks]))
    return Retriever(
        store,
        embedder,
        RetrievalConfig(top_k=3, min_score=0.2),
    )


def test_retrieve_finds_chunk_with_rare_term() -> None:
    results = build_retriever().retrieve("OTD được tính thế nào?")

    assert results
    assert results[0].chunk.id == "otd"


def test_retrieve_ranks_warehouse_question_correctly() -> None:
    results = build_retriever().retrieve("Quy trình kho gồm những bước nào?")

    assert results[0].chunk.id == "kho"


def test_unknown_entity_abstains() -> None:
    assert build_retriever().retrieve("Mô hình GPT-5 dùng bao nhiêu tham số?") == []


def test_empty_question_returns_nothing() -> None:
    assert build_retriever().retrieve("   ") == []


def test_has_evidence_matches_result_list() -> None:
    retriever = build_retriever()
    assert retriever.has_evidence(retriever.retrieve("RAG là gì?")) is True
    assert retriever.has_evidence([]) is False


def test_format_context_includes_source_and_page() -> None:
    results = build_retriever().retrieve("RAG là gì?")
    context = Retriever.format_context(results)

    assert "[ĐOẠN 1]" in context
    assert "tai_lieu.pdf, trang 3" in context
    assert "điểm liên quan" in context
