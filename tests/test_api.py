"""Kiểm thử HTTP API bằng TestClient."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chatbot.api import create_app
from chatbot.embeddings import HashingEmbedder
from chatbot.rag_pipeline import RAGPipeline
from chatbot.retrieval import RetrievalConfig
from chatbot.vector_store import VectorStore


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rag.txt").write_text(
        "RAG là kiến trúc kết hợp truy xuất tài liệu với mô hình ngôn ngữ. "
        "RAG trả lời dựa trên tài liệu và có dẫn nguồn.",
        encoding="utf-8",
    )
    pipeline = RAGPipeline(
        VectorStore(),
        embedder=HashingEmbedder(dimension=512),
        config=RetrievalConfig(top_k=3, min_score=0.2),
        index_path=tmp_path / "index" / "store.json",
        manifest_path=tmp_path / "index" / "manifest.json",
    )
    pipeline.build_index(docs, chunk_size=700, overlap=120, persist=False)
    return TestClient(create_app(pipeline))


def test_health_reports_index_size(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["chunks"] == 1


def test_ask_returns_answer_and_sources(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "RAG là gì?"})

    assert response.status_code == 200
    body = response.json()
    assert body["evidence_found"] is True
    assert "RAG" in body["answer"]
    assert body["sources"][0]["file"] == "rag.txt"


def test_ask_rejects_empty_question(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422


def test_ask_abstains_for_unknown_topic(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "Giá cổ phiếu hôm nay?"})

    assert response.status_code == 200
    assert response.json()["evidence_found"] is False


def test_upload_adds_document_and_can_be_queried(client: TestClient) -> None:
    content = "Quy trình kho gồm receiving, putaway, storage, picking và shipping."
    response = client.post(
        "/documents",
        files={"file": ("kho.txt", content.encode("utf-8"), "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["chunks"] == 1

    answer = client.post("/ask", json={"question": "Quy trình kho gồm gì?"})
    assert answer.status_code == 200
    assert answer.json()["evidence_found"] is True
    assert any(source["file"] == "kho.txt" for source in answer.json()["sources"])


def test_upload_rejects_unsupported_extension(client: TestClient) -> None:
    response = client.post(
        "/documents",
        files={"file": ("data.csv", b"a,b", "text/csv")},
    )

    assert response.status_code == 415
