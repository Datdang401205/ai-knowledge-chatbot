"""Kiểm thử bộ đo đánh giá."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chatbot.embeddings import HashingEmbedder
from chatbot.evaluation import load_questions, run_evaluation
from chatbot.rag_pipeline import RAGPipeline
from chatbot.retrieval import RetrievalConfig
from chatbot.vector_store import VectorStore


def make_pipeline(tmp_path: Path, docs: Path) -> RAGPipeline:
    pipeline = RAGPipeline(
        VectorStore(),
        embedder=HashingEmbedder(dimension=512),
        config=RetrievalConfig(top_k=3, min_score=0.2),
        index_path=tmp_path / "index" / "store.json",
        manifest_path=tmp_path / "index" / "manifest.json",
    )
    pipeline.build_index(docs, chunk_size=200, overlap=20, persist=False)
    return pipeline


def write_questions(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "category": "factual",
                    "question": "RAG là gì?",
                    "expected_sources": [{"file": "rag.txt", "page": 1}],
                    "reference_answer": "RAG kết hợp truy xuất tài liệu với mô hình ngôn ngữ.",
                },
                {
                    "id": "u1",
                    "category": "unanswerable",
                    "question": "Giá vàng hôm nay là bao nhiêu?",
                    "expected_sources": [],
                    "reference_answer": "",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_load_questions_rejects_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="không rỗng"):
        load_questions(path)


def test_load_questions_rejects_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([{"id": "q1"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="thiếu trường"):
        load_questions(path)


def test_run_evaluation_computes_hit_and_abstention(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rag.txt").write_text(
        "RAG là kiến trúc kết hợp truy xuất tài liệu với mô hình ngôn ngữ. "
        "RAG trả lời dựa trên tài liệu và có dẫn nguồn.",
        encoding="utf-8",
    )
    questions = tmp_path / "questions.json"
    write_questions(questions)

    result = run_evaluation(
        eval_path=questions,
        top_k=3,
        chunk_size=200,
        overlap=20,
        pipeline=make_pipeline(tmp_path, docs),
    )

    assert result.summary["questions"] == 2
    assert result.summary["answerable"] == 1
    assert result.summary["unanswerable"] == 1
    assert result.summary["hit_rate_at_k"] == 1.0
    assert result.summary["abstention_accuracy"] == 1.0
    assert result.summary["errors"] == 0
    assert len(result.questions) == 2
