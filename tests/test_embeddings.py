"""Kiểm thử embedder offline."""

from __future__ import annotations

import numpy as np
import pytest

from chatbot.embeddings import EmbeddingError, HashingEmbedder, get_embedder


def test_embedding_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dimension=256)
    first = embedder.embed_query("RAG là gì?")
    second = embedder.embed_query("RAG là gì?")

    assert first.shape == (256,)
    assert np.allclose(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0)


def test_embed_texts_returns_matrix_with_one_row_per_text() -> None:
    embedder = HashingEmbedder(dimension=128)
    matrix = embedder.embed_texts(["một", "hai", "ba"])

    assert matrix.shape == (3, 128)


def test_similar_text_scores_higher_than_unrelated_text() -> None:
    embedder = HashingEmbedder(dimension=512)
    query = embedder.embed_query("Top-k ảnh hưởng tới recall")
    similar = embedder.embed_query("Top-k lớn giúp tăng recall")
    unrelated = embedder.embed_query("Quy trình nhập hàng vào kho")

    assert float(query @ similar) > float(query @ unrelated)


def test_empty_text_returns_zero_vector() -> None:
    embedder = HashingEmbedder(dimension=64)
    assert not np.any(embedder.embed_query("   "))


def test_invalid_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="dimension"):
        HashingEmbedder(dimension=0)


def test_get_embedder_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        get_embedder("khong_ton_tai")


def test_embedding_error_is_public_type() -> None:
    assert issubclass(EmbeddingError, Exception)
