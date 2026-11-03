"""Biến văn bản thành vector (embeddings).

Có hai backend, cùng một interface:

1. ``HashingEmbedder`` (mặc định, offline)
   Băm từ và cụm 3 ký tự vào một vector cố định, chuẩn hoá L2.
   - Ưu điểm: chạy ngay, không cần tải model, kết quả tất định → test ổn định.
   - Nhược điểm: **không hiểu ngữ nghĩa đồng nghĩa** ("ô tô" ≠ "xe hơi").
   Đây là baseline để học cơ chế, không phải chất lượng production.

2. ``OllamaEmbedder`` (khuyến nghị khi có Ollama)
   Gọi ``nomic-embed-text`` hoặc model embedding đa ngôn ngữ khác.
   Hiểu ngữ nghĩa tốt hơn nhiều nhưng cần tải model và chạy local.

Đổi backend bằng biến môi trường ``EMBEDDING_PROVIDER=hashing|ollama``.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from typing import Protocol, Sequence

import numpy as np
import requests

from chatbot.config import (
    HASHING_EMBEDDING_DIM,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
)

_WORD_RE = re.compile(r"[0-9a-zA-ZÀ-ỹà-ỹđĐ]+", re.UNICODE)


class EmbeddingError(Exception):
    """Không tạo được embedding."""


class Embedder(Protocol):
    """Interface tối thiểu để retrieval không phụ thuộc backend cụ thể."""

    dimension: int

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(text: str) -> str:
    """Chữ thường, bỏ dấu câu, gộp khoảng trắng. Giữ dấu tiếng Việt để không mất nghĩa."""
    lowered = text.lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", lowered)
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(without_marks.split())


def _hash_feature(feature: str, dimension: int) -> tuple[int, float]:
    """Băm một feature thành (chỉ số, dấu). Dấu giúp giảm va chạm có hệ thống."""
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    index = value % dimension
    sign = 1.0 if (value >> 63) & 1 else -1.0
    return index, sign


class HashingEmbedder:
    """Baseline offline: bag-of-features + hashing trick + chuẩn hoá L2."""

    def __init__(
        self,
        dimension: int = HASHING_EMBEDDING_DIM,
        char_ngram_size: int = 3,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension phải lớn hơn 0")
        self.dimension = dimension
        self.char_ngram_size = char_ngram_size

    def _features(self, text: str) -> dict[str, float]:
        normalized = _normalize(text)
        words = _WORD_RE.findall(normalized)
        features: dict[str, float] = {}

        for word in words:
            # Từ nguyên vẹn có trọng số cao nhất: câu hỏi chứa đúng thuật ngữ
            # ("OTD", "FastAPI") phải khớp mạnh với chunk chứa thuật ngữ đó.
            features[f"w:{word}"] = features.get(f"w:{word}", 0.0) + 1.0
            padded = f"<{word}>"
            size = self.char_ngram_size
            for start in range(max(0, len(padded) - size + 1)):
                gram = padded[start : start + size]
                features[f"c:{gram}"] = features.get(f"c:{gram}", 0.0) + 0.15

        # Bigram từ giúp phân biệt cụm như "hit rate" với hai từ rời.
        for left, right in zip(words, words[1:]):
            features[f"b:{left}_{right}"] = (
                features.get(f"b:{left}_{right}", 0.0) + 0.8
            )
        return features

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, count in self._features(text).items():
                index, sign = _hash_feature(feature, self.dimension)
                matrix[row, index] += sign * (1.0 + math.log(count))
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


class OllamaEmbedder:
    """Gọi Ollama để tạo embedding ngữ nghĩa (ví dụ ``nomic-embed-text``)."""

    def __init__(
        self,
        model: str = OLLAMA_EMBED_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout: float = OLLAMA_TIMEOUT_SECONDS,
        dimension: int | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.dimension = dimension or 0

    def _request(self, text: str) -> list[float]:
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as exc:
            raise EmbeddingError(f"Không gọi được Ollama embeddings: {exc}") from exc

        if response.status_code != 200:
            raise EmbeddingError(
                f"Ollama embeddings lỗi HTTP {response.status_code}: {response.text[:200]}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise EmbeddingError("Ollama embeddings trả về không phải JSON.") from exc

        vector = data.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise EmbeddingError("Ollama embeddings trả về vector rỗng.")
        return [float(value) for value in vector]

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension or 1), dtype=np.float32)
        vectors = [self._request(text) for text in texts]
        matrix = np.asarray(vectors, dtype=np.float32)
        if self.dimension == 0:
            self.dimension = int(matrix.shape[1])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


def get_embedder(provider: str | None = None) -> Embedder:
    """Tạo embedder theo cấu hình. Mặc định ``hashing`` để chạy offline."""
    from chatbot.config import EMBEDDING_PROVIDER

    chosen = (provider or EMBEDDING_PROVIDER).lower()
    if chosen == "hashing":
        return HashingEmbedder()
    if chosen == "ollama":
        return OllamaEmbedder()
    raise ValueError(f"EMBEDDING_PROVIDER không hợp lệ: {chosen} (chọn hashing|ollama)")
