"""Truy xuất các đoạn tài liệu liên quan nhất tới câu hỏi.

Retrieval là bước *tìm bằng chứng*. Generation là bước *diễn đạt bằng chứng*.
Tách hai bước giúp debug: nếu retrieval sai, đừng đổ lỗi cho LLM.

Chiến lược ở đây là **hybrid**:
- Điểm vector (cosine) bắt được tương đồng ngữ nghĩa.
- Điểm từ khóa (lexical coverage) bắt được thuật ngữ hiếm như "OTD", "FastAPI".

Điểm cuối = 0.6 × cosine + 0.4 × coverage. Với embedding hashing (không hiểu
ngữ nghĩa), phần lexical giúp baseline vẫn dùng được. Khi thay bằng embedding
thật, có thể tăng trọng số cosine.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

from chatbot.config import MIN_RELEVANCE_SCORE, TOP_K
from chatbot.embeddings import Embedder, get_embedder
from chatbot.vector_store import SearchResult, VectorStore

_WORD_RE = re.compile(r"[0-9a-zA-ZÀ-ỹà-ỹđĐ]+", re.UNICODE)
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b")

_STOPWORDS = {
    "la", "gi", "cua", "va", "co", "the", "nao", "tai", "lieu", "cho", "toi",
    "trong", "duoc", "khi", "nhung", "mot", "cac", "nhu", "nay", "do", "thi",
    "ma", "de", "ra", "vao", "tren", "duoi", "bang", "voi", "hon", "cung",
    "se", "da", "dang", "hoac", "nhieu", "qua", "lai", "moi", "chi", "tu",
    "hay", "phai", "lam", "giup", "is", "what", "how", "why", "the", "a", "an",
    "of", "to", "in", "and", "or", "for", "which", "does", "with", "that",
    "this", "it", "are", "was", "be", "by", "on", "as",
}


def _normalize(text: str) -> str:
    lowered = text.lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", lowered)
    return " ".join(
        "".join(char for char in decomposed if not unicodedata.combining(char)).split()
    )


def _content_terms(text: str) -> set[str]:
    return {
        word
        for word in _WORD_RE.findall(_normalize(text))
        if len(word) > 1 and word not in _STOPWORDS
    }


@dataclass(frozen=True)
class RetrievalConfig:
    """Tham số ảnh hưởng trực tiếp tới chất lượng và độ trễ."""

    top_k: int = TOP_K
    min_score: float = MIN_RELEVANCE_SCORE
    vector_weight: float = 0.5
    lexical_weight: float = 0.5


class Retriever:
    """Ghép embedder với vector store để tìm top-k chunk."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder or get_embedder()
        self.config = config or RetrievalConfig()

    def _rerank(
        self,
        question: str,
        candidates: list[SearchResult],
        top_k: int,
        min_score: float,
    ) -> list[SearchResult]:
        query_terms = _content_terms(question)

        # IDF: thuật ngữ hiếm ("OTD") quan trọng hơn từ phổ biến ("tính").
        # Không có IDF, một từ chung xuất hiện ở nhiều chunk sẽ kéo điểm lệch.
        all_terms = [_content_terms(chunk.text) for chunk in self.store.chunks]
        document_count = max(1, len(all_terms))
        idf: dict[str, float] = {}
        for term in query_terms:
            df = sum(1 for terms in all_terms if term in terms)
            idf[term] = math.log((document_count + 1) / (df + 1)) + 1.0
        total_idf = sum(idf.values()) or 1.0

        scored: list[SearchResult] = []
        for candidate in candidates:
            chunk_terms = _content_terms(candidate.chunk.text)
            weighted_match = sum(
                weight for term, weight in idf.items() if term in chunk_terms
            )
            coverage = weighted_match / total_idf if query_terms else 0.0
            combined = (
                self.config.vector_weight * candidate.score
                + self.config.lexical_weight * coverage
            )
            scored.append(SearchResult(chunk=candidate.chunk, score=combined))

        scored.sort(key=lambda item: -item.score)
        return [item for item in scored if item.score >= min_score][:top_k]

    def _has_unknown_entity(self, question: str) -> bool:
        """Phát hiện tên riêng/viết tắt không xuất hiện ở bất kỳ chunk nào.

        Ví dụ hỏi "mô hình GPT-5" trong khi tài liệu không hề nhắc GPT-5: dù các
        từ chung ("mô hình") khớp, ta vẫn phải từ chối thay vì trả lời lan man.
        """
        entities = _ENTITY_RE.findall(question)
        if not entities:
            return False
        corpus = " ".join(_normalize(chunk.text) for chunk in self.store.chunks)
        return any(_normalize(entity) not in corpus for entity in entities)

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Trả về danh sách chunk liên quan, đã lọc theo ngưỡng điểm.

        ``top_k`` lớn hơn → nhiều bằng chứng hơn nhưng prompt dài hơn, dễ nhiễu.
        ``min_score`` cao hơn → ít bịa hơn nhưng dễ bỏ sót câu trả lời đúng.
        """
        question = (question or "").strip()
        if not question:
            return []
        if self._has_unknown_entity(question):
            return []

        k = top_k if top_k is not None else self.config.top_k
        threshold = min_score if min_score is not None else self.config.min_score
        vector = self.embedder.embed_query(question)

        # Lấy nhiều ứng viên hơn top_k để bước rerank có lựa chọn.
        candidates = self.store.search(
            vector,
            top_k=max(k * 3, 12),
            min_score=None,
        )
        return self._rerank(question, candidates, k, threshold)

    @staticmethod
    def has_evidence(results: list[SearchResult]) -> bool:
        """Có ít nhất một chunk vượt ngưỡng liên quan hay không."""
        return bool(results)

    @staticmethod
    def format_context(results: list[SearchResult]) -> str:
        """Định dạng ngữ cảnh cho prompt, mỗi đoạn kèm nhãn nguồn rõ ràng."""
        blocks: list[str] = []
        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            blocks.append(
                f"[ĐOẠN {index}] Nguồn: {chunk.source}, trang {chunk.page} "
                f"(điểm liên quan: {result.score:.3f})\n{chunk.text}"
            )
        return "\n\n".join(blocks)
