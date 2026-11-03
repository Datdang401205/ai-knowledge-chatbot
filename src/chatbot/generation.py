"""Sinh câu trả lời có căn cứ (grounded generation) từ các chunk đã truy xuất.

Hai provider:

- ``extractive``: trích câu từ chính ngữ cảnh, ghép câu trả lời + trích dẫn.
  Không cần LLM, chạy offline, dùng để test pipeline và làm baseline.
- ``ollama``: gửi prompt có ràng buộc chặt tới LLM local.

Cả hai **không được tự thêm dữ kiện ngoài ngữ cảnh**. Nếu không có bằng chứng,
trả về ``NO_EVIDENCE_MESSAGE`` thay vì đoán.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from chatbot.config import NO_EVIDENCE_MESSAGE, RAG_SYSTEM_PROMPT
from chatbot.llm import LLMError, ask_llm
from chatbot.retrieval import Retriever
from chatbot.vector_store import SearchResult

_WORD_RE = re.compile(r"[0-9a-zA-ZÀ-ỹà-ỹđĐ]+", re.UNICODE)

_STOPWORDS = {
    "la", "gi", "cua", "va", "co", "the", "nao", "tai", "lieu", "cho", "toi",
    "trong", "duoc", "khi", "nhung", "mot", "cac", "nhu", "nay", "do", "thi", "ma", "de", "ra", "vao", "tren", "cung", "the", "is", "what",
    "how", "why", "the", "a", "an", "of", "to", "in", "and", "or", "for",
}


@dataclass(frozen=True)
class Source:
    """Nguồn được trích dẫn trong câu trả lời."""

    source: str
    page: int
    score: float
    chunk_id: str


@dataclass(frozen=True)
class Answer:
    """Kết quả cuối cùng của pipeline RAG."""

    text: str
    sources: tuple[Source, ...] = ()
    evidence_found: bool = False
    provider: str = ""
    retrieved: tuple[SearchResult, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.text,
            "evidence_found": self.evidence_found,
            "provider": self.provider,
            "sources": [
                {
                    "file": source.source,
                    "page": source.page,
                    "score": round(source.score, 4),
                    "chunk_id": source.chunk_id,
                }
                for source in self.sources
            ],
            "retrieved": [
                {
                    "file": result.chunk.source,
                    "page": result.chunk.page,
                    "score": round(result.score, 4),
                    "chunk_id": result.chunk.id,
                }
                for result in self.retrieved
            ],
        }


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("đ", "d").split())


def _terms(text: str) -> set[str]:
    words = {word for word in _WORD_RE.findall(_normalize(text)) if len(word) > 1}
    return words - _STOPWORDS


def _passages(text: str) -> list[str]:
    """Gộp dòng tiêu đề/dòng quá ngắn với các dòng ngay sau để câu trả lời có nghĩa.

    Ví dụ chunk có::

        Top-k ảnh hưởng trực tiếp:
        - Top-k nhỏ (1-2): nhanh, ít nhiễu...
        - Top-k lớn (8-10): tăng recall...

    Nếu tách từng dòng, ta chỉ trả về mỗi dòng tiêu đề. Gộp lại giúp extractive
    trả về đủ ý và dễ đọc hơn.
    """
    logical_lines: list[str] = []
    buffer = ""
    for line in text.split("\n"):
        cleaned = " ".join(line.split())
        if not cleaned:
            continue
        buffer = f"{buffer} {cleaned}".strip()
        # PDF ngắt dòng giữa câu; chỉ chốt dòng khi gặp dấu kết thúc câu/tiêu đề.
        if buffer.endswith((".", "!", "?", ":")):
            logical_lines.append(buffer)
            buffer = ""
    if buffer:
        logical_lines.append(buffer)

    sentences: list[str] = []
    for line in logical_lines:
        if len(line) < 20:
            continue
        sentences.extend(
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", line)
            if len(part.strip()) >= 20
        )

    passages: list[str] = []
    index = 0
    while index < len(sentences):
        sentence = sentences[index]
        if sentence.endswith(":"):
            # Dòng tiêu đề ("Top-k ảnh hưởng trực tiếp:") cần gộp với các ý sau.
            parts = [sentence]
            cursor = index + 1
            while cursor < len(sentences) and len(parts) < 3:
                parts.append(sentences[cursor])
                cursor += 1
            passages.append(" ".join(parts))
            index = cursor
        else:
            passages.append(sentence)
            index += 1
    return [_strip_leading_fragment(passage) for passage in passages]


def _strip_leading_fragment(passage: str) -> str:
    """Bỏ mẩu câu cụt ở đầu chunk (chunk bị cắt giữa câu).

    Ví dụ chunk bắt đầu bằng ``trước. FEFO bắt buộc...`` thì ``trước.`` là phần
    đuôi của câu trước, không phải câu trả lời.
    """
    match = re.match(r"^(\S+(?:\s+\S+){0,2})\.\s+", passage)
    if not match:
        return passage
    prefix = match.group(1)
    rest = passage[match.end() :]
    if len(prefix) <= 18 and rest[:1].isupper():
        return rest
    return passage


def _question_subject(question: str) -> str:
    """Lấy thuật ngữ nội dung đầu tiên làm chủ thể (vd: ``OTD`` trong câu hỏi)."""
    for word in _WORD_RE.findall(_normalize(question)):
        if len(word) > 1 and word not in _STOPWORDS:
            return word
    return ""


def _is_definition(subject: str, sentence: str) -> bool:
    """Heuristic nhận câu định nghĩa/giải thích về chủ thể."""
    if not subject:
        return False
    normalized = _normalize(sentence)
    position = normalized.find(subject)
    if position == -1 or position > 60:
        return False
    return " la " in normalized[position : position + 140]


def collect_sources(results: list[SearchResult]) -> tuple[Source, ...]:
    """Gộp nguồn trùng file+trang, giữ điểm cao nhất, sắp theo điểm giảm dần."""
    best: dict[tuple[str, int], Source] = {}
    for result in results:
        key = (result.chunk.source, result.chunk.page)
        current = best.get(key)
        if current is None or result.score > current.score:
            best[key] = Source(
                source=result.chunk.source,
                page=result.chunk.page,
                score=result.score,
                chunk_id=result.chunk.id,
            )
    return tuple(sorted(best.values(), key=lambda source: -source.score))


def build_rag_prompt(question: str, context: str) -> str:
    """Prompt đặt ngữ cảnh trước câu hỏi để LLM dễ đối chiếu."""
    return (
        "NGỮ CẢNH (chỉ được dùng thông tin dưới đây):\n"
        f"{context}\n\n"
        "CÂU HỎI:\n"
        f"{question}\n\n"
        "Hãy trả lời ngắn gọn, chính xác, và trích dẫn nguồn theo mẫu [file, trang N]."
    )


class Generator(Protocol):
    name: str

    def generate(self, question: str, results: list[SearchResult]) -> Answer: ...


class ExtractiveGenerator:
    """Baseline không dùng LLM: chọn các câu trong ngữ cảnh khớp câu hỏi nhất."""

    name = "extractive"

    def generate(self, question: str, results: list[SearchResult]) -> Answer:
        sources = collect_sources(results)
        if not results:
            return Answer(text=NO_EVIDENCE_MESSAGE, sources=(), evidence_found=False, provider=self.name)

        question_terms = _terms(question)
        subject = _question_subject(question)
        scored: list[tuple[float, int, str]] = []

        for result_index, result in enumerate(results):
            for sentence in _passages(result.chunk.text):
                sentence_terms = _terms(sentence)
                if not sentence_terms:
                    continue
                overlap = len(question_terms & sentence_terms)
                if overlap == 0:
                    continue
                coverage = overlap / max(1, len(question_terms))
                # Câu chứa đúng chủ thể được ưu tiên; câu định nghĩa được ưu tiên hơn.
                if _is_definition(subject, sentence):
                    subject_bonus = 0.6
                elif subject and subject in sentence_terms:
                    subject_bonus = 0.25
                else:
                    subject_bonus = 0.0
                score = (
                    coverage
                    + result.score * 0.5
                    + subject_bonus
                    - result_index * 0.08
                )
                scored.append((score, result_index, sentence))

        if not scored:
            return Answer(
                text=NO_EVIDENCE_MESSAGE,
                sources=sources,
                evidence_found=False,
                provider=self.name,
                retrieved=tuple(results),
            )

        # Ưu tiên câu chứa đúng chủ thể câu hỏi. Nếu không có câu nào, quay lại
        # toàn bộ ứng viên để không biến thành "không có bằng chứng" oan.
        focused = [
            item
            for item in scored
            if subject and (subject in _terms(item[2]) or _is_definition(subject, item[2]))
        ]
        if focused:
            scored = focused

        scored.sort(key=lambda item: -item[0])
        chosen: list[str] = []
        used_indices: set[int] = set()
        for _score, result_index, sentence in scored:
            if sentence in chosen:
                continue
            chosen.append(sentence)
            used_indices.add(result_index)
            if len(chosen) == 2:
                break

        # Chỉ trích dẫn nguồn thực sự được dùng để sinh câu trả lời.
        used_results = [results[index] for index in sorted(used_indices)]
        used_sources = collect_sources(used_results) if used_results else sources
        citations = "; ".join(
            f"[{source.source}, trang {source.page}]" for source in used_sources
        )
        body = " ".join(chosen)
        return Answer(
            text=f"Theo tài liệu: {body} (Nguồn: {citations})",
            sources=used_sources,
            evidence_found=True,
            provider=self.name,
            retrieved=tuple(results),
        )


class OllamaGenerator:
    """Sinh câu trả lời bằng LLM local, ràng buộc bởi ``RAG_SYSTEM_PROMPT``."""

    name = "ollama"

    def generate(self, question: str, results: list[SearchResult]) -> Answer:
        sources = collect_sources(results)
        if not results:
            return Answer(text=NO_EVIDENCE_MESSAGE, sources=(), evidence_found=False, provider=self.name)

        context = Retriever.format_context(results)
        prompt = build_rag_prompt(question, context)
        try:
            text = ask_llm(prompt, system_prompt=RAG_SYSTEM_PROMPT)
        except LLMError as exc:
            return Answer(
                text=f"Không gọi được LLM ({exc}). {NO_EVIDENCE_MESSAGE}",
                sources=sources,
                evidence_found=False,
                provider=self.name,
                retrieved=tuple(results),
            )

        # LLM có thể quên trích dẫn; ta luôn gắn nguồn đã dùng để không mất truy vết.
        if "[" not in text and sources:
            citations = "; ".join(
                f"[{source.source}, trang {source.page}]" for source in sources
            )
            text = f"{text.rstrip()} (Nguồn: {citations})"

        return Answer(
            text=text,
            sources=sources,
            evidence_found=True,
            provider=self.name,
            retrieved=tuple(results),
        )


def get_generator(provider: str | None = None) -> Generator:
    """Chọn provider theo ``LLM_PROVIDER`` (extractive|ollama)."""
    from chatbot.config import LLM_PROVIDER

    chosen = (provider or LLM_PROVIDER).lower()
    if chosen == "extractive":
        return ExtractiveGenerator()
    if chosen == "ollama":
        return OllamaGenerator()
    raise ValueError(f"LLM_PROVIDER không hợp lệ: {chosen} (chọn extractive|ollama)")
