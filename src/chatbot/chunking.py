"""Chia văn bản thành các đoạn (chunk) nhỏ, giữ metadata nguồn.

Vì sao phải chunk?
------------------
LLM chỉ có một *context window* hữu hạn. Nếu nhét cả tài liệu 100 trang vào prompt:

1. Tốn rất nhiều token (chậm, đắt, dễ vượt giới hạn).
2. Thông tin không liên quan làm loãng câu trả lời.
3. Không thể trích dẫn chính xác đoạn nào chứa câu trả lời.

Chunk quá nhỏ thì mất ngữ cảnh; quá lớn thì nhiễu. Vì vậy kích thước và độ
chồng lấn (overlap) là hai tham số phải đo bằng thực nghiệm (Tuần 5).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from chatbot.config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_CHARS

# Ưu tiên cắt ở ranh giới tự nhiên: đoạn văn > câu > khoảng trắng.
_BREAK_SEPARATORS = ("\n\n", ". ", ".\n", "! ", "? ", "; ", "\n", " ")


@dataclass(frozen=True)
class PageText:
    """Văn bản của một trang đã trích xuất từ tài liệu."""

    source: str
    page: int
    text: str
    has_text: bool = True


@dataclass(frozen=True)
class Chunk:
    """Một đoạn văn bản kèm metadata để truy vết nguồn."""

    id: str
    text: str
    source: str
    page: int
    chunk_index: int
    start_char: int
    end_char: int

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Chunk":
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            source=str(data["source"]),
            page=int(data["page"]),  # type: ignore[arg-type]
            chunk_index=int(data["chunk_index"]),  # type: ignore[arg-type]
            start_char=int(data["start_char"]),  # type: ignore[arg-type]
            end_char=int(data["end_char"]),  # type: ignore[arg-type]
        )


def _make_chunk_id(source: str, page: int, index: int, start: int) -> str:
    """ID ổn định: cùng tài liệu + cùng tham số chunking → cùng ID."""
    raw = f"{source}|{page}|{index}|{start}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _find_break(text: str, start: int, hard_end: int) -> int:
    """Tìm vị trí cắt đẹp nhất trong nửa sau của cửa sổ.

    Trả về ``hard_end`` nếu không tìm thấy ranh giới nào tốt hơn.
    """
    lower_bound = start + max(1, (hard_end - start) // 2)
    best = hard_end
    for separator in _BREAK_SEPARATORS:
        position = text.rfind(separator, lower_bound, hard_end)
        if position != -1:
            return position + len(separator)
    return best


def chunk_page(
    page: PageText,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[Chunk]:
    """Chia văn bản của một trang thành các chunk có overlap.

    Args:
        page: Trang cần chia.
        chunk_size: Số ký tự mục tiêu cho mỗi chunk (phải > 0).
        overlap: Số ký tự lặp lại giữa hai chunk liền nhau (0 <= overlap < chunk_size).
        min_chars: Chunk ngắn hơn ngưỡng này bị bỏ (trừ khi là chunk duy nhất).

    Raises:
        ValueError: Nếu tham số không hợp lệ.

    Returns:
        Danh sách chunk theo thứ tự xuất hiện trong trang.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size phải lớn hơn 0")
    if overlap < 0:
        raise ValueError("overlap không được âm")
    if overlap >= chunk_size:
        raise ValueError("overlap phải nhỏ hơn chunk_size")

    text = page.text
    if not text or not text.strip():
        return []

    chunks: list[Chunk] = []
    start = 0
    length = len(text)
    index = 0

    while start < length:
        hard_end = min(start + chunk_size, length)
        end = hard_end if hard_end == length else _find_break(text, start, hard_end)
        piece = text[start:end].strip()

        if len(piece) >= min_chars or (not chunks and piece):
            chunks.append(
                Chunk(
                    id=_make_chunk_id(page.source, page.page, index, start),
                    text=piece,
                    source=page.source,
                    page=page.page,
                    chunk_index=index,
                    start_char=start,
                    end_char=end,
                )
            )
            index += 1

        if end >= length:
            break

        # Bước nhảy phải > 0 để không lặp vô hạn, kể cả khi overlap lớn.
        next_start = end - overlap
        start = max(next_start, start + 1)

    return chunks


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[Chunk]:
    """Chia nhiều trang, giữ nguyên thứ tự tài liệu → trang → chunk."""
    result: list[Chunk] = []
    for page in pages:
        result.extend(chunk_page(page, chunk_size, overlap, min_chars))
    return result
