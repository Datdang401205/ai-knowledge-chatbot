"""Đọc tài liệu và trích xuất văn bản theo từng trang.

Hỗ trợ:
- ``.pdf``  : dùng PyMuPDF (``fitz``), giữ số trang để trích dẫn.
- ``.txt``  : đọc UTF-8, coi cả file là một "trang" số 1.
- ``.md``   : giống ``.txt``.

Điểm quan trọng: **không im lặng bỏ qua lỗi**. PDF scan (ảnh) không có lớp text
sẽ được đánh dấu ``has_text=False`` để tầng trên cảnh báo người dùng, thay vì
trả về một kho tri thức rỗng mà không ai biết vì sao.
"""

from __future__ import annotations

import logging
from pathlib import Path

from chatbot.chunking import Chunk, PageText, chunk_pages

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class IngestError(Exception):
    """Không đọc được tài liệu."""


def clean_text(text: str) -> str:
    """Chuẩn hoá ký tự ẩn do PDF sinh ra (non-breaking space, soft hyphen)."""
    return (
        text.replace("\xa0", " ")
        .replace("\xad", "-")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def extract_pdf_pages(path: Path) -> list[PageText]:
    """Trích xuất text từng trang của PDF.

    Raises:
        IngestError: File hỏng, bị mã hoá hoặc không mở được.
    """
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24 khuyến nghị import tên mới
    except ImportError:  # pragma: no cover - tương thích bản cũ
        try:
            import fitz  # type: ignore[no-redef]
        except ImportError as exc:
            raise IngestError(
                "Chưa cài PyMuPDF. Chạy: pip install pymupdf"
            ) from exc

    try:
        document = fitz.open(path)
    except Exception as exc:  # noqa: BLE001 - PyMuPDF ném nhiều loại lỗi
        raise IngestError(f"Không mở được PDF '{path.name}': {exc}") from exc

    if document.needs_pass:
        document.close()
        raise IngestError(f"PDF '{path.name}' có mật khẩu, chưa hỗ trợ giải mã.")

    pages: list[PageText] = []
    try:
        for page_number, page in enumerate(document, start=1):
            text = clean_text(page.get_text("text") or "")
            pages.append(
                PageText(
                    source=path.name,
                    page=page_number,
                    text=text,
                    has_text=bool(text.strip()),
                )
            )
    finally:
        document.close()

    return pages


def _extract_text_file(path: Path) -> list[PageText]:
    """Đọc file text/markdown, thử UTF-8 rồi fallback cp1258 (tiếng Việt Windows)."""
    for encoding in ("utf-8", "utf-8-sig", "cp1258"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - hiếm gặp
        raise IngestError(f"Không giải mã được file text '{path.name}'.")

    return [PageText(source=path.name, page=1, text=text, has_text=bool(text.strip()))]


def load_pages(path: Path) -> list[PageText]:
    """Đọc một tài liệu và trả về danh sách trang.

    Raises:
        IngestError: Định dạng không hỗ trợ hoặc file không đọc được.
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise IngestError(
            f"Định dạng '{suffix}' chưa hỗ trợ. "
            f"Hiện hỗ trợ: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
    if not path.exists():
        raise IngestError(f"Không tìm thấy file: {path}")

    if suffix == ".pdf":
        return extract_pdf_pages(path)
    return _extract_text_file(path)


def discover_documents(directory: Path) -> list[Path]:
    """Liệt kê tài liệu hỗ trợ trong thư mục, sắp xếp ổn định theo tên."""
    if not directory.exists():
        return []
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def ingest_documents(
    directory: Path,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> tuple[list[Chunk], list[str]]:
    """Đọc toàn bộ tài liệu trong thư mục và tạo chunk.

    Returns:
        (danh sách chunk, danh sách cảnh báo). Cảnh báo ví dụ: PDF scan không text.

    Ghi chú: lỗi đọc một file không làm hỏng cả lô; file đó được ghi cảnh báo
    và bỏ qua để người dùng vẫn dùng được các tài liệu còn lại.
    """
    chunks: list[Chunk] = []
    warnings: list[str] = []

    for path in discover_documents(directory):
        try:
            pages = load_pages(path)
        except IngestError as exc:
            warnings.append(str(exc))
            logger.warning("Bỏ qua %s: %s", path.name, exc)
            continue

        text_pages = [page for page in pages if page.has_text]
        empty_pages = len(pages) - len(text_pages)
        if not text_pages:
            warnings.append(
                f"'{path.name}' không có lớp text (có thể là PDF scan). "
                "Cần OCR trước khi đưa vào RAG."
            )
            continue
        if empty_pages:
            warnings.append(
                f"'{path.name}' có {empty_pages} trang không có text; các trang đó bị bỏ qua."
            )

        options: dict[str, int] = {}
        if chunk_size is not None:
            options["chunk_size"] = chunk_size
        if overlap is not None:
            options["overlap"] = overlap
        chunks.extend(chunk_pages(text_pages, **options))

    return chunks, warnings
