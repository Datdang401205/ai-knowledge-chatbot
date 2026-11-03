"""Ghép toàn bộ luồng RAG: PDF → chunk → embedding → store → retrieval → answer.

Một lớp duy nhất để CLI, FastAPI và Streamlit dùng chung. Nhờ vậy ba giao diện
không thể "lệch logic" nhau, và test chỉ cần test lớp này.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from chatbot.chunking import chunk_pages
from chatbot.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    INDEX_FILE,
    MANIFEST_FILE,
    NO_EVIDENCE_MESSAGE,
    SAMPLE_DOCS_DIR,
)
from chatbot.embeddings import Embedder, get_embedder
from chatbot.generation import Answer, Generator, get_generator
from chatbot.ingest import IngestError, discover_documents, load_pages
from chatbot.retrieval import RetrievalConfig, Retriever
from chatbot.vector_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexReport:
    """Kết quả một lần build index."""

    documents: int
    chunks: int
    warnings: tuple[str, ...]
    index_path: str
    chunk_size: int
    overlap: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RAGPipeline:
    """Điểm vào duy nhất cho ingest, retrieval và hỏi đáp."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder | None = None,
        generator: Generator | None = None,
        config: RetrievalConfig | None = None,
        index_path: Path = INDEX_FILE,
        manifest_path: Path = MANIFEST_FILE,
    ) -> None:
        self.store = store
        self.embedder = embedder or get_embedder()
        self.generator = generator or get_generator()
        self.config = config or RetrievalConfig()
        self.index_path = Path(index_path)
        self.manifest_path = Path(manifest_path)
        self.chunk_size = CHUNK_SIZE
        self.overlap = CHUNK_OVERLAP
        self._retriever = Retriever(self.store, self.embedder, self.config)

    # ------------------------------------------------------------ factories
    @classmethod
    def empty(cls, **kwargs: object) -> "RAGPipeline":
        """Pipeline chưa có index (dùng cho test hoặc lần chạy đầu)."""
        embedder = kwargs.pop("embedder", None)  # type: ignore[arg-type]
        return cls(VectorStore(), embedder=embedder, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_index(cls, **kwargs: object) -> "RAGPipeline":
        """Load index có sẵn; nếu chưa có thì trả pipeline rỗng."""
        index_path = Path(kwargs.get("index_path", INDEX_FILE))  # type: ignore[arg-type]
        embedder = kwargs.pop("embedder", None)  # type: ignore[arg-type]
        if index_path.exists():
            try:
                store = VectorStore.load(index_path)
                return cls(store, embedder=embedder, **kwargs)  # type: ignore[arg-type]
            except VectorStoreError:
                logger.warning("Index hỏng, bắt đầu với store rỗng", exc_info=True)
        return cls(VectorStore(), embedder=embedder, **kwargs)  # type: ignore[arg-type]

    # --------------------------------------------------------------- ingest
    def build_index(
        self,
        docs_dir: Path = SAMPLE_DOCS_DIR,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
        persist: bool = True,
    ) -> IndexReport:
        """Đọc toàn bộ tài liệu trong thư mục và ghi index mới.

        Luôn tạo store mới để tránh trộn chunk cũ với tài liệu đã bị xoá.
        """
        docs_dir = Path(docs_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        documents = discover_documents(docs_dir)
        pages = []
        warnings: list[str] = []

        for path in documents:
            try:
                loaded = load_pages(path)
            except IngestError as exc:
                warnings.append(str(exc))
                logger.warning("Bỏ qua %s: %s", path.name, exc)
                continue

            usable = [page for page in loaded if page.has_text]
            empty = len(loaded) - len(usable)
            if not usable:
                warnings.append(
                    f"'{path.name}' không có text (có thể là PDF scan); cần OCR trước."
                )
                continue
            if empty:
                warnings.append(
                    f"'{path.name}' có {empty} trang không có text; đã bỏ qua các trang đó."
                )
            pages.extend(usable)

        chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        store = VectorStore(dimension=None)
        if chunks:
            vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
            store.add(chunks, vectors, embedding_provider=type(self.embedder).__name__)

        self.store = store
        self._retriever = Retriever(store, self.embedder, self.config)

        report = IndexReport(
            documents=len({page.source for page in pages}),
            chunks=len(store),
            warnings=tuple(warnings),
            index_path=str(self.index_path),
            chunk_size=chunk_size,
            overlap=overlap,
        )
        if persist:
            self.persist(report)
        return report

    def add_document(self, path: Path) -> IndexReport:
        """Thêm một tài liệu vào index hiện có (dùng cho upload qua API/UI)."""
        path = Path(path)
        pages = [page for page in load_pages(path) if page.has_text]
        warnings: list[str] = []
        if not pages:
            warnings.append(f"'{path.name}' không có text; cần OCR trước khi ingest.")

        chunks = chunk_pages(pages, chunk_size=self.chunk_size, overlap=self.overlap)
        added = 0
        if chunks:
            vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
            added = self.store.add(
                chunks, vectors, embedding_provider=type(self.embedder).__name__
            )

        report = IndexReport(
            documents=len({page.source for page in pages}),
            chunks=added,
            warnings=tuple(warnings),
            index_path=str(self.index_path),
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )
        self.persist(report)
        return report

    def persist(self, report: IndexReport | None = None) -> None:
        """Ghi store + manifest. Manifest giúp biết index được tạo bằng cấu hình nào."""
        self.store.save(self.index_path)
        manifest = {
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "embedding": {
                "provider": type(self.embedder).__name__,
                "dimension": self.store.dimension,
            },
            "generator": getattr(self.generator, "name", type(self.generator).__name__),
            "retrieval": asdict(self.config),
            "stats": self.store.stats(),
        }
        if report is not None:
            manifest["last_build"] = report.to_dict()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------ ask
    def ask(
        self,
        question: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> Answer:
        """Trả lời câu hỏi kèm nguồn. Không có bằng chứng → nói rõ không tìm thấy."""
        question = (question or "").strip()
        if not question:
            raise ValueError("Câu hỏi không được rỗng.")
        if len(self.store) == 0:
            return Answer(text=NO_EVIDENCE_MESSAGE, evidence_found=False, provider="none")

        results = self._retriever.retrieve(
            question,
            top_k=top_k if top_k is not None else self.config.top_k,
            min_score=min_score if min_score is not None else self.config.min_score,
        )
        if not results:
            return Answer(text=NO_EVIDENCE_MESSAGE, evidence_found=False, provider=self.generator.name)
        return self.generator.generate(question, results)

    def retrieve_only(self, question: str, top_k: int | None = None):
        """Chỉ trả chunk — dùng để debug retrieval trước khi gọi LLM."""
        return self._retriever.retrieve(question, top_k=top_k)

    @property
    def top_k(self) -> int:
        return self.config.top_k

    def stats(self) -> dict[str, object]:
        return self.store.stats()


def default_pipeline() -> RAGPipeline:
    """Pipeline dùng đường dẫn mặc định trong project."""
    return RAGPipeline.from_index()
