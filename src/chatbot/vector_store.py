"""Vector store tối giản: lưu chunk + vector, tìm bằng cosine similarity.

Vì sao tự viết thay vì dùng Chroma/FAISS ngay?
---------------------------------------------
Mục tiêu Tuần 4 là *hiểu* retrieval hoạt động thế nào. Một file JSON + numpy
đủ cho vài nghìn chunk, dễ đọc, dễ debug, không cần service ngoài. Khi dữ liệu
lớn hơn, thay lớp này bằng FAISS/Chroma mà không đổi phần retrieval phía trên
vì interface chỉ gồm ``add`` / ``search`` / ``save`` / ``load``.

Lưu ý: cosine similarity chỉ có nghĩa khi vector đã chuẩn hoá L2. Store chuẩn
hoá lại lần nữa để không phụ thuộc vào việc embedder có làm đúng hay không.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from chatbot.chunking import Chunk


@dataclass(frozen=True)
class SearchResult:
    """Một chunk tìm được kèm điểm tương đồng cosine trong khoảng [-1, 1]."""

    chunk: Chunk
    score: float


class VectorStoreError(Exception):
    """Lỗi khi đọc/ghi vector store."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class VectorStore:
    """Kho vector trong bộ nhớ, lưu xuống một file JSON."""

    def __init__(self, path: Path | None = None, dimension: int | None = None) -> None:
        self.path = Path(path) if path else None
        self.dimension = dimension
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._lock = threading.Lock()
        self.created_at = _utc_now()
        self.updated_at = self.created_at
        self.embedding_provider = ""

    # ------------------------------------------------------------------ add
    def add(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        embedding_provider: str = "",
    ) -> int:
        """Thêm chunk + vector. Trả về số chunk thực sự được thêm.

        Raises:
            VectorStoreError: Số dòng không khớp hoặc sai chiều vector.
        """
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f"Số chunk ({len(chunks)}) khác số vector ({len(vectors)})."
            )
        if not chunks:
            return 0

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise VectorStoreError("vectors phải là ma trận 2 chiều.")
        if self.dimension is None:
            self.dimension = int(matrix.shape[1])
        if matrix.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Vector có {matrix.shape[1]} chiều, store cần {self.dimension}."
            )

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms

        with self._lock:
            existing_ids = {chunk.id for chunk in self._chunks}
            keep = [index for index, chunk in enumerate(chunks) if chunk.id not in existing_ids]
            if not keep:
                return 0
            new_chunks = [chunks[index] for index in keep]
            new_matrix = matrix[keep]

            self._chunks.extend(new_chunks)
            if self._vectors is None:
                self._vectors = new_matrix
            else:
                self._vectors = np.vstack([self._vectors, new_matrix])
            self.embedding_provider = embedding_provider or self.embedding_provider
            self.updated_at = _utc_now()
        return len(new_chunks)

    # --------------------------------------------------------------- search
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 4,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Tìm top-k chunk gần câu hỏi nhất.

        Args:
            query_vector: Vector câu hỏi (cùng chiều với store).
            top_k: Số kết quả tối đa.
            min_score: Nếu đặt, loại kết quả có cosine < ngưỡng.
        """
        with self._lock:
            if not self._chunks or self._vectors is None:
                return []
            if top_k <= 0:
                return []

            query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
            if self.dimension is not None and query.shape[0] != self.dimension:
                raise VectorStoreError(
                    f"Câu hỏi có {query.shape[0]} chiều, store cần {self.dimension}."
                )
            norm = float(np.linalg.norm(query))
            if norm == 0:
                return []
            query = query / norm

            scores = self._vectors @ query
            order = np.argsort(-scores)[:top_k]
            results = [
                SearchResult(chunk=self._chunks[int(index)], score=float(scores[index]))
                for index in order
            ]

        if min_score is not None:
            results = [result for result in results if result.score >= min_score]
        return results

    # ------------------------------------------------------------ lifecycle
    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def stats(self) -> dict[str, object]:
        sources = sorted({chunk.source for chunk in self._chunks})
        return {
            "chunks": len(self._chunks),
            "sources": sources,
            "source_count": len(sources),
            "dimension": self.dimension,
            "embedding_provider": self.embedding_provider,
            "updated_at": self.updated_at,
        }

    def save(self, path: Path | None = None) -> Path:
        """Ghi atomically: ghi file tạm rồi thay thế, tránh file hỏng giữa chừng."""
        target = Path(path) if path else self.path
        if target is None:
            raise VectorStoreError("Chưa cấu hình đường dẫn lưu vector store.")
        target.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            payload = {
                "version": 1,
                "dimension": self.dimension,
                "embedding_provider": self.embedding_provider,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "chunks": [
                    {
                        **chunk.to_dict(),
                        "vector": (
                            self._vectors[index].round(6).tolist()
                            if self._vectors is not None
                            else []
                        ),
                    }
                    for index, chunk in enumerate(self._chunks)
                ],
            }

        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)
        self.path = target
        return target

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        """Đọc store từ JSON.

        Raises:
            VectorStoreError: File không tồn tại hoặc hỏng.
        """
        path = Path(path)
        if not path.exists():
            raise VectorStoreError(f"Chưa có vector store tại {path}.")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VectorStoreError(f"Không đọc được vector store: {exc}") from exc

        store = cls(path=path, dimension=payload.get("dimension"))
        store.embedding_provider = payload.get("embedding_provider", "")
        store.created_at = payload.get("created_at", store.created_at)
        store.updated_at = payload.get("updated_at", store.updated_at)

        chunks: list[Chunk] = []
        vectors: list[list[float]] = []
        for item in payload.get("chunks", []):
            try:
                chunks.append(Chunk.from_dict(item))
                vectors.append([float(value) for value in item["vector"]])
            except (KeyError, TypeError, ValueError) as exc:
                raise VectorStoreError(f"Bản ghi chunk hỏng trong store: {exc}") from exc

        if chunks:
            store._chunks = chunks
            store._vectors = np.asarray(vectors, dtype=np.float32)
        return store
