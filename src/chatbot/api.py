"""HTTP API cho pipeline RAG, dùng FastAPI.

Chạy:
    uvicorn chatbot.api:app --reload --port 8000

Tài liệu tự sinh: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from chatbot.config import MAX_UPLOAD_BYTES, VERSION
from chatbot.ingest import SUPPORTED_EXTENSIONS, IngestError
from chatbot.rag_pipeline import RAGPipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    evidence_found: bool
    provider: str
    sources: list[dict[str, object]]
    retrieved: list[dict[str, object]]


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    app = FastAPI(
        title="AI Knowledge Base Chatbot",
        version=VERSION,
        description="Hỏi đáp tài liệu với RAG, có trích dẫn nguồn.",
    )
    app.state.pipeline = pipeline or RAGPipeline.from_index()

    @app.get("/")
    def root() -> dict[str, object]:
        return {"name": "AI Knowledge Base Chatbot", "version": VERSION, "docs": "/docs"}

    @app.get("/health")
    def health() -> dict[str, object]:
        current: RAGPipeline = app.state.pipeline
        return {
            "status": "ok",
            "chunks": len(current.store),
            "sources": current.stats()["source_count"],
        }

    @app.get("/stats")
    def stats() -> dict[str, object]:
        current: RAGPipeline = app.state.pipeline
        return current.stats()

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        current: RAGPipeline = app.state.pipeline
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="Câu hỏi không được rỗng.")
        answer = current.ask(question, top_k=request.top_k)
        return AskResponse(**answer.to_dict())  # type: ignore[arg-type]

    @app.post("/documents")
    async def upload_document(file: UploadFile = File(...)) -> dict[str, object]:
        filename = Path(file.filename or "upload").name
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Định dạng '{suffix}' chưa hỗ trợ. Chọn: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
            )

        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File vượt quá {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )
        if not contents:
            raise HTTPException(status_code=422, detail="File rỗng.")

        try:
            with TemporaryDirectory(prefix="rag_upload_") as directory:
                temporary_path = Path(directory) / filename
                temporary_path.write_bytes(contents)
                report = app.state.pipeline.add_document(temporary_path)
        except IngestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return report.to_dict()

    return app


app = create_app()
