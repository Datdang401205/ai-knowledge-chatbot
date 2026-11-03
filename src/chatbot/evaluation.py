"""Đánh giá retrieval và câu trả lời trên bộ câu hỏi cố định.

Nguyên tắc: **không đặt số liệu trước khi chạy**. Mọi con số trong báo cáo
được tính từ lần chạy thật và lưu kèm cấu hình (chunk size, top-k, provider).

Các thước đo:
- Hit Rate@k: câu hỏi có ít nhất một nguồn đúng trong top-k không?
- Recall@k: bao nhiêu phần trăm nguồn đúng được tìm thấy?
- MRR: nguồn đúng đầu tiên xuất hiện ở hạng nào?
- Answer correctness: tỷ lệ từ khóa của đáp án tham chiếu xuất hiện trong câu trả lời
  (>= 50%). Đây là kiểm tra tự động thô, **không thay thế chấm thủ công**.
- Abstention accuracy: với câu không có trong tài liệu, hệ thống có từ chối không?
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from chatbot.config import EVAL_QUESTIONS_PATH, REPORTS_DIR
from chatbot.rag_pipeline import RAGPipeline

_WORD_RE = re.compile(r"[0-9a-zA-ZÀ-ỹà-ỹđĐ]+", re.UNICODE)
_STOPWORDS = {
    "la", "gi", "cua", "va", "co", "the", "nao", "tai", "lieu", "cho", "toi",
    "trong", "duoc", "khi", "nhung", "mot", "cac", "is", "what", "how", "why",
    "the", "a", "an", "of", "to", "in", "and", "or", "for", "which", "does",
    "with", "that", "this", "it", "are", "was", "be", "by", "on", "as", "at",
}


def _normalize(text: str) -> str:
    import unicodedata

    lowered = text.lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", lowered)
    return " ".join(
        "".join(char for char in decomposed if not unicodedata.combining(char)).split()
    )


def _terms(text: str) -> set[str]:
    return {
        word
        for word in _WORD_RE.findall(_normalize(text))
        if len(word) > 1 and word not in _STOPWORDS
    }


def _keyword_coverage(reference: str, answer: str) -> float:
    """Tỷ lệ từ khóa của đáp án tham chiếu có mặt trong câu trả lời."""
    reference_terms = _terms(reference)
    if not reference_terms:
        return 0.0
    answer_terms = _terms(answer)
    return len(reference_terms & answer_terms) / len(reference_terms)


@dataclass
class QuestionResult:
    id: str
    category: str
    question: str
    expected_sources: list[dict[str, object]]
    retrieved_sources: list[dict[str, object]] = field(default_factory=list)
    first_relevant_rank: int | None = None
    hit: bool = False
    recall: float = 0.0
    evidence_found: bool = False
    answer: str = ""
    has_citation: bool = False
    keyword_coverage: float = 0.0
    correct: bool = False
    latency_ms: float = 0.0
    error: str = ""


@dataclass
class EvaluationResult:
    summary: dict[str, object]
    questions: list[QuestionResult]
    config: dict[str, object]
    generated_at: str


def load_questions(path: Path = EVAL_QUESTIONS_PATH) -> list[dict[str, object]]:
    """Đọc bộ câu hỏi; báo lỗi rõ nếu JSON hỏng hoặc thiếu trường."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy bộ câu hỏi: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"eval_questions.json không hợp lệ: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise ValueError("Bộ câu hỏi phải là một danh sách không rỗng.")

    required = {"id", "question", "expected_sources"}
    for index, item in enumerate(data):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"Câu hỏi thứ {index} thiếu trường bắt buộc {required}.")
    return data


def _source_key(source: dict[str, object]) -> tuple[str, int]:
    return str(source.get("file", "")), int(source.get("page", 0))


def evaluate_question(
    question: dict[str, object],
    pipeline: RAGPipeline,
    top_k: int,
) -> QuestionResult:
    expected = [
        {"file": source["file"], "page": source["page"]}
        for source in question.get("expected_sources", [])  # type: ignore[union-attr]
    ]
    expected_keys = {_source_key(source) for source in expected}
    result = QuestionResult(
        id=str(question["id"]),
        category=str(question.get("category", "factual")),
        question=str(question["question"]),
        expected_sources=expected,
        evidence_found=False,
    )

    started = time.perf_counter()
    try:
        answer = pipeline.ask(str(question["question"]), top_k=top_k)
    except Exception as exc:  # noqa: BLE001 - ghi lỗi vào báo cáo thay vì dừng cả bộ
        result.error = f"{type(exc).__name__}: {exc}"
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result
    result.latency_ms = (time.perf_counter() - started) * 1000

    retrieved_keys: list[tuple[str, int]] = []
    for item in answer.retrieved:
        key = (item.chunk.source, item.chunk.page)
        if key not in retrieved_keys:
            retrieved_keys.append(key)
    result.retrieved_sources = [
        {"file": file, "page": page} for file, page in retrieved_keys
    ]
    result.answer = answer.text
    result.evidence_found = answer.evidence_found
    result.has_citation = "[" in answer.text and "]" in answer.text

    if expected_keys:
        for rank, key in enumerate(retrieved_keys, start=1):
            if key in expected_keys:
                result.first_relevant_rank = rank
                break
        result.hit = result.first_relevant_rank is not None
        found = len(expected_keys & set(retrieved_keys))
        result.recall = found / len(expected_keys)
        result.keyword_coverage = _keyword_coverage(
            str(question.get("reference_answer", "")), answer.text
        )
        result.correct = result.keyword_coverage >= 0.5
    else:
        # Câu không có trong tài liệu: đúng khi hệ thống không tuyên bố có bằng chứng.
        result.correct = not answer.evidence_found
        result.keyword_coverage = 0.0

    return result


def run_evaluation(
    eval_path: Path = EVAL_QUESTIONS_PATH,
    top_k: int = 4,
    chunk_size: int = 700,
    overlap: int = 120,
    pipeline: RAGPipeline | None = None,
) -> EvaluationResult:
    """Chạy toàn bộ bộ câu hỏi và tổng hợp thước đo."""
    questions = load_questions(eval_path)

    if pipeline is None:
        pipeline = RAGPipeline.empty()
        pipeline.build_index(chunk_size=chunk_size, overlap=overlap, persist=False)

    results = [
        evaluate_question(question, pipeline, top_k=top_k) for question in questions
    ]

    answerable = [item for item in results if item.expected_sources]
    unanswerable = [item for item in results if not item.expected_sources]
    errors = [item for item in results if item.error]

    hits = [item.hit for item in answerable]
    recalls = [item.recall for item in answerable]
    reciprocal_ranks = [
        1.0 / item.first_relevant_rank if item.first_relevant_rank else 0.0
        for item in answerable
    ]
    answer_checks = [item.correct for item in results if not item.error]
    citation_rate = [item.has_citation for item in answerable]

    summary: dict[str, object] = {
        "questions": len(results),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "errors": len(errors),
        "hit_rate_at_k": round(sum(hits) / len(hits), 4) if hits else 0.0,
        "recall_at_k": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)
        if reciprocal_ranks
        else 0.0,
        "answer_accuracy": round(sum(answer_checks) / len(answer_checks), 4)
        if answer_checks
        else 0.0,
        "citation_rate": round(sum(citation_rate) / len(citation_rate), 4)
        if citation_rate
        else 0.0,
        "abstention_accuracy": round(
            sum(1 for item in unanswerable if item.correct) / len(unanswerable), 4
        )
        if unanswerable
        else 0.0,
        "avg_latency_ms": round(
            sum(item.latency_ms for item in results) / len(results), 2
        )
        if results
        else 0.0,
    }

    config = {
        "top_k": top_k,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding": type(pipeline.embedder).__name__,
        "generator": getattr(pipeline.generator, "name", type(pipeline.generator).__name__),
        "chunks": len(pipeline.store),
        "sources": pipeline.store.stats()["source_count"],
    }
    return EvaluationResult(
        summary=summary,
        questions=results,
        config=config,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _pct(value: object) -> str:
    return f"{float(value) * 100:.1f}%"


def render_markdown(result: EvaluationResult) -> str:
    """Sinh báo cáo Markdown từ kết quả thật (không hardcode số)."""
    summary = result.summary
    config = result.config
    lines = [
        "# Báo cáo đánh giá",
        "",
        f"- Thời điểm chạy: `{result.generated_at}`",
        f"- Cấu hình: top_k={config['top_k']}, chunk_size={config['chunk_size']}, "
        f"overlap={config['overlap']}",
        f"- Embedding: `{config['embedding']}` — Generator: `{config['generator']}`",
        f"- Index: {config['chunks']} chunk từ {config['sources']} tài liệu",
        "",
        "## 1. Kết quả tổng hợp",
        "",
        "| Thước đo | Kết quả |",
        "|---|---|",
        f"| Số câu hỏi | {summary['questions']} "
        f"({summary['answerable']} có đáp án, {summary['unanswerable']} không có) |",
        f"| Hit Rate@k | {_pct(summary['hit_rate_at_k'])} |",
        f"| Recall@k | {_pct(summary['recall_at_k'])} |",
        f"| MRR | {float(summary['mrr']):.3f} |",
        f"| Answer accuracy (tự động, theo từ khóa) | {_pct(summary['answer_accuracy'])} |",
        f"| Tỷ lệ câu trả lời có trích dẫn | {_pct(summary['citation_rate'])} |",
        f"| Abstention accuracy (câu ngoài tài liệu) | {_pct(summary['abstention_accuracy'])} |",
        f"| Độ trễ trung bình mỗi câu | {summary['avg_latency_ms']} ms |",
        f"| Số câu lỗi | {summary['errors']} |",
        "",
        "## 2. Kết quả theo loại câu hỏi",
        "",
        "| Loại | Số câu | Hit | Recall TB | Đúng (auto) |",
        "|---|---|---|---|---|",
    ]

    categories: dict[str, list[QuestionResult]] = {}
    for item in result.questions:
        categories.setdefault(item.category, []).append(item)
    for category, items in sorted(categories.items()):
        with_expected = [item for item in items if item.expected_sources]
        hit_count = sum(1 for item in with_expected if item.hit)
        avg_recall = (
            sum(item.recall for item in with_expected) / len(with_expected)
            if with_expected
            else 0.0
        )
        correct = sum(1 for item in items if item.correct)
        lines.append(
            f"| {category} | {len(items)} | {hit_count}/{len(with_expected)} | "
            f"{_pct(avg_recall)} | {correct}/{len(items)} |"
        )

    lines.extend(
        [
            "",
            "## 3. Các trường hợp thất bại nổi bật",
            "",
            "| ID | Câu hỏi | Nguồn mong đợi | Nguồn truy xuất | Vấn đề |",
            "|---|---|---|---|---|",
        ]
    )
    failures = [item for item in result.questions if not item.correct or item.error]
    if not failures:
        lines.append("| — | Không có | — | — | — |")
    for item in failures[:20]:
        expected = ", ".join(
            f"{source['file']} p{source['page']}" for source in item.expected_sources
        ) or "—"
        retrieved = ", ".join(
            f"{source['file']} p{source['page']}" for source in item.retrieved_sources[:3]
        ) or "—"
        issue = item.error or (
            "Không truy xuất đúng nguồn"
            if item.expected_sources and not item.hit
            else "Câu trả lời thiếu từ khóa / trích dẫn"
            if item.expected_sources
            else "Vẫn tuyên bố có bằng chứng cho câu ngoài tài liệu"
        )
        lines.append(
            f"| {item.id} | {item.question} | {expected} | {retrieved} | {issue} |"
        )

    lines.extend(
        [
            "",
            "## 4. Cách đọc kết quả",
            "",
            "- **Hit Rate@k** cao nhưng **answer accuracy** thấp: retrieval đúng, khâu sinh câu trả lời kém.",
            "- **Hit Rate@k** thấp: vấn đề nằm ở chunking/embedding/top-k, không phải LLM.",
            "- **Abstention accuracy** thấp: ngưỡng liên quan quá thấp hoặc embedding quá nhạy.",
            "- Answer accuracy ở đây là kiểm tra từ khóa tự động, có thể đánh giá thấp các câu trả lời "
            "diễn đạt khác đáp án tham chiếu. Cần chấm thủ công một mẫu để xác nhận.",
            "",
            "## 5. Giới hạn của lần đánh giá này",
            "",
            "- Bộ câu hỏi do chính người làm project soạn, có thể thiên lệch theo cách diễn đạt của tài liệu.",
            "- Chỉ có 4 câu không có đáp án; chưa đủ để kết luận về khả năng từ chối ngoài phân phối.",
            "- Generator `extractive` chỉ trích câu, không diễn giải; điểm answer accuracy vì vậy thấp hơn "
            "so với khi dùng LLM thật.",
            "- Chưa đo chi phí token, chưa đo trên tài liệu dài hàng trăm trang.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(
    result: EvaluationResult,
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    """Ghi ``evaluation.md`` và ``evaluation_results.json``."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = reports_dir / "evaluation.md"
    markdown_path.write_text(render_markdown(result), encoding="utf-8")

    json_path = reports_dir / "evaluation_results.json"
    json_path.write_text(
        json.dumps(
            {
                "summary": result.summary,
                "config": result.config,
                "generated_at": result.generated_at,
                "questions": [asdict(item) for item in result.questions],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return markdown_path
