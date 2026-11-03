"""Giao diện dòng lệnh cho toàn bộ project.

Cách dùng::

    python src/main.py                      # chat rule-based / LLM như Tuần 2
    python src/main.py chat --rag           # chat dựa trên tài liệu đã index
    python src/main.py ingest               # đọc data/sample_docs và tạo index
    python src/main.py ask "RAG là gì?"     # hỏi một câu, in nguồn
    python src/main.py eval                 # chạy bộ đánh giá Tuần 5
    python src/main.py serve                # chạy FastAPI tại /docs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chatbot.config import API_HOST, API_PORT, SAMPLE_DOCS_DIR
from chatbot.main import configure_console_encoding, run_chatbot
from chatbot.rag_pipeline import RAGPipeline


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_chat(args: argparse.Namespace) -> int:
    if not args.rag:
        run_chatbot()
        return 0

    pipeline = RAGPipeline.from_index()
    if len(pipeline.store) == 0:
        print(
            "Chưa có index. Chạy: python src/main.py ingest",
            file=sys.stderr,
        )
        return 1

    def rag_responder(question: str) -> str:
        answer = pipeline.ask(question, top_k=args.top_k)
        if not answer.sources:
            return answer.text
        lines = [answer.text, "", "Nguồn:"]
        lines.extend(
            f"- {source.source}, trang {source.page} (score {source.score:.3f})"
            for source in answer.sources
        )
        return "\n".join(lines)

    run_chatbot(responder=rag_responder)
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    pipeline = RAGPipeline.empty()
    report = pipeline.build_index(
        docs_dir=Path(args.docs),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    _print_json(report.to_dict())
    if report.warnings:
        print("\nCảnh báo:", file=sys.stderr)
        for warning in report.warnings:
            print(f"- {warning}", file=sys.stderr)
    if report.chunks == 0:
        print("\nKhông tạo được chunk nào. Kiểm tra tài liệu đầu vào.", file=sys.stderr)
        return 1
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    pipeline = RAGPipeline.from_index()
    if len(pipeline.store) == 0:
        print("Chưa có index. Chạy: python src/main.py ingest", file=sys.stderr)
        return 1

    if args.retrieve_only:
        results = pipeline.retrieve_only(args.question, top_k=args.top_k)
        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            print(
                f"[{index}] {chunk.source} trang {chunk.page} "
                f"score={result.score:.3f} chunk#{chunk.chunk_index}"
            )
            print(chunk.text)
            print()
        return 0

    answer = pipeline.ask(args.question, top_k=args.top_k, min_score=args.min_score)
    _print_json(answer.to_dict())
    return 0 if answer.evidence_found else 2


def cmd_eval(args: argparse.Namespace) -> int:
    from chatbot.evaluation import run_evaluation, write_report

    result = run_evaluation(
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    report_path = write_report(result)
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    print(f"\nĐã ghi báo cáo: {report_path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("chatbot.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-knowledge-chatbot",
        description="Chatbot hỏi đáp tài liệu với RAG.",
    )
    subparsers = parser.add_subparsers(dest="command")

    chat = subparsers.add_parser("chat", help="Chat trong terminal (mặc định)")
    chat.add_argument("--rag", action="store_true", help="Trả lời dựa trên index tài liệu")
    chat.add_argument("--top-k", type=int, default=None, help="Số chunk truy xuất")
    chat.set_defaults(func=cmd_chat)

    ingest = subparsers.add_parser("ingest", help="Đọc tài liệu và tạo index")
    ingest.add_argument("--docs", default=str(SAMPLE_DOCS_DIR), help="Thư mục tài liệu")
    ingest.add_argument("--chunk-size", type=int, default=700, help="Kích thước chunk (ký tự)")
    ingest.add_argument("--overlap", type=int, default=120, help="Overlap giữa hai chunk")
    ingest.set_defaults(func=cmd_ingest)

    ask = subparsers.add_parser("ask", help="Hỏi một câu và in câu trả lời + nguồn")
    ask.add_argument("question", help="Câu hỏi")
    ask.add_argument("--top-k", type=int, default=None)
    ask.add_argument("--min-score", type=float, default=None)
    ask.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Chỉ in chunk truy xuất, không gọi generator (debug retrieval)",
    )
    ask.set_defaults(func=cmd_ask)

    evaluation = subparsers.add_parser("eval", help="Chạy bộ đánh giá và ghi báo cáo")
    evaluation.add_argument("--top-k", type=int, default=4)
    evaluation.add_argument("--chunk-size", type=int, default=700)
    evaluation.add_argument("--overlap", type=int, default=120)
    evaluation.set_defaults(func=cmd_eval)

    serve = subparsers.add_parser("serve", help="Chạy FastAPI (uvicorn)")
    serve.add_argument("--host", default=API_HOST)
    serve.add_argument("--port", type=int, default=API_PORT)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_console_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        args = parser.parse_args(["chat"])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
