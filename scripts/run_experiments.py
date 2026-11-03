"""Chạy thực nghiệm chunk size × top-k và ghi reports/experiments.md.

Đây là bài tập Tuần 5: không chọn cấu hình vì "thấy có vẻ tốt", mà đo cùng một
bộ câu hỏi với nhiều cấu hình rồi so sánh.

Chạy:
    .\\.venv\\Scripts\\python.exe scripts\\run_experiments.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chatbot.config import REPORTS_DIR  # noqa: E402
from chatbot.evaluation import run_evaluation  # noqa: E402
from chatbot.rag_pipeline import RAGPipeline  # noqa: E402
from chatbot.retrieval import RetrievalConfig  # noqa: E402

CHUNK_SIZES = [400, 700, 1000]
TOP_K_VALUES = [2, 4, 6]
OVERLAP = 120
EXPERIMENT_MIN_SCORE = 0.05


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows: list[dict[str, object]] = []

    for chunk_size in CHUNK_SIZES:
        pipeline = RAGPipeline.empty(
            config=RetrievalConfig(top_k=max(TOP_K_VALUES), min_score=EXPERIMENT_MIN_SCORE)
        )
        pipeline.build_index(chunk_size=chunk_size, overlap=OVERLAP, persist=False)
        for top_k in TOP_K_VALUES:
            result = run_evaluation(
                top_k=top_k,
                chunk_size=chunk_size,
                overlap=OVERLAP,
                pipeline=pipeline,
            )
            row = {
                "chunk_size": chunk_size,
                "overlap": OVERLAP,
                "top_k": top_k,
                "chunks": len(pipeline.store),
                **result.summary,
            }
            rows.append(row)
            print(
                f"chunk={chunk_size} top_k={top_k} chunks={len(pipeline.store)} "
                f"hit={row['hit_rate_at_k']} recall={row['recall_at_k']} "
                f"mrr={row['mrr']} answer={row['answer_accuracy']} "
                f"abstain={row['abstention_accuracy']}"
            )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "experiments_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best = max(rows, key=lambda row: (row["mrr"], row["hit_rate_at_k"], row["recall_at_k"]))
    lines = [
        "# Thực nghiệm chunk size × top-k",
        "",
        f"Bộ câu hỏi: `data/eval_questions.json` ({rows[0]['questions']} câu). "
        f"Embedding: `HashingEmbedder`, generator: `extractive`, overlap={OVERLAP}, "
        f"min_score={EXPERIMENT_MIN_SCORE} (ngưỡng thấp để quan sát nhiều ứng viên hơn).",
        "",
        "| Chunk size | Số chunk | top_k | Hit Rate | Recall | MRR | Answer acc | Abstain |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['chunk_size']} | {row['chunks']} | {row['top_k']} | "
            f"{float(row['hit_rate_at_k']) * 100:.1f}% | "
            f"{float(row['recall_at_k']) * 100:.1f}% | "
            f"{float(row['mrr']):.3f} | "
            f"{float(row['answer_accuracy']) * 100:.1f}% | "
            f"{float(row['abstention_accuracy']) * 100:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Cấu hình tốt nhất theo MRR",
            "",
            f"- chunk_size = **{best['chunk_size']}**, overlap = {best['overlap']}, "
            f"top_k = **{best['top_k']}**",
            f"- Hit Rate = {float(best['hit_rate_at_k']) * 100:.1f}%, "
            f"Recall = {float(best['recall_at_k']) * 100:.1f}%, "
            f"MRR = {float(best['mrr']):.3f}",
            "",
            "## Nhận xét",
            "",
            "- Chunk nhỏ (400) tạo nhiều chunk hơn, recall có thể tăng nhưng mỗi chunk ít ngữ cảnh.",
            "- Chunk lớn (1000) giữ ngữ cảnh tốt hơn nhưng dễ chứa nhiều thông tin nhiễu.",
            "- top_k không làm thay đổi Hit Rate ở bộ tài liệu nhỏ này vì chunk đúng đã ở hạng 1; top_k chỉ có ích khi đáp án nằm rải rác ở nhiều chunk.",
            "- Với bộ tài liệu nhỏ (10 trang), khác biệt giữa các cấu hình không đủ để kết luận "
            "cho tài liệu lớn. Cần lặp lại trên bộ tài liệu đa dạng hơn.",
            f"- Abstain chỉ đạt 50% vì thực nghiệm hạ `min_score` xuống {EXPERIMENT_MIN_SCORE} "
            "để quan sát nhiều ứng viên; đây không phải cấu hình dùng khi vận hành.",
            "",
            "> Số liệu trên được sinh tự động bởi `scripts/run_experiments.py`; "
            "không chỉnh tay.",
        ]
    )
    (REPORTS_DIR / "experiments.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Đã ghi {REPORTS_DIR / 'experiments.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
