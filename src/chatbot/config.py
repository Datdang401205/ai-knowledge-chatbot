"""Cấu hình tập trung cho chatbot.

Mọi chuỗi có thể thay đổi (tên app, câu chào, câu thoát) nằm ở đây để
phần xử lý logic không bị rải hằng số rải rác khắp nơi.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
INDEX_DIR = DATA_DIR / "index"
EVAL_QUESTIONS_PATH = DATA_DIR / "eval_questions.json"
REPORTS_DIR = BASE_DIR / "reports"

APP_NAME = "AI Knowledge Base Chatbot"
VERSION = "0.6.0"
MODE = os.getenv("CHATBOT_MODE", "rule-based")

WELCOME_MESSAGE = (
    f"Xin chào! Tôi là {APP_NAME} (v{VERSION}, chế độ {MODE}).\n"
    "Tôi chưa đọc được tài liệu nào ở phiên bản này. "
    "Hỏi tôi về chức năng, hoặc gõ 'help' để xem các chủ đề mình hỗ trợ."
)

PROMPT = "Bạn> "
FALLBACK_REPLY = (
    "Tôi chưa hiểu câu hỏi của bạn. Chế độ chat mặc định chỉ trả lời theo luật cố định, "
    "gõ 'help' để xem các chủ đề mình hỗ trợ."
)
EMPTY_INPUT_MESSAGE = "Bạn chưa nhập nội dung. Hãy gõ câu hỏi hoặc 'exit' để thoát."
GOODBYE_MESSAGE = "Tạm biệt! Hẹn gặp lại."
ERROR_MESSAGE = "Đã xảy ra lỗi không mong muốn: {error}"

EXIT_COMMANDS = ("exit", "quit", "q", "thoat", "ket")

# Cấu hình LLM (Ollama) - dùng cho Tuần 2
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))

# System prompt yêu cầu trả lời thận trọng, không bịa
LLM_SYSTEM_PROMPT = (
    "Bạn là một trợ lý AI giúp trả lời câu hỏi một cách ngắn gọn, chính xác và trung thực.\n"
    "Quy tắc:\n"
    "- Chỉ trả lời dựa trên kiến thức chung nếu có, nhưng nếu không chắc chắn hãy thừa nhận không biết.\n"
    "- Trả lời bằng ngôn ngữ phù hợp với câu hỏi của người dùng (nếu người dùng hỏi bằng tiếng Việt, hãy trả lời bằng tiếng Việt).\n"
    "- Không bịa số liệu, không đưa thông tin không có căn cứ.\n"
    "- Nếu câu hỏi vượt quá kiến thức hoặc không có đủ thông tin để trả lời chắc chắn, hãy nói rõ: 'Tôi không chắc chắn về điều này.'\n"
)

# ---------------------------------------------------------------------------
# Tuần 3: PDF, chunking, embeddings
# ---------------------------------------------------------------------------
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "hashing")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
HASHING_EMBEDDING_DIM = int(os.getenv("HASHING_EMBEDDING_DIM", "1024"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "40"))

# ---------------------------------------------------------------------------
# Tuần 4: retrieval + RAG
# ---------------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "4"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.25"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "extractive")

INDEX_FILE = INDEX_DIR / "store.json"
MANIFEST_FILE = INDEX_DIR / "manifest.json"

NO_EVIDENCE_MESSAGE = (
    "Tôi chưa tìm thấy bằng chứng phù hợp trong tài liệu để trả lời câu hỏi này. "
    "Bạn thử hỏi cụ thể hơn, hoặc tải thêm tài liệu liên quan."
)

# Prompt cho RAG: chỉ dùng ngữ cảnh được cung cấp, bắt buộc trích dẫn nguồn.
RAG_SYSTEM_PROMPT = (
    "Bạn là trợ lý hỏi đáp tài liệu. Bạn CHỈ được dùng thông tin trong phần NGỮ CẢNH.\n"
    "Quy tắc bắt buộc:\n"
    "1. Nếu ngữ cảnh không chứa câu trả lời, hãy nói: 'Tôi chưa tìm thấy bằng chứng trong tài liệu.' "
    "Không dùng kiến thức bên ngoài để lấp chỗ trống.\n"
    "2. Không bịa số liệu, tên riêng hay kết luận không có trong ngữ cảnh.\n"
    "3. Trả lời ngắn gọn bằng ngôn ngữ của câu hỏi.\n"
    "4. Cuối câu trả lời, liệt kê nguồn theo đúng mẫu [file, trang N].\n"
    "5. Nếu các đoạn ngữ cảnh mâu thuẫn nhau, nói rõ là có mâu thuẫn.\n"
)

# ---------------------------------------------------------------------------
# Tuần 6: API / UI
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
