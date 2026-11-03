"""AI Knowledge Base Chatbot.

Phiên bản 0.6 có hai phần: chat rule-based/LLM (Tuần 1–2) và RAG hỏi đáp
tài liệu (Tuần 3–6). Mỗi module giữ một trách nhiệm riêng:

- ``config``      : hằng số cấu hình của ứng dụng.
- ``intents``     : nhận diện ý định của người dùng và sinh câu trả lời.
- ``main``        : vòng lặp CLI, chỉ lo phần nhập/xuất.
- ``ingest``      : đọc PDF/TXT/MD.
- ``chunking``    : chia tài liệu thành đoạn có metadata.
- ``embeddings``  : biến đoạn văn thành vector.
- ``vector_store``: lưu và tìm vector.
- ``retrieval``   : truy xuất hybrid.
- ``generation``  : sinh câu trả lời có trích dẫn.
- ``rag_pipeline``: ghép toàn bộ luồng.
- ``api``         : FastAPI.
- ``streamlit_app``: giao diện web.
"""

from chatbot.config import APP_NAME, VERSION

__all__ = ["APP_NAME", "VERSION", "__version__"]

__version__ = VERSION
