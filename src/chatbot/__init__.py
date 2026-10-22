"""AI Knowledge Base Chatbot.

Phiên bản 0.1 chạy hoàn toàn bằng luật (rule-based), chưa dùng LLM.
Mỗi module giữ một trách nhiệm riêng:

- ``config``     : hằng số cấu hình của ứng dụng.
- ``intents``    : nhận diện ý định của người dùng và sinh câu trả lời.
- ``main``       : vòng lặp CLI, chỉ lo phần nhập/xuất.
"""

from chatbot.config import APP_NAME, VERSION

__all__ = ["APP_NAME", "VERSION", "__version__"]

__version__ = VERSION
