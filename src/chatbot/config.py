"""Cấu hình tập trung cho chatbot.

Mọi chuỗi có thể thay đổi (tên app, câu chào, câu thoát) nằm ở đây để
phần xử lý logic không bị rải hằng số rải rác khắp nơi.
"""

from __future__ import annotations

APP_NAME = "AI Knowledge Base Chatbot"
VERSION = "0.1.0"
MODE = "rule-based"

WELCOME_MESSAGE = (
    f"Xin chào! Tôi là {APP_NAME} (v{VERSION}, chế độ {MODE}).\n"
    "Tôi chưa đọc được tài liệu nào ở phiên bản này. "
    "Hỏi tôi về chức năng, hoặc gõ 'help' để xem các lệnh."
)

PROMPT = "Bạn> "
FALLBACK_REPLY = (
    "Tôi chưa hiểu câu hỏi của bạn. Phiên bản 0.1 chỉ trả lời theo luật cố định, "
    "gõ 'help' để xem các chủ đề mình hỗ trợ."
)
EMPTY_INPUT_MESSAGE = "Bạn chưa nhập nội dung. Hãy gõ câu hỏi hoặc 'exit' để thoát."
GOODBYE_MESSAGE = "Tạm biệt! Hẹn gặp lại."
ERROR_MESSAGE = "Đã xảy ra lỗi không mong muốn: {error}"

EXIT_COMMANDS = ("exit", "quit", "q", "thoat", "ket")
