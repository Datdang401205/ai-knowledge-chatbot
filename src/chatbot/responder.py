"""Tích hợp LLM vào luồng trả lời của chatbot.

Thêm chế độ LLM (chat với Ollama). Vẫn giữ nguyên rule-based để backward compatible.
Có thể bật bằng biến môi trường CHATBOT_MODE=llm hoặc gọi trực tiếp get_llm_reply.
"""

from __future__ import annotations

from chatbot.config import MODE
from chatbot.intents import get_reply as get_rule_reply
from chatbot.llm import (
    LLMConnectionError,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
    ask_llm,
)


def get_llm_reply(user_input: str) -> str:
    """Trả lời bằng LLM với xử lý lỗi thân thiện."""
    try:
        return ask_llm(user_input)
    except LLMConnectionError as exc:
        return f"{exc}\nGợi ý: chạy 'ollama serve' hoặc 'ollama run qwen2.5:3b'."
    except LLMTimeoutError as exc:
        return str(exc)
    except LLMResponseError as exc:
        return f"Lỗi phản hồi từ LLM: {exc}"
    except LLMError as exc:
        return f"Lỗi LLM: {exc}"


def get_reply(user_input: str) -> str:
    """Sinh câu trả lời theo chế độ cấu hình.

    - rule-based: dùng luật cố định (như phiên bản 0.1)
    - llm: dùng Ollama để sinh câu trả lời
    """
    if MODE.lower() == "llm":
        return get_llm_reply(user_input)
    return get_rule_reply(user_input)
