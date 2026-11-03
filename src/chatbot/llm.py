"""Gọi mô hình LLM local qua Ollama API.

Module này tách riêng logic gọi HTTP, xử lý lỗi, timeout để dễ test
và dễ thay thế sau này (khi chuyển sang RAG).
"""

from __future__ import annotations

import json
from typing import Any

import requests

from chatbot.config import (
    LLM_SYSTEM_PROMPT,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
)


class LLMError(Exception):
    """Lỗi khi gọi LLM."""


class LLMConnectionError(LLMError):
    """Không kết nối được tới Ollama."""


class LLMTimeoutError(LLMError):
    """Timeout khi gọi Ollama."""


class LLMResponseError(LLMError):
    """Phản hồi từ Ollama không hợp lệ."""


def _build_payload(question: str, system_prompt: str | None = None) -> dict[str, Any]:
    """Tạo payload gửi tới Ollama /api/chat."""
    if system_prompt is None:
        system_prompt = LLM_SYSTEM_PROMPT
    return {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "stream": False,
    }


def ask_llm(question: str, system_prompt: str | None = None) -> str:
    """Gửi câu hỏi tới Ollama và trả về nội dung trả lời.

    Args:
        question: Câu hỏi của người dùng.
        system_prompt: Tùy chỉnh system prompt (nếu None dùng mặc định).

    Raises:
        LLMConnectionError: Không kết nối được.
        LLMTimeoutError: Hết thời gian chờ.
        LLMResponseError: Phản hồi không hợp lệ/thiếu nội dung.
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = _build_payload(question, system_prompt)
    try:
        response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError as exc:
        raise LLMConnectionError(
            "Không kết nối được tới Ollama. Hãy đảm bảo Ollama đang chạy (ollama serve) "
            "và OLLAMA_BASE_URL đúng."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise LLMTimeoutError(
            f"Hết thời gian chờ ({OLLAMA_TIMEOUT_SECONDS}s) khi gọi Ollama. "
            "Hãy thử model nhỏ hơn hoặc tăng OLLAMA_TIMEOUT_SECONDS."
        ) from exc

    if response.status_code != 200:
        raise LLMResponseError(
            f"Ollama trả về lỗi HTTP {response.status_code}: {response.text[:200]}"
        )

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise LLMResponseError("Phản hồi từ Ollama không phải JSON hợp lệ.") from exc

    message = data.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LLMResponseError("Ollama trả về câu trả lời rỗng.")
    return content.strip()
