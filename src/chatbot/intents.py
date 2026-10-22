"""Nhận diện ý định (intent) và sinh câu trả lời cho chatbot v0.1.

Cách hoạt động:

1. Chuẩn hóa câu vào (lowercase, bỏ dấu, bỏ dấu câu, gộp khoảng trắng).
2. Tách thành danh sách từ (token).
3. So khớp từ khóa của từng intent với các cụm từ liên tiếp trong câu.
4. Trả lời bằng reply của intent tìm được, hoặc fallback nếu không khớp.

Tại sao khớp theo *cụm từ* chứ không dùng ``keyword in text``?
Vì ``"hi" in "khi nào"`` là True, sẽ khớp nhầm. So khớp theo token tránh lỗi này.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from chatbot.config import EMPTY_INPUT_MESSAGE, FALLBACK_REPLY

_PUNCTUATION = ".,!?;:\"'()[]{}-_/\\…“”‘’@#$%^&*+=<>|~`"
_TRANSLATION = {ord(char): None for char in _PUNCTUATION}


@dataclass(frozen=True)
class Intent:
    """Một nhóm câu hỏi có cùng kiểu trả lời."""

    name: str
    keywords: tuple[str, ...]
    reply: str


INTENTS: tuple[Intent, ...] = (
    Intent(
        name="greeting",
        keywords=("xin chao", "chao ban", "chao", "hello", "hi", "hey", "good morning"),
        reply="Chào bạn! Tôi có thể giải thích chức năng của chatbot này. Bạn muốn biết gì?",
    ),
    Intent(
        name="capabilities",
        keywords=(
            "giup duoc gi",
            "ban lam duoc gi",
            "chuc nang",
            "khuyen nghi",
            "huong dan",
            "help",
            "capabilities",
        ),
        reply=(
            "Chatbot này hiện ở phiên bản rule-based: nhận diện chủ đề theo từ khóa "
            "(chào hỏi, chức năng, tải tài liệu, cảm ơn) và trả lời cố định. "
            "Tuần sau sẽ thay phần trả lời bằng LLM."
        ),
    ),
    Intent(
        name="upload_document",
        keywords=("tai len", "upload", "them tai lieu", "nap tai lieu", "tai lieu moi"),
        reply=(
            "Ở phiên bản 0.1 chưa có chức năng tải tài liệu lên. "
            "Tính năng đọc PDF và lưu vào vector store sẽ được xây dựng ở Tuần 3."
        ),
    ),
    Intent(
        name="thanks",
        keywords=("cam on", "cam on ban", "thanks", "thank you"),
        reply="Không có gì! Nếu bạn cần thêm thông tin, cứ hỏi tiếp nhé.",
    ),
    Intent(
        name="query_document",
        keywords=("tai lieu nay", "tai lieu", "tong hop", "tom tat", "ket qua", "nghien cuu gi"),
        reply=(
            "Hiện tôi chưa có tài liệu nào trong kho kiến thức để tra cứu. "
            "Khi có pipeline PDF -> chunks -> embeddings -> retrieval, tôi sẽ trả lời kèm trích dẫn nguồn."
        ),
    ),
)


def normalize(text: str) -> str:
    """Chuẩn hóa câu để so khớp ổn định.

    - Chuyển chữ hoa thành chữ thường.
    - Bỏ dấu tiếng Việt (nên "chao" cũng khớp "chào").
    - Bỏ dấu câu.
    - Gộp khoảng trắng liên tiếp và cắt khoảng trắng hai đầu.
    """
    lowered = text.lower()
    without_punctuation = lowered.translate(_TRANSLATION)
    decomposed = unicodedata.normalize("NFD", without_punctuation)
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(without_marks.split())


def _phrase_tokens(text: str) -> tuple[str, ...]:
    """Chuyển một cụm từ thành danh sách token đã chuẩn hóa."""
    return tuple(normalize(text).split())


def _contains_phrase(tokens: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    """Kiểm tra ``phrase`` có xuất hiện liên tiếp trong ``tokens`` hay không."""
    phrase_length = len(phrase)
    if phrase_length == 0 or phrase_length > len(tokens):
        return False
    return any(
        tokens[start : start + phrase_length] == phrase
        for start in range(len(tokens) - phrase_length + 1)
    )


KEYWORD_INDEX: dict[str, tuple[tuple[str, ...], ...]] = {
    intent.name: tuple(_phrase_tokens(keyword) for keyword in intent.keywords)
    for intent in INTENTS
}


def find_intent(text: str) -> Intent | None:
    """Trả về intent khớp với ``text``, hoặc ``None`` nếu không có."""
    tokens = _phrase_tokens(text)
    if not tokens:
        return None
    for intent in INTENTS:
        for phrase in KEYWORD_INDEX[intent.name]:
            if _contains_phrase(tokens, phrase):
                return intent
    return None


def get_reply(user_input: str) -> str:
    """Sinh câu trả lời cho một câu đầu vào của người dùng.

    Đây là hàm "logic thuần": không in ra màn hình, không đọc bàn phím.
    Nhờ vậy hàm này có thể kiểm thử tự động bằng pytest.
    """
    if not user_input or not user_input.strip():
        return EMPTY_INPUT_MESSAGE
    intent = find_intent(user_input)
    if intent is None:
        return FALLBACK_REPLY
    return intent.reply
