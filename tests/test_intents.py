"""Kiểm thử phần nhận diện intent và sinh câu trả lời."""

from __future__ import annotations

import pytest

from chatbot.config import EMPTY_INPUT_MESSAGE, FALLBACK_REPLY
from chatbot.intents import find_intent, get_reply, normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Xin chào", "xin chao"),
        ("  Xin   Chào  bạn ", "xin chao ban"),
        ("Cảm ơn!", "cam on"),
        ("chao ban, ban khoe?", "chao ban ban khoe"),
        ("", ""),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


@pytest.mark.parametrize(
    "question",
    ["Xin chào", "hello", "Chào bạn", "chao ban", "hey", "Good morning"],
)
def test_find_intent_detects_greeting(question: str) -> None:
    intent = find_intent(question)
    assert intent is not None
    assert intent.name == "greeting"


@pytest.mark.parametrize(
    "question",
    ["khi nào nào giao hàng", "chi phí này là bao nhiêu", "cho tôi biết thông tin"],
)
def test_find_intent_ignores_unrelated_questions(question: str) -> None:
    assert find_intent(question) is None


def test_get_reply_returns_intent_reply() -> None:
    reply = get_reply("xin chao")
    assert reply != FALLBACK_REPLY
    assert reply == find_intent("xin chao").reply


def test_get_reply_handles_empty_input() -> None:
    assert get_reply("") == EMPTY_INPUT_MESSAGE
    assert get_reply("   ") == EMPTY_INPUT_MESSAGE


def test_get_reply_falls_back_when_no_intent_matches() -> None:
    assert get_reply("abcxyz 12345") == FALLBACK_REPLY
