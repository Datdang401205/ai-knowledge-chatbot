"""Kiểm thử xử lý lỗi khi gọi LLM."""

from __future__ import annotations

from chatbot import llm, responder


def test_get_llm_reply_connection_error(monkeypatch) -> None:
    def raise_conn(*args, **kwargs):
        raise llm.LLMConnectionError("Không kết nối được tới Ollama.")

    monkeypatch.setattr(responder, "ask_llm", raise_conn)
    reply = responder.get_llm_reply("xin chao")
    assert "Không kết nối được" in reply
    assert "ollama" in reply.lower()


def test_get_llm_reply_timeout_error(monkeypatch) -> None:
    def raise_timeout(*args, **kwargs):
        raise llm.LLMTimeoutError("Hết thời gian chờ")

    monkeypatch.setattr(responder, "ask_llm", raise_timeout)
    reply = responder.get_llm_reply("xin chao")
    assert "Hết thời gian chờ" in reply


def test_get_llm_reply_response_error(monkeypatch) -> None:
    def raise_resp(*args, **kwargs):
        raise llm.LLMResponseError("rỗng")

    monkeypatch.setattr(responder, "ask_llm", raise_resp)
    reply = responder.get_llm_reply("xin chao")
    assert "Lỗi phản hồi" in reply


def test_responder_uses_rule_based_by_default(monkeypatch) -> None:
    monkeypatch.setattr(responder, "MODE", "rule-based")
    reply = responder.get_reply("xin chao")
    assert reply != ""
    # rule-based vẫn trả lời theo intent
    assert "chào" in reply.lower()
