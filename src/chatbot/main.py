"""Vòng lặp CLI của chatbot.

Nguyên tắc của tuần này: ``main`` chỉ lo phần *giao tiếp* (đọc input, in output,
thoát chương trình); phần *quyết định trả lời* nằm trong ``chatbot.intents``.

Nhờ tách vậy, ``run_chatbot`` nhận sẵn hàm đọc input và hàm in ra, nên test
được hoàn toàn mà không cần gõ tay trong terminal.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable

from chatbot.config import (
    ERROR_MESSAGE,
    EXIT_COMMANDS,
    GOODBYE_MESSAGE,
    PROMPT,
    WELCOME_MESSAGE,
)
from chatbot.intents import normalize
from chatbot.responder import get_reply

logger = logging.getLogger(__name__)

Responder = Callable[[str], str]
InputFunc = Callable[[str], str]
OutputFunc = Callable[[str], None]

_EXIT_COMMANDS = frozenset(normalize(command) for command in EXIT_COMMANDS)


def is_exit_command(text: str) -> bool:
    """Kiểm tra người dùng có yêu cầu thoát chương trình hay không."""
    return normalize(text) in _EXIT_COMMANDS


def run_chatbot(
    responder: Responder = get_reply,
    input_func: InputFunc = input,
    output: OutputFunc = print,
) -> None:
    """Chạy vòng hội thoại cho tới khi người dùng yêu cầu thoát.

    Args:
        responder: hàm sinh câu trả lời, mặc định là ``chatbot.intents.get_reply``.
        input_func: hàm đọc một dòng input.
        output: hàm in một dòng ra màn hình.
    """
    output(WELCOME_MESSAGE)
    while True:
        try:
            user_input = input_func(PROMPT)
        except (EOFError, KeyboardInterrupt):
            output(GOODBYE_MESSAGE)
            break

        if is_exit_command(user_input):
            output(GOODBYE_MESSAGE)
            break

        output(responder(user_input))


def configure_console_encoding() -> None:
    """Ép console dùng UTF-8 để đọc/ghi được tiếng Việt trên Windows.

    Mặc định Windows dùng cp1252, in chữ có dấu sẽ ném UnicodeEncodeError.
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                logger.debug("Khong reconfigure duoc console encoding", exc_info=True)


def main() -> int:
    """Điểm vào của chương trình, trả về mã thoát cho hệ điều hành."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    configure_console_encoding()
    try:
        run_chatbot()
    except Exception as error:  # noqa: BLE001 - lớp lưới an toàn cuối cùng
        logger.exception("Chatbot dừng do lỗi ngoài dự kiến")
        print(ERROR_MESSAGE.format(error=error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
