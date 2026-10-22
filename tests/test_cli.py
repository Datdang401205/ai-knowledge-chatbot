"""Kiểm thử vòng lặp CLI mà không cần gõ tay trong terminal."""

from __future__ import annotations

import pytest

from chatbot.config import EMPTY_INPUT_MESSAGE, GOODBYE_MESSAGE, WELCOME_MESSAGE
from chatbot.main import is_exit_command, run_chatbot


class FakeTerminal:
    """Thu thập input có sẵn và ghi lại output đã in."""

    def __init__(self, inputs: list[str]) -> None:
        self.inputs = list(inputs)
        self.outputs: list[str] = []
        self.prompts: list[str] = []

    def read(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.inputs:
            raise EOFError
        return self.inputs.pop(0)

    def write(self, message: str) -> None:
        self.outputs.append(message)

    @property
    def spoken(self) -> list[str]:
        return [line for line in self.outputs if not line.startswith("Bạn>")]


@pytest.mark.parametrize("command", ["exit", "EXIT", "quit", " thoát ", "q"])
def test_is_exit_command_accepts_variants(command: str) -> None:
    assert is_exit_command(command) is True


@pytest.mark.parametrize("command", ["exit now", "keth", "hello"])
def test_is_exit_command_rejects_other_text(command: str) -> None:
    assert is_exit_command(command) is False


def test_run_chatbot_stops_on_exit_command() -> None:
    terminal = FakeTerminal(["xin chao", "exit", "cam on"])

    run_chatbot(input_func=terminal.read, output=terminal.write)

    assert terminal.outputs[0] == WELCOME_MESSAGE
    assert terminal.outputs[-1] == GOODBYE_MESSAGE
    assert "cam on" not in " ".join(terminal.spoken)


def test_run_chatbot_handles_empty_input() -> None:
    terminal = FakeTerminal(["", "   ", "exit"])

    run_chatbot(input_func=terminal.read, output=terminal.write)

    assert terminal.spoken.count(EMPTY_INPUT_MESSAGE) == 2


def test_run_chatbot_stops_on_end_of_input() -> None:
    terminal = FakeTerminal([])

    run_chatbot(input_func=terminal.read, output=terminal.write)

    assert terminal.outputs[-1] == GOODBYE_MESSAGE
