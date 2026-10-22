"""Kiểm thử tích hợp: chạy thật file ``src/main.py`` như một tiến trình con."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = PROJECT_ROOT / "src" / "main.py"


def run_cli(user_input: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ENTRYPOINT)],
        input=user_input,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=PROJECT_ROOT,
        timeout=60,
    )


def test_cli_prints_welcome_and_exits_cleanly() -> None:
    result = run_cli("exit\n")

    assert result.returncode == 0
    assert "AI Knowledge Base Chatbot" in result.stdout


def test_cli_answers_greeting_in_vietnamese() -> None:
    result = run_cli("Xin chào\nexit\n")

    assert result.returncode == 0
    assert "Chào bạn" in result.stdout
