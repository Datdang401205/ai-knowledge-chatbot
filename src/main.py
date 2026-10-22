"""Điểm chạy CLI: ``python src/main.py``.

Khi chạy trực tiếp một file bằng ``python src/main.py``, Python chỉ thêm
``src`` vào ``sys.path`` chứ không thêm thư mục cha. Vì vậy file này tự
thêm ``src`` vào ``sys.path`` để có thể ``import chatbot...``.

Cách chạy tương đương, không cần file này:
``cd src && python -m chatbot.main``
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chatbot.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
