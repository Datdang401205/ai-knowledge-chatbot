# Bài học Tuần 1 — Python project & chatbot CLI rule-based

> Tài liệu này dành cho **người tự học**. Đọc theo thứ tự từ trên xuống, mỗi phần có:
> *kiến thức cần biết* → *code thật trong project* → *tại sao phải làm vậy* → *bài tập tự làm*.
>
> Sau khi đọc xong, bạn phải trả lời được phần "Tự kiểm tra" ở cuối tài liệu mà **không cần mở lại code**.

---

## Mục lục

1. [Bức tranh tổng thể: Tuần 1 làm gì](#1-bức-tranh-tổng-thể-tuần-1-làm-gì)
2. [Bản đồ file đã tạo](#2-bản-đồ-file-đã-tạo)
3. [Bước 1 — Cấu trúc project và virtual environment](#3-bước-1--cấu-trúc-project-và-virtual-environment)
4. [Bước 2 — Kiểm tra Python và pip](#4-bước-2--kiểm-tra-python-và-pip)
5. [Bước 3 — Ôn Python: từng khái niệm nằm ở đâu](#5-bước-3--ôn-python-từng-khái-niệm-nằm-ở-đâu)
6. [Bước 4 — Chatbot CLI rule-based v0.1](#6-bước-4--chatbot-cli-rule-based-v01)
7. [Bước 5 — Tách logic khỏi `main()`](#7-bước-5--tách-logic-khỏi-main)
8. [Bước 6 — Input rỗng, lệnh thoát, exception](#8-bước-6--input-rỗng-lệnh-thoát-exception)
9. [Bước 7 — Git](#9-bước-7--git)
10. [Bước 8 — README](#10-bước-8--readme)
11. [Hai bug thật đã gặp (đọc để học)](#11-hai-bug-thật-đã-gặp-đọc-để-học)
12. [Test: tại sao 30 test lại quan trọng](#12-test-tại-sao-30-test-lại-quan-trọng)
13. [Bài tập Tuần 1 + gợi ý lời giải](#13-bài-tập-tuần-1--gợi-ý-lời-giải)
14. [Tự kiểm tra (12 câu)](#14-tự-kiểm-tra-12-câu)
15. [Bảng tra cứu nhanh](#15-bảng-tra-cứu-nhanh)
16. [Kết nối Tuần 1 với RAG ở Tuần 4](#16-kết-nối-tuần-1-với-rag-ở-tuần-4)

---

## 1. Bức tranh tổng thể: Tuần 1 làm gì

Tuần 1 **không** tạo ra chatbot thông minh. Nó tạo ra **nền móng kỹ thuật** mà 5 tuần sau sẽ xây trên đó.

```
Tuần 1 (nền)          Tuần 2          Tuần 3              Tuần 4              Tuần 5-6
Python + Git   →   Gọi LLM   →   PDF/Chunk/Embed   →   Retrieval + RAG   →   Đánh giá + API/UI
      ↑
      └── Nếu bước này làm cẩu thả, mọi bước sau đều phải viết lại
```

Bốn nguyên tắc của plan mà Tuần 1 thực sự rèn luyện:

| Nguyên tắc | Thực hành cụ thể trong Tuần 1 |
|---|---|
| "Hiểu rồi mới dùng framework" | Tuần 1 **không** dùng LangChain/FastAPI/Streamlit — chỉ Python thuần |
| "Đo chất lượng thay vì làm giao diện" | Viết 30 test + kiểm chứng bằng lệnh thật, không "chắc là chạy" |
| "Không tuyên bố chưa kiểm chứng" | README mục 11 liệt kê rõ giới hạn hiện tại |
| "Mỗi tính năng = một commit nhỏ" | 2 commit: code (2b6f5c8) + tài liệu (2cb11e4) |

**Kết quả kiểm chứng của Tuần 1:**

```text
$ .\.venv\Scripts\python.exe -m pytest
..............................                                           [100%]
30 passed in 0.28s

$ git log --oneline
2cb11e4 docs: mark week 1 as done and record decisions in PROJECT_PLAN
2b6f5c8 feat(week-1): rule-based chatbot CLI with tests, README and project setup
```

---

## 2. Bản đồ file đã tạo

```text
AI_Knowledge_Base/
├── src/
│   ├── main.py                    # launcher: python src/main.py
│   └── chatbot/
│       ├── __init__.py           # gói, export APP_NAME/VERSION
│       ├── config.py              # hằng số cấu hình (dòng 1-27)
│       ├── intents.py             # LOGIC: intent + câu trả lời (dòng 1-141)
│       └── main.py                # I/O: vòng lặp CLI (dòng 1-94)
├── tests/
│   ├── test_intents.py            # unit test cho logic
│   ├── test_cli.py                # unit test có fake terminal
│   └── test_cli_entrypoint.py     # integration test: chạy thật tiến trình con
├── data/sample_docs/              # tài liệu mẫu (Tuần 3)
├── reports/                       # báo cáo đánh giá (Tuần 5)
├── pyproject.toml                 # metadata + cấu hình pytest
├── requirements.txt               # dependency cho người khác cài lại
├── .gitignore                     # loại .venv, __pycache__, .env khỏi Git
├── .gitattributes                 # chuẩn hoá line ending
├── .env.example                   # TÊN biến môi trường (không chứa giá trị thật)
├── README.md                      # hướng dẫn 12 mục
└── PROJECT_PLAN.md                # plan gốc + nhật ký tiến độ
```

**Nguyên tắc "không tạo file rỗng":** plan nói rõ *"Không tạo file rỗng chỉ để đủ cây thư mục; mỗi module được thêm khi bắt đầu dùng"*.
→ Vì vậy `ingest.py`, `chunking.py`, `embeddings.py`, `vector_store.py`, `retrieval.py`, `generation.py`, `rag_pipeline.py` **chưa tồn tại**. Tuần 3 tạo `ingest.py` khi thực sự cần.

---

## 3. Bước 1 — Cấu trúc project và virtual environment

### 3.1 Vì sao cần virtual environment

Python dùng **một thư mục `site-packages` chung** cho toàn hệ thống. Hãy tưởng tượng:

```text
Bạn đang làm 2 project:
  project-a cần requests==2.28
  project-b cần requests==2.31   ← xung đột, không thể cài cả hai vào cùng chỗ
```

Virtual environment tạo một thư mục riêng, trong đó `pip` cài vào `site-packages` **của riêng thư mục đó**. Các project không đụng nhau.

```text
máy bạn
├── Python hệ thống        (pip cài chung - đừng đụng vào)
│   └── site-packages/
├── project-a/.venv/Scripts/        ← riêng biệt
│   └── Lib/site-packages/
└── project-b/.venv/Scripts/        ← riêng biệt
    └── Lib/site-packages/
```

### 3.2 Các lệnh và ý nghĩa từng lệnh

```powershell
python -m venv .venv
```
- `python -m venv` = chạy module `venv` của Python → tạo môi trường ảo.
- `.venv` = tên thư mục (`.` ẩn để gợi ý "cục bộ, không phải thư viện hệ thống"; `.gitignore` đã loại nó).

```powershell
.\.venv\Scripts\Activate.ps1
```
- Trên Windows, "kích hoạt" = thêm `.venv\Scripts` vào đầu `PATH` → gõ `python` là Python trong venv.
- Trên macOS/Linux: `source .venv/bin/activate`.

```powershell
python -m pip install -r requirements.txt
```
- `-m pip` (không phải `pip` trần) → dùng đúng pip của interpreter đang active. Đây là nguyên tắc số 1 của bài phỏng vấn.
- `-r requirements.txt` → đọc danh sách dependency từ file.

### 3.3 Cách kiểm tra "đã kích hoạt đúng"

```powershell
python -c "import sys; print(sys.prefix)"
```
→ phải in `C:\...\AI_Knowledge_Base\.venv`
→ Nếu in ra đường dẫn Python hệ thống → venv **chưa** được kích hoạt.

### 3.4 Vì sao `.venv` không được commit lên Git

- `.venv/` là **kết quả sinh ra** từ `requirements.txt` + Python version, không phải mã nguồn.
- Nó nặng hàng trăm MB và chứa binary khác nhau trên mỗi OS.
- Người khác clone repo của bạn phải tự tạo lại: đây là 1 dòng lệnh trong README.

→ Nguyên tắc: **Repo chứa *mô tả* môi trường (`requirements.txt`), không chứa *bản sao* môi trường (`.venv/`).**

### 3.5 Câu hỏi phỏng vấn liên quan

> *"Vì sao dùng virtual environment?"*

Trả lời mẫu (nói bằng lời của bạn):
> "Vì mỗi project cần một tập thư viện với phiên bản khác nhau. Virtual environment cô lập `site-packages` của từng project, tránh xung đột phiên bản và tránh phá hỏng môi trường Python hệ thống. Ngoài ra `.venv` bị gitignore vì nó được sinh ra từ `requirements.txt`, còn repo chỉ giữ lại file mô tả dependency."

---

## 4. Bước 2 — Kiểm tra Python và pip

```powershell
python --version        # Python 3.12.7
python -m pip --version # pip 26.2 (nâng cấp trong .venv)
```

**Cần kiểm tra gì và tại sao:**

| Kiểm tra | Vì sao quan trọng |
|---|---|
| Phiên bản Python | Plan yêu cầu 3.11/3.12. Code dùng `X \| None` (PEP 604) và `list[str]` → cần 3.10+ |
| pip hoạt động trong venv | Nếu sai, mọi `pip install` sau này cài nhầm chỗ |
| Phiên bản pip | pip quá cũ có thể không hỗ trợ `pyproject.toml` |

**Bài học về `pyproject.toml`:** từ pip 21.3+, `pyproject.toml` là cách khai báo metadata chuẩn. File này của bạn làm 3 việc:

```toml
[project]
name = "ai-knowledge-chatbot"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
pythonpath = ["src"]        # ← quan trọng nhất Tuần 1
testpaths = ["tests"]
```

`pythonpath = ["src"]` nói cho pytest: "khi import, hãy thêm thư mục `src` vào đường dẫn tìm module".
→ Nhờ đó `from chatbot.intents import ...` trong test hoạt động mà **không cần** `pip install -e .`.

---

## 5. Bước 3 — Ôn Python: từng khái niệm nằm ở đâu

Plan yêu cầu ôn: *hàm, chuỗi, điều kiện, vòng lặp, `return`, import, exception*.
Thay vì đọc lý thuyết rời rạc, mỗi khái niệm đã được **đặt vào đúng chỗ** trong code thật.

### 5.1 Hàm và `return`

```python
# src/chatbot/intents.py:130
def get_reply(user_input: str) -> str:
    if not user_input or not user_input.strip():
        return EMPTY_INPUT_MESSAGE
    intent = find_intent(user_input)
    if intent is None:
        return FALLBACK_REPLY
    return intent.reply
```

**`return` vs `print()` — khác biệt cốt lõi:**

| | `return` | `print()` |
|---|---|---|
| Bản chất | **trả về giá trị** cho lời gọi | **hiển thị** ra màn hình |
| Lưu được không | Có → gán vào biến | Không |
| Test được không | Dễ — assert trên giá trị trả về | Khó — phải bắt stdout |
| Dùng để ghép | Được (trả về để nối chuỗi) | Không được |

```python
reply = get_reply("xin chao")   # ✅ lấy được giá trị
print(get_reply("xin chao"))     # chỉ để xem
```

**Vì sao quan trọng cho RAG:** Tuần 4, hàm `answer(question)` phải trả về cả *câu trả lời* lẫn *danh sách nguồn* → bắt buộc dùng `return` (hoặc `dataclass`), không dùng `print`.

**Type hints** (`user_input: str) -> str`): không bắt buộc cho Python chạy, nhưng:
- IDE gợi ý được, bạn đọc code nhanh hơn;
- `mypy` kiểm tra được;
- **Kỹ thuật phỏng vấn**: `Intent | None` nói rõ "hàm này có thể không trả kết quả".

### 5.2 Chuỗi (string)

```python
# src/chatbot/intents.py:81
def normalize(text: str) -> str:
    lowered = text.lower()                              # chữ thường
    without_punctuation = lowered.translate(_TRANSLATION)  # bỏ dấu câu
    decomposed = unicodedata.normalize("NFD", without_punctuation)  # tách dấu ra
    without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(without_marks.split())               # gộp + cắt khoảng trắng
```

Các thao tác chuỗi bạn cần thuộc:

| Thao tác | Ví dụ | Kết quả |
|---|---|---|
| `.lower()` | `"Xin Chào".lower()` | `"xin chào"` |
| `.strip()` | `"  abc  ".strip()` | `"abc"` |
| `.split()` | `"a  b c".split()` | `['a','b','c']` (tự bỏ chuỗi rỗng) |
| `.split(sep)` | `"a,b".split(",")` | `['a','b']` |
| `" ".join(list)` | `" ".join(['a','b'])` | `"a b"` |
| `.translate(dict)` | bỏ ký tự trong dict | xem bên dưới |
| `.startswith()` / `.endswith()` | kiểm tra đầu/cuối | `True`/`False` |
| f-string | `f"{APP_NAME} v{VERSION}"` | nội suy biến |

**`.split()` không có đối số** là "bí kíp" hay: `"  a   b  ".split()` → `['a','b']` — tự bỏ mọi khoảng trắng thừa và chuỗi rỗng. Kết hợp với `" ".join(...)` → gọn 1 dòng là xong chuẩn hóa khoảng trắng:

```python
return " ".join(text.split())
```

**`.translate()` — bỏ dấu câu:**

```python
_PUNCTUATION = ".,!?;:\"'()[]{}-_/\\…“”‘’@#$%^&*+=<>|~`"
_TRANSLATION = {ord(char): None for char in _PUNCTUATION}

"Xin chào!".translate(_TRANSLATION)   # → "Xin chào"
```
`ord(char)` lấy mã Unicode của ký tự; `None` nghĩa là **xoá** ký tự đó.

**Bỏ dấu tiếng Việt bằng Unicode (chuyên sâu):**

```python
"chào".normalize("NFD")   # → 'ch' + 'a' + 'o' + dấu sắc (mã U+0301) tách riêng ra
```
Chuẩn Unicode có 2 dạng cho chữ "có dấu": **NFC** (dựng sẵn: `á` = 1 ký tự U+00E1) và **NFD** (tách dấu: `a` + U+0301).
`normalize("NFD")` → tách dấu ra; `unicodedata.combining(c)` → `True` nếu ký tự đó là dấu; lọc bỏ là xong.

```python
without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
# "chào" → "chao"
```

**Vì sao cần bỏ dấu?** Ba lý do rất thực tế:
1. Người dùng gõ "chao ban" (không dấu) vẫn khớp intent "chào bạn".
2. Tuần 3, text trích xuất từ PDF có thể lẫn ký tự Unicode lạ → cần chuẩn hóa.
3. Tuần 4, khi so khớp câu hỏi tiếng Việt với tài liệu, việc bỏ dấu làm tăng recall.

### 5.3 Điều kiện (if/elif/else)

```python
if not user_input or not user_input.strip():
    return EMPTY_INPUT_MESSAGE
```

- `not X` = đảo ngược giá trị chân lý.
- `A or B` = đúng nếu **một trong hai** đúng (short-circuit: nếu `A` đúng thì không evaluate `B`).
- `A and B` = đúng nếu **cả hai** đúng.

**`is None` chứ không phải `== None`:**

```python
if intent is None:      # ✅ đúng
if intent == None:      # ❌ không
```
`== None` dùng phép so sánh `__eq__` → chậm hơn và có thể bị `__eq__` ghi đè sai. `is` kiểm tra **cùng thể hiện** (identity) — đúng ngữ nghĩa cho `None`.

### 5.4 Vòng lặp (for) và `break`/`return` sớm

```python
# src/chatbot/intents.py:118
def find_intent(text: str) -> Intent | None:
    tokens = _phrase_tokens(text)
    if not tokens:
        return None
    for intent in INTENTS:
        for phrase in KEYWORD_INDEX[intent.name]:
            if _contains_phrase(tokens, phrase):
                return intent          # ← thoát sớm, không cần break
    return None
```

**Vòng lặp lồng nhau** ở đây: duyệt `INTENTS` (ngoài) × duyệt từ khóa của intent (trong).
Khi tìm thấy → `return` ngay, vòng lặp kết thúc luôn (giống `break` nhưng thoát cả hàm).

**Hiệu năng:** `KEYWORD_INDEX` (dòng 112) được **tính một lần lúc import**, không tính lại mỗi lần gọi hàm.
Nếu viết `_phrase_tokens(keyword)` ngay trong vòng lặp, mỗi câu hỏi sẽ chuẩn hóa lại toàn bộ từ khóa → lãng phí CPU.

```python
KEYWORD_INDEX: dict[str, tuple[tuple[str, ...], ...]] = {
    intent.name: tuple(_phrase_tokens(keyword) for keyword in intent.keywords)
    for intent in INTENTS
}
```

**Dict comprehension** (`{k: v for ...}`) — cú pháp ngắn gọn tạo dict. Tương tự:
- List comprehension: `[x for x in items if điều_kiện]`
- Set comprehension: `{x for x in items}`
- Generator: `(x for x in items)`

### 5.5 Import

```python
from chatbot.config import EMPTY_INPUT_MESSAGE, FALLBACK_REPLY
```
Đây là **absolute import** — đường dẫn đầy đủ từ gốc package.

**Vì sao `import chatbot.config` lại tìm thấy được?**

| Khi chạy | `sys.path` chứa gì | Cách sửa |
|---|---|---|
| `python src/main.py` | `src/` (Python tự thêm thư mục của file đang chạy) | Import được luôn |
| `python src/chatbot/main.py` | `src/chatbot/` → import `chatbot` **thất bại** | Không chạy kiểu này |
| `pytest` | thư mục gốc project | Thêm `pythonpath = ["src"]` trong `pyproject.toml` |
| `python -m chatbot.main` từ `src/` | `src/` | Cách sạch nhất |

**Vì sao lại có file launcher `src/main.py`?**

Khi bạn chạy `python src/main.py`, Python chỉ thêm `src/` vào `sys.path` — đủ để `import chatbot` hoạt động! Nhưng launcher vẫn tự chèn `src` vào `sys.path` cho chắc:
```python
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
```
→ Khi đó launcher **có thể đặt ở bất kỳ đâu** và vẫn import được. Đây là mẹo nhỏ nhưng rất hữu ích.

### 5.6 Exception

```python
# src/chatbot/main.py:46
try:
    user_input = input_func(PROMPT)
except (EOFError, KeyboardInterrupt):
    output(GOODBYE_MESSAGE)
    break
```

**Exception nào xảy ra khi nào:**

| Exception | Khi nào | Xử lý trong project |
|---|---|---|
| `KeyboardInterrupt` | Người dùng nhấn Ctrl+C | Thoát gọn, in lời chào |
| `EOFError` | Ctrl+Z / Ctrl+D / hết stdin (pipe) | Thoát gọn |
| `Exception` | Lỗi bất kỳ khác | Bắt ở `main()`, log + trả mã thoát `1` |

**Nguyên tắc vàng:**
1. **Bắt lỗi cụ thể, không bắt trần.** `except:` trần bắt cả `KeyboardInterrupt` lẫn lỗi hệ thống → rất khó debug.
2. **Chỉ bắt lỗi bạn biết cách xử lý.** Không dùng `try/except` để "giấu" lỗi.
3. **Lớp lưới an toàn cuối cùng** ở `main()`:
```python
try:
    run_chatbot()
except Exception as error:
    logger.exception("Chatbot dừng do lỗi ngoài dự kiến")
    print(ERROR_MESSAGE.format(error=error), file=sys.stderr)
    return 1
```
`logger.exception(...)` tự in cả **traceback** + stack — cần thiết khi debug.

---

## 6. Bước 4 — Chatbot CLI rule-based v0.1

### 6.1 "Rule-based" nghĩa là gì?

Không có LLM, không có hiểu ngữ nghĩa. Chỉ có **bảng từ khóa → câu trả lời cố định**:

```python
# src/chatbot/intents.py:34
INTENTS = (
    Intent(name="greeting", keywords=("xin chao", "chao ban", "hello", ...),
           reply="Chào bạn! Tôi có thể giải thích chức năng của chatbot này. Bạn muốn biết gì?"),
    Intent(name="capabilities", keywords=("giup duoc gi", "chuc nang", "help", ...),
           reply="Chatbot này hiện ở phiên bản rule-based: ..."),
    Intent(name="upload_document", keywords=("tai len", "upload", ...),
           reply="Ở phiên bản 0.1 chưa có chức năng tải tài liệu lên. ..."),
    Intent(name="thanks", keywords=("cam on", "thanks", ...),
           reply="Không có gì! Nếu bạn cần thêm thông tin, cứ hỏi tiếp nhé."),
    Intent(name="query_document", keywords=("tai lieu nay", "tai lieu", "tom tat", ...),
           reply="Hiện tôi chưa có tài liệu nào trong kho kiến thức để tra cứu. ..."),
)
```

**Đây là bản "giả lập" (mock) của RAG**: Tuần 2 sẽ thay `reply` cố định bằng LLM, Tuần 4 sẽ thay phần "tìm intent" bằng "tìm chunk".
Nhờ tách kiến trúc sẵn, việc thay chỉ là **đổi nội dung hàm**, không phải viết lại chương trình.

### 6.2 `dataclass` — gói dữ liệu gọn gàng

```python
# src/chatbot/intents.py:25
@dataclass(frozen=True)
class Intent:
    name: str
    keywords: tuple[str, ...]
    reply: str
```

**`@dataclass`** tự sinh `__init__`, `__repr__`, `__eq__` từ các thuộc tính khai báo.
So với viết thủ công:
```python
class Intent:
    def __init__(self, name, keywords, reply):
        self.name = name
        self.keywords = keywords
        self.reply = reply
```
→ `dataclass` ngắn hơn, ít lỗi copy-paste, và tự có type hints.

**`frozen=True`** → object **bất biến** (immutable): sửa `intent.name = "x"` sẽ ném `FrozenInstanceError`.
Vì sao cần? Vì `INTENTS` là hằng số dùng chung; bất biến thì không ai sửa nhầm được, và thread-safe.

**`tuple[str, ...]`** — kiểu tuple có thay đổi độ dài (`...` = variadic). `tuple` bất biến → an toàn hơn `list` cho hằng số.

### 6.3 Luồng khớp từ khóa (chi tiết nhất của Tuần 1)

```text
"Xin chào, tôi muốn tải lên tài liệu"
   │
   ▼  normalize()            "xin chao toi muon tai len tai lieu"
   │  lower + bỏ dấu + bỏ dấu câu + gộp khoảng trắng
   ▼  _phrase_tokens()        ("xin","chao","toi","muon","tai","len","tai","lieu")
   │  .split()
   ▼  _contains_phrase()     so từng cụm 2-3 từ liên tiếp
   │
   ├─ "xin chao"  ✓ khớp → Intent(name="greeting") → reply
   └─ (các intent khác chưa tới)
```

**Hàm khớp cụm từ (`intents.py:101`):**

```python
def _contains_phrase(tokens: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    phrase_length = len(phrase)
    if phrase_length == 0 or phrase_length > len(tokens):
        return False
    return any(
        tokens[start : start + phrase_length] == phrase
        for start in range(len(tokens) - phrase_length + 1)
    )
```

Giải thích từng phần:
- `phrase_length == 0` → cụm rỗng thì không khớp gì cả (tránh khớp "mọi thứ").
- `phrase_length > len(tokens)` → cụm dài hơn câu → không thể khớp.
- `range(len(tokens) - phrase_length + 1)` → duyệt **mọi vị trí bắt đầu** có thể.
- `tokens[start : start + phrase_length] == phrase` → so sánh ** lát cắt (slice)** với cụm từ khóa.
- `any(...)` → `True` nếu có **ít nhất một** vị trí khớp.

**Ví dụ minh họa:**
```python
tokens = ("khi", "nao", "giao", "hang")
_contains_phrase(tokens, ("hi",))        # False  ← "hi" không phải token riêng
_contains_phrase(tokens, ("khi", "nao")) # True
```

### 6.4 Vì sao KHÔNG dùng `keyword in text`

```python
if "hi" in "khi nào giao hàng":   # → True  (SAI!)
```
`in` kiểm tra **chuỗi con**, không kiểm tra từ.
Người dùng hỏi "khi nào giao hàng?" sẽ bị bot chào hỏi. Lỗi này rất dễ bỏ sót nếu không có test.

→ Đây là lý do test `test_find_intent_ignores_unrelated_questions` tồn tại.

### 6.5 `None` = "không tìm thấy", không phải "trả rỗng"

```python
def find_intent(text: str) -> Intent | None:   # None nghĩa là không khớp intent nào
def get_reply(user_input: str) -> str:         # luôn trả về một câu trả lời
```
Phân biệt này quan trọng: `find_intent` trả **cấu trúc** (intent) hoặc `None`;
`get_reply` trả **chuỗi** luôn — vì người dùng không bao giờ nên nhận `None` từ chatbot.

---

## 7. Bước 5 — Tách logic khỏi `main()`

Đây là **bài học quan trọng nhất của Tuần 1**.

### 7.1 Cách làm sai

```python
def main():
    while True:
        q = input("Bạn> ").strip()
        if "xin chào" in q:
            print("Chào bạn!")
        elif "cảm ơn" in q:
            print("Không có gì!")
        else:
            print("Tôi không hiểu")
```
Vấn đề:
- Không test được → phải gõ tay mỗi lần đổi logic.
- Muốn dùng lại phần "trả lời" cho web/API sau này → phải copy lại.
- Thêm intent mới → sửa trong một hàm dài, dễ phá chỗ khác.

### 7.2 Cách đúng trong project này

```text
main.py:39  run_chatbot()   →  I/O: đọc input, in output, quyết định thoát
                │
                └──► intents.py:130  get_reply()   →  logic thuần: str → str
                          │
                          └──► intents.py:118  find_intent()  →  str → Intent | None
                                    │
                                    └──► intents.py:101  _contains_phrase()
```

**Quy tắc kiểm tra:** nếu bạn thấy `print(` hoặc `input(` bên trong `get_reply`/`find_intent` → đã vi phạm.

### 7.3 Dependency injection — kỹ thuật giúp test được

```python
# src/chatbot/main.py:39
def run_chatbot(
    responder: Responder = get_reply,
    input_func: InputFunc = input,
    output: OutputFunc = print,
) -> None:
```

Ba tham số này là **"kết nối"** mà vòng lặp dùng để nói chuyện với thế giới bên ngoài.
Mặc định = hành vi thật. Khi test, ta thay bằng bản giả:

```python
# tests/test_cli.py
class FakeTerminal:
    def __init__(self, inputs): self.inputs = list(inputs); self.outputs = []
    def read(self, prompt): 
        if not self.inputs: raise EOFError
        return self.inputs.pop(0)
    def write(self, message): self.outputs.append(message)

def test_run_chatbot_stops_on_exit_command():
    terminal = FakeTerminal(["xin chao", "exit", "cam on"])
    run_chatbot(input_func=terminal.read, output=terminal.write)
    assert terminal.outputs[-1] == GOODBYE_MESSAGE
```

**Đây là chuẩn mực của Python (duck typing)**: không cần class/interface trừu tượng — chỉ cần object có đúng phương thức.
`FakeTerminal` không kế thừa gì từ `input`/`print`, nhưng vẫn chạy được.

**Tên gọi:**
- **Dependency Injection (DI)**: truyền phụ thuộc từ bên ngoài vào thay vì tự tạo bên trong.
- **Monkeypatch**: thay thế hàm/hàm viết tạm thời trong lúc test.
- **Test double**: đối tượng giả (ở đây là `FakeTerminal`).

### 7.4 Tại sao điều này quyết định Tuần 4

Tuần 4, bạn sẽ viết:

```python
def run_chatbot(responder=answer_question, ...):   # giữ nguyên cấu trúc Tuần 1
```
Chỉ cần thay `responder`. Toàn bộ CLI, test, cấu trúc file giữ nguyên.
Nếu Tuần 1 bạn gộp logic vào `main()`, Tuần 4 bạn phải viết lại **từ đầu** — mất 2–3 ngày.

---

## 8. Bước 6 — Input rỗng, lệnh thoát, exception

### 8.1 Input rỗng

```python
# src/chatbot/intents.py:133
if not user_input or not user_input.strip():
    return EMPTY_INPUT_MESSAGE
```
Hai điều kiện: `not user_input` (chuỗi rỗng `""`) hoặc `not user_input.strip()` (chỉ toàn khoảng trắng).
→ Trả lời thân thiện thay vì crash với `IndexError`.

### 8.2 Lệnh thoát

```python
# src/chatbot/config.py:25
EXIT_COMMANDS = ("exit", "quit", "q", "thoat", "ket")

# src/chatbot/main.py:26
_EXIT_COMMANDS = frozenset(normalize(command) for command in EXIT_COMMANDS)

# src/chatbot/main.py:34
def is_exit_command(text: str) -> bool:
    return normalize(text) in _EXIT_COMMANDS
```

Ba điểm kỹ thuật đáng học:
1. **So khớp sau khi `normalize()`** → người dùng gõ `" Thoát "` vẫn thoát được.
2. **`frozenset`** — set bất biến. Tra cứu trong set O(1), nhanh hơn list.
3. **Tính sẵn 1 lần ở module level** (`_EXIT_COMMANDS`), không `normalize()` lại mỗi vòng lặp.

### 8.3 Mã thoát (exit code) — chuẩn CLI

```python
# src/chatbot/main.py:80
def main() -> int:
    ...
    try:
        run_chatbot()
    except Exception as error:
        logger.exception("Chatbot dừng do lỗi ngoài dự kiến")
        print(ERROR_MESSAGE.format(error=error), file=sys.stderr)
        return 1
    return 0

# src/chatbot/main.py:93
if __name__ == "__main__":
    raise SystemExit(main())
```

| Mã | Ý nghĩa | Quy ước Unix |
|---|---|---|
| `0` | Thành công | `0` = OK |
| `1` | Lỗi chung | `1` = lỗi chung |

**`raise SystemExit(main())`** = "thoát chương trình với mã thoát là giá trị trả về của `main()`".
`SystemExit` là cách chuẩn để kết thúc chương trình với mã lỗi cụ thể.

**`print(..., file=sys.stderr)`**: thông báo lỗi đi ra **stderr**, thông tin bình thường đi ra **stdout`.
Khi bạn `python main.py > out.txt`, mọi thông báo lỗi vẫn hiện trên màn hình.

### 8.4 Giải thích `if __name__ == "__main__":`

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

**`__name__` là biến đặc biệt** mà Python tự gán khi nạp một module:

| Cách nạp | Giá trị `__name__` | Khối `if` chạy? |
|---|---|---|
| `python src/main.py` (chạy trực tiếp) | `"__main__"` | ✅ Có |
| `import chatbot.main` | `"chatbot.main"` | ❌ Không |
| `from chatbot.main import main` | `"chatbot.main"` | ❌ Không |
| Test import module của bạn | tên module | ❌ Không |

**Tại sao cần?** Vì một file Python nên vừa làm **thư viện** (để import) vừa làm **chương trình** (để chạy trực tiếp).
Nếu không có `if __name__ == "__main__":`, thì **mọi lần import** file đó cũng chạy lại toàn bộ code bên trong → ví dụ:

```python
# main.py KHÔNG có if __name__ == "__main__":
while True:
    user_input = input("Bạn> ")   # ← khi test import, chương trình chạy vô hạn!
```

**Câu trả lời chuẩn cho phỏng vấn:**
> "Khi một file được thực thi trực tiếp bằng `python file.py`, Python đặt `__name__` bằng `'__main__'`. Khi file được `import`, `__name__` là tên đầy đủ của module. Khối `if __name__ == '__main__':` đảm bảo code chỉ chạy khi file được chạy như chương trình, không chạy lại mỗi lần module được import. Điều này rất quan trọng khi viết test: nếu thiếu, việc `import` module trong test sẽ kích hoạt toàn bộ logic chương trình."

---

## 9. Bước 7 — Git

### 9.1 Ba lệnh cơ bản

```powershell
git init                          # tạo repo trong thư mục hiện tại
git add .                         # stage tất cả thay đổi
git commit -m "feat(week-1): ..." # ghi lại một mốc
```

### 9.2 Ba vùng làm việc của Git

```text
Working Directory          Staging Area              Repository
(thư mục bạn đang sửa)  --git add-->   (khu vực chuẩn bị)  --git commit-->  (lịch sử)
        ↑                                                              │
        └──────────────────── git restore --staged ────────────────────┘
```

### 9.3 `.gitignore` — 5 nhóm cần loại

```gitignore
# Python
__pycache__/          # bytecode sinh ra lúc chạy, tái sinh được
*.py[cod]             # file bytecode .pyc .pyo .pyd
*.egg-info/           # metadata do setuptools sinh
build/  dist/
.venv/  venv/         # môi trường ảo - tái sinh từ requirements.txt

# Test / tooling
.pytest_cache/        # cache của pytest
.mypy_cache/  .ruff_cache/

# Secret & biến môi trường
.env                  # ⚠️ TUYỆT ĐỐI không commit

# IDE
.idea/  .vscode/  .DS_Store
```

**`.env` vs `.env.example`:**
- `.env` → chứa **giá trị thật** (API key) → **không** commit, chỉ để local.
- `.env.example` → chứa **tên biến + giá trị mẫu** → commit được, giúp người khác biết cần đặt biến gì.

```gitignore
.env
```
→ Git bỏ qua **mọi file** tên `.env` ở mọi thư mục.

### 9.4 Quy tắc commit tốt (áp dụng từ Tuần 1)

| Nên | Không nên |
|---|---|
| Một commit = một thay đổi logic | Commit tất cả mọi thứ cuối ngày |
| Message mô tả **ý định** (`feat: add feedback intent`) | Message vô nghĩa (`update`, `fix`) |
| Commit `.gitignore` + `.env.example` từ đầu | Để sót secret rồi commit sau |
| `git status` trước khi commit | Commit thẳng không kiểm tra |

**Conventional Commits** (dùng trong project này):

| Prefix | Dùng khi | Ví dụ |
|---|---|---|
| `feat` | Tính năng mới | `feat: add feedback intent` |
| `fix` | Sửa lỗi | `fix: handle EOFError on piped input` |
| `docs` | Chỉ sửa tài liệu | `docs: mark week 1 as done` |
| `test` | Thêm/sửa test | `test: cover empty input case` |
| `refactor` | Đổi cấu trúc, không đổi hành vi | `refactor: extract intent matching` |
| `chore` | Việc vặt (deps, config) | `chore: add pytest to dev deps` |

### 9.5 Vì sao Git quan trọng cho CV

1. **Nhà tuyển dụng xem lịch sử commit** như bản CV thật — commit rõ ràng, đều đặn = "làm việc có hệ thống".
2. **Portfolio**: đẩy lên GitHub Public, README có ảnh demo.
3. **An toàn**: luôn có thể quay lại phiên bản trước khi thử ý tưởng mới.
4. **Quy trình nhóm (tuần 6)**: commit → pull request → review → merge là quy trình bạn sẽ gặp ở công ty.

```powershell
# Đẩy lên GitHub (làm ở Tuần 6)
git remote add origin https://github.com/<user>/ai-knowledge-chatbot.git
git push -u origin master
```
(Lưu ý: nhánh mặc định của Git hiện nay là `main`; nếu muốn đổi: `git branch -M main` trước khi push.)

---

## 10. Bước 8 — README

README là **tài liệu đầu tiên** người đánh giá (nhà tuyển dụng, giảng viên) đọc. 12 mục trong README hiện tại:

| Mục | Nội dung | Vì sao cần |
|---|---|---|
| 1 | Mục tiêu sản phẩm | Người đọc hiểu ngay project làm gì |
| 2 | Yêu cầu môi trường | Kiểm tra máy có đủ không |
| 3 | Cài đặt | Copy-paste là chạy |
| 4 | Cách chạy + mẫu hội thoại | Thấy được kết quả ngay |
| 5 | Cấu trúc thư mục | Định hướng cho người đọc code |
| 6 | Chatbot hoạt động thế nào | Giải thích thiết kế |
| 7 | Chạy test + bảng 3 tầng test | Chứng minh chất lượng |
| 8 | Git | Quy trình làm việc |
| 9 | **Bài tập Tuần 1** | Bạn tự học tiếp |
| 10 | Kế hoạch Tuần 2 | Minh họa tư duy có kế hoạch |
| 11 | **Giới hạn hiện tại** | Trung thực — không tuyên bố quá |
| 12 | Bản quyền | Rõ ràng pháp lý |

**Mục 11 là điểm ăn tiền trong phỏng vấn.** Một ứng viên nói "chatbot của tôi trả lời chính xác 95%" mà không kèm bằng chứng sẽ bị hỏi thêm. Còn ứng viên nói rõ:
> "Hiện tại chưa hỗ trợ PDF scan, bảng phức tạp, tài liệu rất dài và câu hỏi mơ hồ. Kết quả đánh giá sẽ bổ sung ở Tuần 5."

→ Đó là dấu hiệu của người làm được sản phẩm thật.

### Checklist README chuẩn (nhớ để dùng cho các tuần sau)

```text
[ ] Dự án làm gì (1-2 câu)
[ ] Ảnh/GIF demo ở đầu file
[ ] Yêu cầu cài đặt (phiên bản, OS)
[ ] Lệnh cài đặt + lệnh chạy (copy-paste được)
[ ] Ví dụ sử dụng thật (đầu vào → đầu ra)
[ ] Kiến trúc (sơ đồ hoặc mô tả luồng)
[ ] Cách chạy test + kết quả test thật
[ ] Cấu trúc thư mục
[ ] Quyết định thiết kế đáng chú ý
[ ] Hạn chế đã biết
[ ] Giấy phép
```

---

## 11. Hai bug thật đã gặp (đọc để học)

Bug là bài học tốt nhất. Hai lỗi dưới đây **đã xảy ra thật** khi dựng Tuần 1.

### 11.1 Bug 1 — `UnicodeEncodeError` khi in tiếng Việt trên Windows

```text
UnicodeEncodeError: 'charmap' codec can't encode character '\u1ebf' in position 54
```

**Nguyên nhân:** Windows console mặc định dùng bảng mã `cp1252` (Latin-1), không chứa đủ ký tự tiếng Việt như `ợ` (U+1EE3).

**Cách sửa trong project (`main.py:66`):**
```python
def configure_console_encoding() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                logger.debug("Khong reconfigure duoc console encoding", exc_info=True)
```

**Ba chi tiết đáng học:**
1. `getattr(stream, "reconfigure", None)` — kiểm tra object có phương thức không trước khi gọi. Khi test, `sys.stdout` có thể bị thay bằng object khác không có `reconfigure`.
2. `errors="replace"` — nếu vẫn có ký tự lạ, thay bằng `?` thay vì crash. **Luôn có fallback.**
3. Bọc trong `try/except` — không để việc chỉnh encoding làm hỏng chương trình.

**Mẹo thay thế:** `chcp 65001` trong PowerShell (đổi console sang UTF-8) hoặc chạy `python -X utf8 file.py`.

### 11.2 Bug 2 — Khớp từ khóa sai vì dùng `in` trên chuỗi

**Hiện tượng:** người dùng hỏi *"khi nào giao hàng?"* → bot trả lời *"Chào bạn!"*.

**Nguyên nhân:** `"hi" in "khi nào giao hàng"` → `True`.

**Cách sửa:** chuyển sang khớp theo **cụm từ liên tiếp trên token** (`_contains_phrase`).

**Bài học chung:**
> Trong project NLP/RAG, **naive string matching gần như luôn khớp sai**. Luôn nghĩ đến các ca đối lập và viết test cho chúng.

Các ca đối lập tương tự bạn sẽ gặp:
- "tai lieu" (không dấu) vs "tài liệu"
- "cong ty" trong "công ty"
- "báo cáo" vs "báo cáo cáo" (lặp chữ)

### 11.3 Một lỗi công cụ (file .md, ký tự Unicode)

Khi sửa `PROJECT_PLAN.md` bằng script, chuỗi tìm kiếm không khớp với nội dung file do một số ký tự tiếng Việt ở dạng **NFC** (dựng sẵn) trong file nhưng dạng **NFD** (tách dấu) trong script.

**Bài học:** khi so khớp chuỗi Unicode, hãy chuẩn hóa cả hai về cùng dạng trước khi so sánh:
```python
import unicodedata
text = unicodedata.normalize("NFC", text)
```

---

## 12. Test: tại sao 30 test lại quan trọng

### 12.1 Ba tầng test trong project

| Tệp | Tầng | Kiểm tra gì |
|---|---|---|
| `tests/test_intents.py` | **unit** | `normalize`, `find_intent`, `get_reply` — hàm thuần, gọi trực tiếp |
| `tests/test_cli.py` | **unit + test double** | Vòng lặp CLI với `FakeTerminal` |
| `tests/test_cli_entrypoint.py` | **integration** | Chạy thật `src/main.py` như tiến trình con, kiểm tra exit code + output |

### 12.2 Cấu trúc một test tốt

```python
def test_get_reply_handles_empty_input() -> None:
    assert get_reply("") == EMPTY_INPUT_MESSAGE
    assert get_reply("   ") == EMPTY_INPUT_MESSAGE
```

Đặc điểm:
- Tên test **mô tả hành vi**, không mô tả hàm: `test_get_reply_handles_empty_input` tốt hơn `test_get_reply_2`.
- **Một test = một hành vi.** Nếu test fail, bạn biết ngay chỗ nào hỏng.
- `assert` không có message → khi fail, pytest hiện giá trị thực vs giá trị mong đợi.

### 12.3 `@pytest.mark.parametrize` — test nhiều case gọn

```python
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Xin chào", "xin chao"),
        ("  Xin   Chào  bạn ", "xin chao ban"),
        ("Cảm ơn!", "cam on"),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected
```
Một hàm test → 3 trường hợp, mỗi case là **một test riêng** trong báo cáo pytest.
Dùng cho bảng dữ liệu dài, đặc biệt là **edge case** (biên).

### 12.4 Test tích hợp — chạy thật tiến trình con

```python
def run_cli(user_input: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ENTRYPOINT)],
        input=user_input, capture_output=True, text=True,
        encoding="utf-8", cwd=PROJECT_ROOT, timeout=60,
    )

def test_cli_prints_welcome_and_exits_cleanly() -> None:
    result = run_cli("exit\n")
    assert result.returncode == 0
    assert "AI Knowledge Base Chatbot" in result.stdout
```

**Vì sao cần test này?** Nó kiểm tra những thứ unit test không kiểm tra được:
- launcher `src/main.py` có import đúng không;
- encoding UTF-8 có hoạt động trong tiến trình thật không;
- mã thoát `0` có đúng không.

`sys.executable` = đường dẫn tới Python đang chạy test → test luôn dùng đúng interpreter (kể cả trong venv).

### 12.5 Test là "tài liệu sống"

Khi đọc `tests/test_intents.py`, bạn hiểu hành vi mong đợi của chatbot **mà không cần đọc code chính**.
Đây là lý do plan yêu cầu test ngay từ Tuần 1, không phải để dành cuối cùng.

### 12.6 pytest nâng cao (để biết trước cho Tuần 5)

```python
def test_retrieval_returns_expected_chunks(): ...
def test_answer_is_grounded_in_context(): ...
```
Khi bắt đầu Tuần 5, bạn sẽ viết test kiểu này cho chất lượng retrieval.

---

## 13. Bài tập Tuần 1 + gợi ý lời giải

> **Làm bài tập TRƯỚC khi đọc mục 14.** Học bằng cách tự làm sẽ hiệu quả hơn nhiều.
> Gợi ý được để dưới dạng `<details>` trong bản render, ở đây viết thẳng — cố gắng tự nghĩ trước.

### Bài 1 — Tự thêm intent mới (`feedback`)

**Mục tiêu:** tự thêm một intent mà không cần AI viết giúp.

**Các bước:**
1. Mở `src/chatbot/intents.py`.
2. Thêm `Intent` mới vào tuple `INTENTS` (dòng 34).
3. Chạy `python src/main.py` và thử.
4. Thêm test vào `tests/test_intents.py`.
5. Commit riêng.

**Gợi ý:**
```python
Intent(
    name="feedback",
    keywords=("gop y", "phan hoi", "feedback", "sai roi"),
    reply="Cảm ơn phản hồi của bạn! Hãy ghi rõ phần nào chưa đúng để tôi cải thiện.",
),
```

**Bẫy cần tránh:** nếu đặt `feedback` **sau** `query_document`, mà `query_document` có từ khóa `"tai lieu"`, thì câu *"góp ý về tài liệu này"* sẽ khớp `query_document` trước (vì `INTENTS` được duyệt theo thứ tự).
→ **Thứ tự trong `INTENTS` quyết định intent nào thắng.** Intent cụ thể hơn nên đặt trước.

**Test:**
```python
def test_find_intent_detects_feedback() -> None:
    intent = find_intent("góp ý của tôi là")
    assert intent is not None
    assert intent.name == "feedback"
```

### Bài 2 — Giải thích `if __name__ == "__main__":`

→ Xem mục **8.4**. Tự viết lại bằng lời của bạn (không copy).

**Tự kiểm tra:** tạo file `thu_nghiem.py`:
```python
print("module đang được load")
if __name__ == "__main__":
    print("chạy như chương trình")
```
Chạy `python thu_nghiem.py` → thấy cả 2 dòng.
Sau đó tạo `dung_thu_nghiem.py`:
```python
import thu_nghiem
```
Chạy `python dung_thu_nghiem.py` → **chỉ** thấy dòng đầu.
→ Đó là bằng chứng cụ thể cho câu trả lời của bạn.

### Bài 3 — Chạy chương trình không cần chép lại từng bước

Tự nhớ và chạy lại (không mở README):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\main.py
```
Ghi lại những gì bạn sai (ví dụ: quên kích hoạt venv, quên `-m` trước pip) vào nhật ký tiến độ.

### Bài 4 — Ôn lại Python (plan mục 9)

Trả lời bằng lời của bạn:
1. Khác nhau giữa list, dict, tuple, set?
2. `return` khác `print()` như thế nào?
3. Vì sao dùng virtual environment?
4. API, HTTP request, JSON là gì?
5. Xử lý exception và logging để làm gì?

### Bài 5 (thêm) — Refactor nhỏ

Thêm `src/chatbot/version.py` chứa `VERSION = "0.1.0"`, sửa `config.py` import từ đó, chạy test lại.
→ Bài học: **refactor = đổi cấu trúc, không đổi hành vi** → test tự chứng minh điều đó.

---

## 14. Tự kiểm tra (12 câu)

Trả lời xong rồi mới xem đáp án.

**Câu 1.** Vì sao không được commit thư mục `.venv/` lên Git?

**Câu 2.** `python -m pip install X` khác `pip install X` như thế nào?

**Câu 3.** `return "abc"` khác `print("abc")` như thế nào? Cho ví dụ nơi sự khác biệt này quan trọng.

**Câu 4.** Vì sao dùng `is None` thay vì `== None`?

**Câu 5.** Câu này cho ra kết quả gì?
```python
tokens = ("khi", "nao", "giao", "hang")
print("hi" in "khi nao giao hang")
```

**Câu 6.** Vì sao `run_chatbot` nhận `input_func` và `output` qua tham số?

**Câu 7.** `if __name__ == "__main__":` có ý nghĩa gì? Hậu quả nếu bỏ nó?

**Câu 8.** Vì sao cần `normalize()` trước khi so khớp từ khóa?

**Câu 9.** `intents.py` có dùng `print()` ở đâu không? Vì sao?

**Câu 10.** `.env` và `.env.example` khác nhau thế nào? File nào được commit?

**Câu 11.** `UnicodeEncodeError` khi in tiếng Việt trên Windows xảy ra vì lý do gì và sửa thế nào?

**Câu 12.** Nếu Tuần 4 thay hàm `get_reply` bằng hàm `answer_question` (có trả về cả nguồn), bạn cần sửa những chỗ nào trong project này? (Gợi ý: nghĩ xem CLI có cần đổi không.)

<details>
<summary>Đáp án</summary>

**1.** `.venv/` được sinh ra từ `requirements.txt` + phiên bản Python + hệ điều hành, không phải mã nguồn. Nó nặng, chứa binary khác nhau trên mỗi OS, và mỗi người clone repo phải tự tạo lại bằng 1 lệnh.

**2.** `python -m pip` chạy pip **từ interpreter Python đang active** → luôn cài vào đúng môi trường. `pip` trần dùng pip trên PATH → có thể cài nhầm vào Python hệ thống.

**3.** `return` trả **giá trị** cho lời gọi (gán được vào biến, test được bằng `assert`); `print` chỉ **hiển thị**, không lưu được, không test được bằng assert. Quan trọng ở Tuần 4: `answer_question()` phải `return` cả câu trả lời lẫn danh sách nguồn để CLI in ra và test kiểm tra.

**4.** `is None` kiểm tra **identity** (cùng thể hiện) — đúng ngữ nghĩa cho `None`, nhanh hơn, và không bị ảnh hưởng nếu ai đó ghi đè `__eq__`. `== None` gọi phép so sánh và có thể trả về `True` cho object khác.

**5.** `True`. Và đó chính là **bug** — "khi nào" chứa chuỗi con "hi" → bot chào nhầm. Sửa bằng cách khớp theo cụm từ trên token (`_contains_phrase`).

**6.** Để **test được mà không cần gõ tay** (dependency injection). Test truyền vào `FakeTerminal` giả lập input/output → chạy nhanh, ổn định, không phụ thuộc môi trường terminal.

**7.** Khi file được chạy trực tiếp (`python file.py`), `__name__ == "__main__"` → khối code chạy. Khi file được `import`, `__name__` là tên module → khối code bị bỏ qua. Bỏ đi thì mỗi lần import sẽ chạy lại toàn bộ logic (ví dụ vòng `while True: input()` → treo máy khi test import).

**8.** Vì: (a) người dùng gõ không dấu / sai dấu câu / thừa khoảng trắng vẫn phải khớp; (b) text trích xuất từ PDF có ký tự lạ cần chuẩn hóa; (c) Tuần 4 cần so khớp câu hỏi với tài liệu.

**9.** Không. `intents.py` chỉ có logic thuần — nhận `str`, trả `str`. Mọi `print` nằm ở `main.py`. Nhờ vậy `test_intents.py` gọi trực tiếp hàm và assert được.

**10.** `.env` chứa **giá trị thật** (API key) → gitignore, không commit. `.env.example` chứa **tên biến + giá trị mẫu** → commit được để người khác biết cần cấu hình gì.

**11.** Windows console mặc định dùng bảng mã `cp1252` không chứa đủ ký tự tiếng Việt. Sửa: `stream.reconfigure(encoding="utf-8", errors="replace")` cho stdin/stdout/stderr trong `configure_console_encoding()`, hoặc chạy `chcp 65001` / `python -X utf8`.

**12.** CLI **không cần đổi gì cấu trúc** — chỉ đổi giá trị mặc định của tham số `responder` trong `run_chatbot` (`responder=answer_question`), và bổ sung phần in nguồn trong `run_chatbot`. Đây chính là lý do Tuần 1 yêu cầu tách logic khỏi `main()`. Test cũng vẫn chạy được vì `responder` được inject.

</details>

---

## 15. Bảng tra cứu nhanh

### Lệnh thường dùng

```powershell
python -m venv .venv                        # tạo venv
.\.venv\Scripts\Activate.ps1                # kích hoạt venv
python -m pip install -r requirements.txt   # cài dependency
python -m pip list                          # xem đã cài gì
.\.venv\Scripts\python.exe -m pytest        # chạy test (không cần kích hoạt venv)
.\.venv\Scripts\python.exe -m pytest -v     # test chi tiết từng case
python src\main.py                          # chạy chatbot
git status / git add . / git commit -m "..."  # git
```

### Từ khóa Python đã dùng trong Tuần 1

| Từ khóa / khái niệm | Ý nghĩa | Ví dụ trong project |
|---|---|---|
| `def` | định nghĩa hàm | `def get_reply(...)` |
| `return` | trả về giá trị | `intents.py:141` |
| `if / elif / else` | rẽ nhánh | `intents.py:135` |
| `for ... in ...` | duyệt | `intents.py:121` |
| `any()` | đúng nếu có ít nhất 1 phần tử đúng | `intents.py:106` |
| `all()` | đúng nếu tất cả đúng | (chưa dùng) |
| `tuple` / `list` / `dict` / `set` | cấu trúc dữ liệu | `INTENTS`, `KEYWORD_INDEX` |
| `frozenset` | set bất biến | `_EXIT_COMMANDS` |
| `|` (PEP 604) | union type: `Intent \| None` | `intents.py:118` |
| `@dataclass(frozen=True)` | class dữ liệu bất biến | `Intent` |
| `try / except / finally` | xử lý lỗi | `main.py:46`, `main.py:85` |
| `raise` | ném lỗi | `raise SystemExit(main())` |
| `with` | context manager (đóng file an toàn) | (Tuần 3 sẽ dùng) |
| `import` / `from ... import ...` | nạp module | đầu mỗi file |

### Python idioms đã dùng (rất hay gặp trong code thật)

```python
x = value if condition else default      # ternary
f"{name} v{version}"                      # f-string
[name for name in items if cond]          # list comprehension
{item.name: expr for item in items}       # dict comprehension
{a, b} = 1, 2                             # unpack
text = " ".join(words.split())             # chuẩn hóa khoảng trắng
result = func() or default                 # fallback giá trị falsy
*args, **kwargs                            # biến tham số linh hoạt
```

---

## 16. Kết nối Tuần 1 với RAG ở Tuần 4

Đây là phần quan trọng nhất để bạn **không phải làm lại từ đầu** khi đến Tuần 4.

| Khái niệm Tuần 1 | Tương đương Tuần 4 | Ghi chú |
|---|---|---|
| `get_reply(text)` → `str` | `answer(question)` → `Answer` | Cùng vai trò: nhận câu hỏi, trả kết quả |
| `find_intent()` tìm "ý định" | `retrieve()` tìm "chunk" | Cùng bản chất: **tìm trong kho** |
| `INTENTS` (bảng từ khóa) | `chunks` + `embeddings` | Cùng bản chất: **kho tri thức** |
| `reply` cố định | LLM sinh câu trả lời có grounding | Chỉ khác *cách sinh*, không khác *vị trí trong kiến trúc* |
| `run_chatbot(responder=...)` | `run_chatbot(responder=answer_question, ...)` | **Cùng cấu trúc, chỉ đổi hàm** |
| `FakeTerminal` trong test | `FakeRetriever` trong test | Cùng kỹ thuật DI |
| `normalize()` | `normalize()` + `chunk_text()` | Chuẩn hóa là nền của mọi bước sau |
| `find_intent` chọn intent đầu tiên khớp | `retrieve` chọn top-k chunk theo điểm vector | Khác thuật toán, cùng cấu trúc hàm |

**Sơ đồ kiến trúc Tuần 1 (sẽ mở rộng ở Tuần 4):**

```text
Tuần 1 (hiện tại)                  Tuần 4 (mục tiêu)
──────────────────                  ───────────────────
input → run_chatbot                input → run_chatbot
          │                                  │
          ▼                                  ▼
     get_reply()                       answer_question()
          │                                  │
          ▼                                  ▼
     find_intent()        ───────►     retrieve_chunks()
          │                                  │
          ▼                                  ▼
      INTENTS (từ khóa)                 Vector Store
          │                                  │
          ▼                                  ▼
     reply cố định                  LLM + trích dẫn nguồn
```

**Bài học mang về:**
> Tuần 1 tưởng như "chỉ làm CLI vô nghĩa", nhưng nó đã chứa **toàn bộ khung xương** của RAG.
> Đó chính là lý do plan viết: *"Hiểu rồi mới dùng framework"* và *"Ở giai đoạn học ban đầu, có thể viết phiên bản CLI trước UI/API để hiểu luồng dữ liệu."*

---

## Phụ lục — Câu hỏi phỏng vấn trong plan và nơi tìm câu trả lời

| Câu hỏi (plan mục 9) | Đã trả lời ở đâu |
|---|---|
| Vì sao dùng virtual environment? | Mục 3.5 |
| `return` vs `print`? | Mục 5.1 |
| Xử lý exception và logging để làm gì? | Mục 5.6 |
| list/dict/tuple/set khác nhau? | Mục 5.4, bảng 15 |
| API, HTTP request, JSON? | Sẽ học ở Tuần 6 (FastAPI) |

---

*Ngày hoàn thành: 2026-10-22 · Tuần 1 — Python project và chatbot CLI.*
*Nhớ cập nhật nhật ký tiến độ trong `PROJECT_PLAN.md` sau mỗi buổi học.*
