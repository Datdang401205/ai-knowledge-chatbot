# AI Knowledge Base Chatbot

Chatbot hỏi đáp tài liệu dựa trên LLM + RAG (Retrieval-Augmented Generation).
Dự án này được xây dựng theo lộ trình 6 tuần trong [`PROJECT_PLAN.md`](PROJECT_PLAN.md),
mỗi tuần một chặng nhỏ, ưu tiên **hiểu rồi mới dùng** và **đo chất lượng** thay vì làm giao diện.

> **Trạng thái hiện tại: Tuần 1 xong — CLI rule-based v0.1.**
> Chưa có LLM, chưa đọc PDF, chưa có vector store. Mục tiêu của tuần này là
> dựng được nền móng Python + Git + test mà mọi phần sau sẽ dựa vào.

---

## 1. Mục tiêu sản phẩm

Người dùng tải tài liệu PDF lên, đặt câu hỏi bằng tiếng Việt hoặc tiếng Anh và
nhận câu trả lời **dựa trên nội dung tài liệu**, kèm **trích dẫn nguồn**.
Khi tài liệu không có bằng chứng phù hợp, chatbot phải nói rõ *chưa tìm thấy bằng chứng*
thay vì tự bịa.

---

## 2. Yêu cầu môi trường

| Công cụ | Phiên bản | Ghi chú |
|---|---|---|
| Python | 3.11 hoặc 3.12 | Đã kiểm tra với 3.12.7 |
| pip | kèm theo Python | Kiểm tra bằng `python -m pip --version` |
| Git | bất kỳ | Dùng để lưu lịch sử và làm portfolio |

Không cần cài Ollama ở tuần 1. Sẽ cài ở Tuần 2 nếu phần cứng đáp ứng.

---

## 3. Cài đặt

```powershell
# 1. Tạo môi trường ảo (tách biệt dependency của project khỏi máy)
python -m venv .venv

# 2. Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1

# 3. Cài thư viện cho môi trường này
python -m pip install -r requirements.txt
```

Nếu PowerShell báo lỗi khi kích hoạt, chạy:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Kiểm tra đã kích hoạt đúng: `python -c "import sys; print(sys.prefix)"` phải trỏ vào
`...AI_Knowledge_Base\.venv`.

---

## 4. Cách chạy

Cách 1 — chạy file launcher (không cần kích hoạt venv nếu dùng đường dẫn đầy đủ):

```powershell
.\.venv\Scripts\python.exe src\main.py
```

Cách 2 — chạy như một module (đã kích hoạt venv):

```powershell
python src\main.py
# hoặc
cd src
python -m chatbot.main
```

Mẫu hội thoại:

```text
Xin chào! Tôi là AI Knowledge Base Chatbot (v0.1.0, chế độ rule-based).
...
Bạn> Xin chào
Chào bạn! Tôi có thể giải thích chức năng của chatbot này. Bạn muốn biết gì?
Bạn> exit
Tạm biệt! Hẹn gặp lại.
```

Lệnh thoát: `exit`, `quit`, `q`, `thoát`.

### Về tiếng Việt trên Windows

Windows mặc định dùng bảng mã `cp1252`, in tiếng Việt sẽ lỗi
`UnicodeEncodeError`. Code đã tự ép console về UTF-8 trong
`configure_console_encoding()`. Nếu vẫn lỗi khi *paste* từ file, chạy:

```powershell
chcp 65001
```

---

## 5. Cấu trúc thư mục

```text
AI_Knowledge_Base/
├── data/
│   └── sample_docs/           # Tài liệu mẫu (Tuần 3)
├── reports/                   # Báo cáo đánh giá (Tuần 5)
├── src/
│   ├── main.py                # Launcher: python src/main.py
│   └── chatbot/
│       ├── __init__.py
│       ├── config.py          # Hằng số cấu hình
│       ├── intents.py         # Nhận diện intent + sinh câu trả lời
│       └── main.py            # Vòng lặp CLI
├── tests/
│   ├── test_intents.py
│   ├── test_cli.py
│   └── test_cli_entrypoint.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
└── PROJECT_PLAN.md
```

Quy ước quan trọng: **`main` chỉ lo giao tiếp, logic nằm ở module khác.**

```text
input() ──► run_chatbot ──► get_reply ──► find_intent ──► reply
                                              ▲
                        (không có I/O trong chuỗi này → dễ test)
```

---

## 6. Chatbot v0.1 hoạt động thế nào

1. **Chuẩn hóa** (`normalize`): chữ thường, bỏ dấu tiếng Việt, bỏ dấu câu, gộp khoảng trắng.
   Nhờ vậy `"Xin chào"`, `"xin chào!"`, `"chao"` đều ra cùng một kết quả.
2. **Tách token**: `"xin chao ban"` → `("xin", "chao", "ban")`.
3. **Khớp cụm từ** (`_contains_phrase`): intent khớp nếu một *cụm từ liên tiếp* trong
   câu bằng với từ khóa của intent.
   Không dùng `keyword in text` vì `"hi" in "khi nào"` là `True` → khớp nhầm.
4. **Sinh trả lời**: trả `reply` của intent tìm được; không khớp thì trả `FALLBACK_REPLY`.

Các intent hiện có: `greeting`, `capabilities`, `upload_document`, `thanks`, `query_document`.

---

## 7. Chạy test

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Kết quả hiện tại: **30 test passed**.

Ba tầng test trong dự án:

| Tệp | Tầng | Kiểm tra gì |
|---|---|---|
| `test_intents.py` | unit | Hàm `normalize`, `find_intent`, `get_reply` |
| `test_cli.py` | unit + giả lập terminal | Vòng lặp CLI với input/output giả |
| `test_cli_entrypoint.py` | integration | Chạy thật `src/main.py` như tiến trình con |

Kỹ thuật đáng học ở `test_cli.py`: `run_chatbot` nhận sẵn `input_func` và `output`
qua tham số nên test được **không cần gõ tay** — đây là *dependency injection*.

---

## 8. Git

```powershell
git init
git add .
git commit -m "feat: week 1 - rule-based chatbot CLI with tests and README"
```

Quy tắc: một tính năng = một commit nhỏ, không commit `.env`, `.venv/`, `__pycache__/`
(đã cấu hình trong `.gitignore`).

---

## 9. Bài tập Tuần 1 (bắt buộc để đóng Tuần 1)

> 📘 **Tài liệu học chi tiết:** [`docs/BAI_HOC_TUAN_1.md`](docs/BAI_HOC_TUAN_1.md) —
> giải thích từng bước của Tuần 1, kiến thức nền, 2 bug thật đã gặp, đáp án bài tập
> và 12 câu tự kiểm tra. Đọc file này trước khi làm bài tập bên dưới.

### Bài 1 — Tự thêm một intent mới

Thêm intent `feedback`:

1. Mở `src/chatbot/intents.py`.
2. Thêm một `Intent(...)` mới vào `INTENTS` — **nhớ đặt trước `query_document`**
   nếu từ khóa có thể trùng (thứ tự khớp quyết định intent nào thắng).
3. Chạy lại `python src/main.py` và thử gõ "góp ý của tôi là ..."
4. Thêm test tương ứng trong `tests/test_intents.py`.

### Bài 2 — Giải thích `if __name__ == "__main__":`

Khi chạy `python src/main.py`, Python đặt `__name__ = "__main__"`.
Khi file được `import` (ví dụ `from chatbot.main import main`), `__name__` là `"chatbot.main"`.
Vì vậy khối `if __name__ == "__main__":` **chỉ chạy khi file được thực thi trực tiếp**,
không chạy khi file bị import — điều kiện cần thiết để một module vừa là thư viện
vừa là chương trình chạy được mà không lặp vô hạn.

### Bài 3 — Chạy chương trình mà không cần chép lại từng bước

Tự nhớ thứ tự: tạo venv → kích hoạt → cài requirements → chạy.
Ghi lại các lệnh đó vào mục "Cài đặt" của README (nếu README của bạn khác bản này).

### Bài 4 — Ôn lại Python

Tự trả lời được các câu trong `PROJECT_PLAN.md` mục 9 (Python và backend):
list/dict/tuple/set, `return` vs `print`, vì sao dùng virtual environment,
API/HTTP/JSON, exception và logging để làm gì.

---

## 10. Kế hoạch Tuần 2 (xem trước)

Thay phần trả lời theo luật bằng LLM:

- Cài Ollama, chọn model nhỏ phù hợp phần cứng.
- Gọi model từ Python qua API local (`http://localhost:11434`).
- Tách cấu hình model/URL/timout ra `config.py` (đọc từ biến môi trường, dùng `.env`).
- Xử lý timeout, lỗi kết nối, câu trả lời rỗng.
- Viết prompt yêu cầu trả lời bằng đúng ngôn ngữ và thừa nhận khi không biết.

---

## 11. Giới hạn hiện tại (nói rõ để không tuyên bố quá)

- Chưa dùng LLM: câu trả lời đến từ từ khóa cố định.
- Chưa đọc PDF, chưa có embeddings, chưa có vector store.
- Chưa có trích dẫn nguồn (sẽ có ở Tuần 4).
- Chưa có UI/API (sẽ có ở Tuần 6).

---

## 12. Bản quyền / giấy phép

Chưa chọn license. Sẽ thêm khi hoàn thiện dự án.
