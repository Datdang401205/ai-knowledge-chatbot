# Bài học Tuần 6 — API, giao diện và đóng gói demo

> Mục tiêu: đưa pipeline RAG ra khỏi terminal, cho người khác dùng được qua HTTP
> và giao diện web, đồng thời giữ đúng logic và bảo mật cơ bản.

---

## 1. Vì sao tách API và UI?

```text
Streamlit UI ─┐
              ├──► RAGPipeline (một logic duy nhất)
FastAPI ──────┘
```

Nếu UI gọi thẳng các hàm ingest/retrieval, API gọi một bản khác, hai bên sẽ lệch
nhau. `RAGPipeline` là **điểm vào duy nhất**; API và UI chỉ là lớp giao tiếp.

Lợi ích:

- Sửa logic một chỗ, cả hai giao diện cập nhật.
- Test pipeline một lần thay vì test lại qua từng UI.
- Có thể thay UI mà không đụng retrieval.

---

## 2. FastAPI

File: `src/chatbot/api.py`.

### 2.1 App factory

```python
def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    app = FastAPI(...)
    app.state.pipeline = pipeline or RAGPipeline.from_index()
    ...
    return app
```

Vì sao không tạo pipeline cố định ở module?

- Test có thể truyền pipeline tạm (tài liệu trong `tmp_path`).
- Tránh đọc index thật khi chạy test.
- Dễ thay cấu hình.

`app = create_app()` ở cuối file để `uvicorn chatbot.api:app` dùng được.

### 2.2 Endpoint

| Method | Path | Input | Output |
|---|---|---|---|
| GET | `/` | — | tên, version, link docs |
| GET | `/health` | — | status, số chunk, số nguồn |
| GET | `/stats` | — | thống kê index |
| POST | `/ask` | `AskRequest` | `AskResponse` |
| POST | `/documents` | file upload | `IndexReport` |

### 2.3 Pydantic model và validation

```python
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
```

FastAPI tự trả **422** nếu:

- Thiếu `question`.
- `question` rỗng hoặc quá dài.
- `top_k` ngoài khoảng 1–20.

Nhờ vậy không phải viết tay hàng loạt `if`. Nhưng vẫn cần kiểm tra `strip()` vì
`"   "` dài 3 ký tự, vượt `min_length=1` mà vẫn rỗng về nghĩa.

### 2.4 Upload an toàn

Các bước trong `/documents`:

1. Lấy **tên file gốc**, bỏ đường dẫn (`Path(file.filename).name`) để tránh
   path traversal.
2. Chỉ cho `.pdf`, `.txt`, `.md`.
3. Đọc tối đa `MAX_UPLOAD_BYTES + 1` để biết có vượt giới hạn không, tránh đọc
   cả file 1 GB vào RAM.
4. Ghi vào thư mục tạm **với tên gốc** để citation giữ đúng tên.
5. Gọi `pipeline.add_document(path)`.
6. Thư mục tạm tự xoá sau khi xong.

Mã lỗi:

- **415**: định dạng không hỗ trợ.
- **413**: file quá lớn.
- **422**: file rỗng hoặc không đọc được.

### 2.5 Vì sao giữ tên gốc?

Nếu lưu `upload_abc123.pdf`, câu trả lời sẽ trích dẫn tên vô nghĩa. Người dùng mất
khả năng đối chiếu. Đây là lỗi dễ mắc khi dùng `NamedTemporaryFile`.

---

## 3. Streamlit

File: `src/chatbot/streamlit_app.py`.

### 3.1 Session state

Streamlit chạy lại script sau mỗi tương tác. Nếu tạo pipeline mỗi lần, index bị
nạp lại và upload biến mất. Vì vậy:

```python
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline.from_index()
```

`messages` cũng lưu trong session state để hội thoại không mất khi rerun.

### 3.2 Sidebar

- Hiển thị số chunk, số tài liệu, embedder.
- Nút build lại index từ `data/sample_docs`.
- Upload tài liệu mới.
- Nút xoá hội thoại.

Sau khi build/upload, gọi `st.rerun()` để cập nhật số liệu.

### 3.3 Chat

```python
question = st.chat_input("Nhập câu hỏi...")
if question:
    ...
    answer = pipeline.ask(question)
    st.markdown(answer.text)
```

Nguồn được đặt trong `st.expander` để không chiếm chỗ.

### 3.4 Trạng thái và lỗi

- `st.spinner` khi build/ask.
- `st.warning` khi index rỗng hoặc PDF scan.
- `st.error` khi upload sai.

Nguyên tắc: người dùng phải biết **đang chờ gì** và **sai ở đâu**, không im lặng.

---

## 4. Logging và bảo mật

### 4.1 Không log nội dung nhạy cảm

Project log:

- Lỗi ingest (tên file, lý do).
- Lỗi LLM (timeout, HTTP status).
- Cảnh báo index hỏng.

Không log:

- Toàn bộ câu hỏi/người dùng.
- Nội dung tài liệu.
- API key.

### 4.2 `.env`

- `.env` nằm trong `.gitignore`.
- Chỉ commit `.env.example`.
- Không đưa API key vào code, README hoặc log.

### 4.3 Giới hạn cần nói rõ

- Chưa có xác thực người dùng.
- Chưa có rate limiting.
- Chưa chống prompt injection triệt để.
- File upload chỉ kiểm tra đuôi và kích thước, chưa quét malware.

Đây là demo local, không phải hệ thống production.

---

## 5. Đóng gói và tài liệu

### 5.1 README phải trả lời được

1. Cài gì?
2. Chạy lệnh nào?
3. Kỳ vọng thấy gì?
4. Kết quả đo được là bao nhiêu?
5. Giới hạn là gì?

Nếu người khác clone repo mà không chạy được theo README, project chưa hoàn thành.

### 5.2 Cấu trúc test

| Tầng | Ví dụ | Mục đích |
|---|---|---|
| Unit | chunking, embedding | Logic nhỏ, chạy nhanh |
| Integration | pipeline, API | Nhiều module phối hợp |
| End-to-end | `src/main.py` | Tiến trình thật |

Tổng hiện tại: **84 test**.

### 5.3 Demo còn thiếu

- Chưa có ảnh/GIF/video.
- Chưa deploy.
- Chưa chạy `eval` với Ollama thật.

Không được ghi "đã demo video" khi chưa có.

---

## 6. Chạy thử toàn bộ

```powershell
# 1. Test
.\.venv\Scripts\python.exe -m pytest

# 2. Build index
.\.venv\Scripts\python.exe src\main.py ingest

# 3. Hỏi qua CLI
.\.venv\Scripts\python.exe src\main.py ask "RAG là gì?"

# 4. API
.\.venv\Scripts\python.exe src\main.py serve --reload

# 5. UI
.\.venv\Scripts\streamlit.exe run src\chatbot\streamlit_app.py

# 6. Đánh giá
.\.venv\Scripts\python.exe src\main.py eval --top-k 4
```

---

## 7. Kiểm thử Tuần 6

| Test | Bảo vệ điều gì |
|---|---|
| `test_health_reports_index_size` | Health phản ánh đúng index |
| `test_ask_returns_answer_and_sources` | API trả answer + nguồn |
| `test_ask_rejects_empty_question` | Validation 422 |
| `test_ask_abstains_for_unknown_topic` | API không bịa |
| `test_upload_adds_document_and_can_be_queried` | Upload → hỏi được |
| `test_upload_rejects_unsupported_extension` | 415 cho `.csv` |

Điểm đáng học: test dùng `TestClient` nên **không cần chạy server thật**, nhưng
vẫn đi qua đầy đủ validation và routing.

---

## 8. Bài tập

1. Thêm endpoint `GET /sources` trả danh sách file + số chunk mỗi file.
2. Thêm tham số `min_score` cho `/ask` và kiểm tra validation.
3. Viết test upload file PDF scan, kiểm tra response có cảnh báo.
4. Thêm nút "tải index" trong Streamlit (dùng `st.download_button`).
5. Đổi `MAX_UPLOAD_BYTES` xuống 1 MB, viết test trả 413.
6. Chạy `uvicorn` và thử gọi API bằng `curl`/Postman, ghi lại response.

---

## 9. Câu hỏi tự kiểm tra

1. Vì sao dùng app factory thay vì tạo pipeline ở module?
2. `Field(min_length=1)` đã đủ chặn câu hỏi rỗng chưa? Vì sao?
3. Vì sao phải đọc `MAX_UPLOAD_BYTES + 1` byte?
4. Vì sao lưu file upload bằng tên gốc?
5. Streamlit rerun gây ra vấn đề gì nếu không dùng session state?
6. Vì sao không được log toàn bộ câu hỏi và tài liệu?
7. `TestClient` kiểm tra được gì mà gọi hàm trực tiếp không kiểm tra được?
8. Ba giới hạn bảo mật nào phải ghi rõ trước khi gọi là "production"?

---

## 10. Tóm tắt một dòng

**Tuần 6 không thêm trí tuệ mới, mà làm cho pipeline đã có được dùng đúng cách,
an toàn vừa đủ và có tài liệu để người khác tái lập.**
