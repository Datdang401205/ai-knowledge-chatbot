# Bài học Tuần 2: Làm quen LLM và prompt

> Mục tiêu: Thay luật cố định bằng mô hình ngôn ngữ (LLM). Hiểu prompt, system/user message, token, context window, cách gọi LLM local qua API, xử lý lỗi và giữ mã nguồn an toàn.

## 1. Tổng quan Tuần 2

Tuần 2 tập trung vào việc tích hợp **Ollama** (LLM chạy local) vào chatbot hiện tại. Thay vì chỉ dùng rule-based như Tuần 1, giờ đây chatbot có thể sinh câu trả lời bằng mô hình ngôn ngữ.

**Những gì đã làm:**

- Tìm hiểu khái niệm LLM, prompt, system message, user message, token, context window
- Thêm cấu hình LLM trong `config.py` (đọc từ `.env`)
- Tạo module `llm.py` để gọi Ollama API
- Tạo module `responder.py` để chuyển đổi giữa `rule-based` và `llm`
- Viết system prompt yêu cầu trả lời thận trọng, không bịa
- Xử lý timeout, lỗi kết nối, câu trả lời rỗng
- Thêm test xử lý lỗi LLM
- Giữ chế độ mặc định `rule-based` để không phá vỡ code cũ

## 2. Kiến thức cần nắm

### 2.1. LLM là gì?

**Large Language Model (LLM)** là mô hình ngôn ngữ lớn được huấn luyện trên lượng lớn văn bản để dự đoán từ tiếp theo. LLM có thể hiểu ngôn ngữ tự nhiên, trả lời câu hỏi, tóm tắt, dịch,... 

**Tại sao LLM có thể sai?**

- **Hallucination**: Mô hình tự tin bịa thông tin không có thật
- **Thiếu ngữ cảnh**: Không có đủ thông tin để trả lời chính xác
- **Kiến thức hạn chế**: Dữ liệu huấn luyện có cutoff date
- **Hiểu sai câu hỏi**: Diễn giải sai ý người dùng

### 2.2. Prompt, System/User Message

**Prompt** là câu lệnh/yêu cầu gửi đến LLM để hướng dẫn cách trả lời.

**Trong API chat**, thường chia thành các role:

| Role | Vai trò | Ví dụ |
|---|---|---|
| `system` | Hướng dẫn hành vi tổng quát của assistant (persona, quy tắc) | "Bạn là trợ lý trả lời ngắn gọn, trung thực. Nếu không biết hãy thừa nhận." |
| `user` | Câu hỏi/yêu cầu từ người dùng | "Giải thích RAG là gì?" |
| `assistant` | Câu trả lời trước đó (dùng cho hội thoại nhiều lượt) | (tự động thêm) |

**Ví dụ payload gửi tới Ollama:**

```json
{
  "model": "qwen2.5:3b",
  "messages": [
    {"role": "system", "content": "Bạn là trợ lý AI..."},
    {"role": "user", "content": "Xin chào"}
  ],
  "stream": false
}
```

### 2.3. Token và Context Window

- **Token**: Đơn vị nhỏ nhất mà LLM xử lý (từ, phần từ, dấu câu). Tiếng Việt thường tốn nhiều token hơn tiếng Anh.
- **Context window**: Số token tối đa mà mô hình có thể xử lý cùng lúc (bao gồm cả prompt và response). Nếu vượt quá, mô hình sẽ "quên" phần đầu.

**Lưu ý thực tế:**
- Model nhỏ (vd `qwen2.5:3b`) thường có context window vừa phải nhưng chạy nhanh hơn
- Cần quan tâm đến RAM khi chạy local
- Không gửi quá nhiều ngữ cảnh không cần thiết

## 3. Cấu trúc code đã thêm

### 3.1. `src/chatbot/config.py`

Thêm cấu hình LLM đọc từ biến môi trường:

```python
# Cấu hình LLM (Ollama) - dùng cho Tuần 2
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))

# System prompt yêu cầu trả lời thận trọng, không bịa
LLM_SYSTEM_PROMPT = (
    "Bạn là một trợ lý AI giúp trả lời câu hỏi một cách ngắn gọn, chính xác và trung thực.\n"
    "Quy tắc:\n"
    "- Chỉ trả lời dựa trên kiến thức chung nếu có, nhưng nếu không chắc chắn hãy thừa nhận không biết.\n"
    "- Trả lời bằng ngôn ngữ phù hợp với câu hỏi của người dùng (nếu người dùng hỏi bằng tiếng Việt, hãy trả lời bằng tiếng Việt).\n"
    "- Không bịa số liệu, không đưa thông tin không có căn cứ.\n"
    "- Nếu câu hỏi vượt quá kiến thức hoặc không có đủ thông tin để trả lời chắc chắn, hãy nói rõ: 'Tôi không chắc chắn về điều này.'\n"
)
```

**Điểm quan trọng:** Sử dụng `python-dotenv` để đọc `.env`, không hardcode URL/model. `.env` đã được thêm vào `.gitignore`.

### 3.2. `src/chatbot/llm.py`

Module này chỉ lo **gọi HTTP** và **xử lý lỗi**:

```python
class LLMError(Exception): ...
class LLMConnectionError(LLMError): ...
class LLMTimeoutError(LLMError): ...
class LLMResponseError(LLMError): ...

def ask_llm(question: str, system_prompt: str | None = None) -> str:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = _build_payload(question, system_prompt)
    try:
        response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError as exc:
        raise LLMConnectionError("Không kết nối được tới Ollama...") from exc
    except requests.exceptions.Timeout as exc:
        raise LLMTimeoutError(f"Hết thời gian chờ ({OLLAMA_TIMEOUT_SECONDS}s)...") from exc
    ...
    # Kiểm tra response, parse JSON, lấy message.content
    ...
    if not isinstance(content, str) or not content.strip():
        raise LLMResponseError("Ollama trả về câu trả lời rỗng.")
    return content.strip()
```

**Nguyên tắc:** Tách biệt hoàn toàn với UI/CLI. Module này có thể test độc lập, dễ thay thế sau này.

### 3.3. `src/chatbot/responder.py`

Quyết định dùng rule-based hay LLM theo cấu hình:

```python
def get_llm_reply(user_input: str) -> str:
    try:
        return ask_llm(user_input)
    except LLMConnectionError as exc:
        return f"{exc}\nGợi ý: chạy 'ollama serve' hoặc 'ollama run qwen2.5:3b'."
    except LLMTimeoutError as exc:
        return str(exc)
    except LLMResponseError as exc:
        return f"Lỗi phản hồi từ LLM: {exc}"
    except LLMError as exc:
        return f"Lỗi LLM: {exc}"


def get_reply(user_input: str) -> str:
    if MODE.lower() == "llm":
        return get_llm_reply(user_input)
    return get_rule_reply(user_input)
```

**Lợi ích:** Backward compatible. `main.py` không cần thay đổi nhiều, chỉ đổi import từ `chatbot.intents` sang `chatbot.responder`.

### 3.4. Cập nhật `main.py`

```python
# Trước:
from chatbot.intents import get_reply, normalize

# Sau:
from chatbot.intents import normalize
from chatbot.responder import get_reply
```

Vòng lặp CLI vẫn giữ nguyên logic cũ, chỉ thay nguồn sinh câu trả lời.

## 4. Cấu hình môi trường

### `.env.example` (đã cập nhật)

```env
# Copy file này thành .env rồi điền giá trị thật. KHÔNG commit file .env.
# Tuần 2 sẽ dùng các biến này để gọi LLM local qua Ollama.

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_TIMEOUT_SECONDS=120
```

### `.env` (tạo local nếu cần test)

```bash
# Copy từ .env.example
cp .env.example .env

# Bật chế độ LLM
CHATBOT_MODE=llm
```

**Quan trọng:** Chỉ commit `.env.example`, tuyệt đối không commit `.env`.

## 5. Thư viện đã thêm

Cập nhật `requirements.txt`:

```txt
pytest==9.1.1
requests==2.34.2
python-dotenv==1.2.4
```

Cài đặt:

```bash
pip install -r requirements.txt
```

## 6. Test xử lý lỗi LLM

Tạo `tests/test_llm_errors.py` để đảm bảo chatbot xử lý lỗi một cách thân thiện:

```python
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
    assert "chào" in reply.lower()
```

**Kỹ thuật dùng:** `monkeypatch` để giả lập lỗi mà không cần Ollama thật.

## 7. Chạy thử nghiệm

### Chế độ rule-based (mặc định)

```bash
cd C:\Users\Admin\Desktop\AI_Knowledge_Base
$env:PYTHONPATH="src"
python src/main.py
```

Vẫn hoạt động như Tuần 1.

### Chế độ LLM (khi có Ollama)

**Bước 1:** Cài Ollama (nếu chưa có) - https://ollama.com/download

**Bước 2:** Tải model nhỏ

```bash
ollama pull qwen2.5:3b
```

**Bước 3:** Chạy Ollama

```bash
ollama serve
# hoặc đơn giản ollama run qwen2.5:3b
```

**Bước 4:** Tạo `.env` và bật chế độ LLM

```bash
cp .env.example .env
# Sửa .env: CHATBOT_MODE=llm
```

**Bước 5:** Chạy chatbot

```bash
$env:PYTHONPATH="src"
python src/main.py
```

Bây giờ chatbot sẽ gọi LLM để trả lời thay vì dùng luật cố định.

### Chạy test

```bash
$env:PYTHONPATH="src"
python -m pytest -v
```

**Kết quả:** 21 tests passed

## 8. So sánh prompt - Bài tập thực hành

**Mục tiêu:** Hiểu ảnh hưởng của prompt đến câu trả lời.

### Thử nghiệm 1: Prompt đơn giản

**System prompt:**
```text
Bạn là trợ lý AI.
```

**Câu hỏi:** "Hãy giải thích RAG là gì?"

**Kết quả mong đợi:** Có thể trả lời dài dòng, thiếu cấu trúc.

### Thử nghiệm 2: Prompt chi tiết (như đã dùng)

**System prompt:** (xem `LLM_SYSTEM_PROMPT` trong config)

**Cùng câu hỏi:** "Hãy giải thích RAG là gì?"

**Quan sát:**
- Câu trả lời ngắn gọn, có cấu trúc hơn
- Có xu hướng thừa nhận nếu không chắc chắn
- Trả lời đúng ngôn ngữ với câu hỏi

### Kết luận

**Prompt ảnh hưởng lớn đến:**
- **Tone & style**: Trang trọng/thoải mái, ngắn/dài
- **Hành vi**: Có thừa nhận không biết hay tự tin bịa
- **Ngôn ngữ**: Tuân theo ngôn ngữ yêu cầu
- **Độ chính xác**: Giảm hallucination nhờ quy tắc rõ ràng

**Lưu ý quan trọng:** Prompt không đảm bảo sự thật tuyệt đối. LLM vẫn có thể sai dù prompt đã cẩn thận. Đó là lý do tại sao ở Tuần 4 chúng ta sẽ thêm **RAG** (truy xuất từ tài liệu) để buộc LLM chỉ trả lời dựa trên ngữ cảnh có bằng chứng.

## 9. Bài tập Tuần 2

### Bài tập bắt buộc

- [x] Tìm hiểu LLM, prompt, system/user message, token, context window
- [x] Cài dependencies (`requests`, `python-dotenv`)
- [x] Tạo module gọi Ollama API với xử lý lỗi
- [x] Tách cấu hình ra `.env`
- [x] Thêm chế độ LLM vào chatbot
- [x] Viết system prompt thận trọng
- [x] Thêm test xử lý lỗi
- [x] Chạy toàn bộ test pass

### Bài tập mở rộng (thực hành thêm)

1. **Thêm retry logic:** Khi timeout, thử lại 1-2 lần trước khi báo lỗi
2. **Thêm temperature:** Thử `temperature=0.3` vs `temperature=0.7` xem khác biệt
3. **Logging:** Thêm log thời gian gọi API (không log nội dung câu hỏi/câu trả lời nhạy cảm)
4. **So sánh models:** Nếu có nhiều model, so sánh chất lượng câu trả lời

### Câu hỏi tự kiểm tra

1. **Sự khác biệt giữa system message và user message là gì?**
   - System: hướng dẫn hành vi tổng quát
   - User: câu hỏi/yêu cầu cụ thể

2. **Token là gì và tại sao cần quan tâm?**
   - Token là đơn vị xử lý của LLM. Vượt quá context window sẽ mất ngữ cảnh.

3. **Tại sao cần xử lý lỗi khi gọi API local?**
   - Ollama có thể chưa chạy, timeout, response không hợp lệ. Cần thông báo thân thiện.

4. **Tại sao giữ chế độ rule-based làm mặc định?**
   - Đảm bảo backward compatible, không phụ thuộc Ollama, test vẫn chạy được.

5. **Prompt có thể ngăn LLM bịa hoàn toàn không?**
   - Không. Chỉ giảm thiểu. Cần kết hợp với RAG để tăng độ tin cậy.

## 10. Kết luận

Tuần 2 đã hoàn thành mục tiêu: chuyển từ chatbot rule-based sang có khả năng dùng LLM. Đã xây dựng kiến trúc rõ ràng (tách config/llm/responder), xử lý lỗi đầy đủ, giữ an toàn với Git, và tất cả test đều pass.

**Trạng thái:** Tích hợp hoàn tất. Cần cài Ollama để test thực tế chế độ `llm`. 

**Bước tiếp theo:** Tuần 3 - Đọc PDF, chunking, tạo embeddings để chuẩn bị cho RAG.
