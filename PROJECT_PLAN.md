# AI Knowledge Base Chatbot — Project Plan

> **Mục tiêu:** Tự xây dựng một chatbot hỏi đáp tài liệu có sử dụng LLM và RAG, đủ hoàn thiện để đưa vào CV khi ứng tuyển **AI Engineer Intern**.  
> **Thời gian dự kiến:** 6 tuần, 1–2 giờ/ngày, khoảng 5 ngày/tuần. Có thể kéo dài tùy lịch học.  
> **Nguyên tắc:** Hiểu rồi mới dùng framework; đo chất lượng thay vì chỉ làm giao diện; không tuyên bố tính năng hoặc số liệu chưa kiểm chứng.

---

## 1. Tầm nhìn sản phẩm

Người dùng tải tài liệu lên (bắt đầu với PDF), đặt câu hỏi bằng tiếng Việt hoặc tiếng Anh và nhận câu trả lời dựa trên nội dung tài liệu, kèm trích dẫn nguồn. Khi tài liệu không có thông tin phù hợp, chatbot phải nói rõ chưa tìm thấy bằng chứng thay vì tự bịa.

### Phạm vi MVP

- [x] Tải và đọc file PDF.
- [x] Chia văn bản thành các đoạn (chunking).
- [x] Tạo embeddings và lưu vector.
- [x] Tìm các đoạn liên quan đến câu hỏi (retrieval).
- [x] Gửi câu hỏi cùng ngữ cảnh tìm được tới LLM. — Có provider `ollama`; mặc định `extractive` để chạy offline.
- [x] Trả lời kèm tên file và trang/đoạn nguồn nếu có thể xác định.
- [x] Xử lý câu hỏi không có bằng chứng phù hợp.
- [x] Có giao diện đơn giản và hướng dẫn chạy local. — Streamlit + FastAPI.
- [x] Có bộ câu hỏi đánh giá và báo cáo kết quả thực nghiệm.
- [ ] Có README, sơ đồ kiến trúc, kiểm thử và ảnh/video demo. — README/sơ đồ/test có; **ảnh/video demo chưa có**.

### Không nằm trong MVP

- Huấn luyện LLM từ đầu.
- Multi-agent phức tạp.
- Hỗ trợ mọi định dạng file ngay từ đầu.
- Đăng nhập nhiều người dùng, thanh toán hoặc hệ thống production lớn.
- Khẳng định chatbot “không bao giờ hallucinate”.

---

## 2. Người dùng và ví dụ sử dụng

**Người dùng mục tiêu:** sinh viên, người nghiên cứu hoặc nhân viên cần tra cứu tài liệu dài.

Ví dụ:
1. “Tài liệu này nghiên cứu vấn đề gì?”
2. “Dữ liệu đầu vào và phương pháp là gì?”
3. “Kết quả chính được báo cáo ở đâu?”
4. “Tài liệu có đề cập đến X không?”
5. “Hãy tóm tắt phần hạn chế và dẫn nguồn.”

**Hành vi mong đợi:** câu trả lời ngắn gọn, có bằng chứng truy xuất; không trả lời chắc chắn nếu nguồn không hỗ trợ.

---

## 3. Công nghệ đề xuất

Bắt đầu tối giản; chỉ thêm thư viện khi hiểu lý do sử dụng.

| Thành phần | Lựa chọn ban đầu | Vai trò |
|---|---|---|
| Ngôn ngữ | Python 3.11 hoặc 3.12 | Logic ứng dụng |
| IDE | VS Code | Viết và debug |
| Môi trường | `venv` + `pip` | Cô lập dependencies |
| LLM local | Ollama + một mô hình phù hợp phần cứng | Sinh câu trả lời |
| Embeddings | Mô hình embedding đa ngôn ngữ chạy local | Biến văn bản thành vector |
| Vector store | Chroma hoặc FAISS (chọn một) | Lưu và tìm vector |
| PDF parsing | PyMuPDF hoặc pypdf (chọn một) | Trích xuất text |
| Backend | FastAPI | Cung cấp API |
| UI | Streamlit cho bản đầu | Giao diện demo nhanh |
| Test | pytest | Kiểm thử tự động |
| Quản lý mã | Git + GitHub | Lịch sử thay đổi và portfolio |

**Lưu ý:** Ollama không đảm bảo mọi mô hình đều chạy mượt trên mọi máy. Kiểm tra RAM, dung lượng ổ đĩa và khả năng phần cứng trước khi chọn model. Có thể dùng API có hạn mức miễn phí nếu nhà cung cấp đang hỗ trợ, nhưng cần kiểm tra điều kiện hiện hành và không đưa API key lên GitHub.

---

## 4. Kiến trúc mục tiêu

```text
User
  |
  v
Streamlit UI
  |
  v
FastAPI
  |
  +--> Document ingestion
  |      +--> PDF text extraction
  |      +--> Chunking + metadata (file, page)
  |      +--> Embedding model
  |      +--> Vector store
  |
  +--> Question answering
         +--> Embed question
         +--> Retrieve top-k chunks
         +--> Check relevance / evidence
         +--> Build grounded prompt
         +--> LLM
         +--> Answer + source citations
```

Ở giai đoạn học ban đầu, có thể viết phiên bản CLI trước UI/API để hiểu luồng dữ liệu. Không cần triển khai toàn bộ kiến trúc ngay trong ngày đầu.

---

## 5. Cấu trúc thư mục mục tiêu

```text
ai-knowledge-chatbot/
├── data/
│   ├── sample_docs/          # Tài liệu mẫu được phép chia sẻ
│   └── eval_questions.json   # Bộ câu hỏi đánh giá
├── src/
│   └── chatbot/
│       ├── __init__.py
│       ├── config.py
│       ├── ingest.py         # Đọc tài liệu và tạo chunks
│       ├── chunking.py       # Chia đoạn
│       ├── embeddings.py     # Tạo embeddings
│       ├── vector_store.py   # Lưu/tìm vector
│       ├── retrieval.py      # Truy xuất top-k
│       ├── generation.py     # Gọi LLM, prompt
│       ├── rag_pipeline.py   # Ghép các bước RAG
│       └── main.py           # Điểm chạy CLI/API
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_chunking.py
│   ├── test_ingest.py
│   ├── test_retrieval.py
│   └── test_rag_pipeline.py
├── reports/
│   ├── evaluation.md
│   └── experiments.md
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── PROJECT_PLAN.md
```

Có thể giữ cấu trúc đơn giản hơn trong tuần đầu. Không tạo file rỗng chỉ để đủ cây thư mục; mỗi module được thêm khi bắt đầu dùng.

---

## 6. Lộ trình 6 tuần

### Tuần 1 — Python project và chatbot CLI

**Mục tiêu:** hiểu cấu trúc project và viết được chương trình chạy được.

- [x] Tạo thư mục project và môi trường ảo `.venv`. — `.venv/` đã tạo, đã thêm vào `.gitignore`.
- [x] Kiểm tra phiên bản Python và pip. — Python 3.12.7, pip 26.2 (nâng cấp trong venv).
- [x] Ôn hàm, chuỗi, điều kiện, vòng lặp, `return`, import, exception. — Ôn lại trong README mục 9 và áp dụng ngay trong code Tuần 1.
- [x] Viết chatbot rule-based CLI phiên bản 0.1. — `src/chatbot/intents.py` + `src/chatbot/main.py` + launcher `src/main.py`.
- [x] Tách logic trả lời khỏi hàm `main()`. — `main()` chỉ lo I/O; `get_reply()`/`find_intent()` là logic thuần, không in/ghi.
- [x] Thêm xử lý input rỗng và lệnh thoát. — `exit/quit/q/thoát`, input rỗng, `EOFError`/`KeyboardInterrupt`.
- [x] Khởi tạo Git repository và commit đầu tiên. — commit `2b6f5c8` "feat(week-1): rule-based chatbot CLI with tests, README and project setup".
- [x] Viết README sơ bộ: mục tiêu, cài đặt, cách chạy. — `README.md` (12 mục, có cả phần bài tập Tuần 1).

**Bài tập kiểm tra:**
- [ ] Tự thêm một intent mới (`feedback`) — đã ghi hướng dẫn từng bước trong README mục 9.
- [ ] Giải thích `if __name__ == "__main__":` — đã ghi trong README mục 9.
- [ ] Chạy chương trình mà không cần chép lại từng bước — đã ghi sẵn chuỗi lệnh trong README mục 3 và 4.

**Definition of Done:** ✅ Đạt — chạy được trên máy local, code dễ đọc, có Git commit và README sơ bộ.

**Tài liệu học:** `docs/BAI_HOC_TUAN_1.md` — giải thích từng bước Tuần 1, kiến thức Python nền, 2 bug thật đã gặp, đáp án bài tập và 12 câu tự kiểm tra.

### Ghi chú Tuần 1 (quyết định đã chốt)

| Vấn đề | Quyết định | Lý do |
|---|---|---|
| Cấu trúc thư mục | Giữ `src/chatbot/` nhưng chưa tạo file rỗng cho các module của Tuần 2–6 | Đúng nguyên tắc "mỗi module thêm khi bắt đầu dùng" |
| Cách chạy | Thêm launcher `src/main.py` (tự chèn `src` vào `sys.path`) | Vừa khớp lệnh `python src/main.py` trong plan, vừa giữ được package `chatbot` |
| So khớp từ khóa | Khớp theo *cụm từ liên tiếp* trên token, không dùng `keyword in text` | `"hi" in "khi nào"` là `True` → khớp sai intent |
| Tiếng Việt trên Windows | `configure_console_encoding()` ép UTF-8 cho stdin/stdout/stderr | Console mặc định `cp1252` gây `UnicodeEncodeError` |
| Test CLI | `run_chatbot` nhận `input_func`/`output` qua tham số | Test được không cần gõ tay (dependency injection) |
| Test tự động | `pyproject.toml` khai báo `pythonpath = ["src"]` | `pytest` tự tìm package `chatbot` mà không cần cài đặt |
| Kết quả chạy | 30 test passed (`pytest`) | Xác minh bằng lệnh thật, không ước lượng |

### Tuần 2 — Làm quen LLM và prompt

**Mục tiêu:** thay luật cố định bằng mô hình ngôn ngữ.

- [x] Tìm hiểu LLM, prompt, system message, user message, token, context window. — Đã ghi trong docs/BAI_HOC_TUAN_1.md và áp dụng qua code.
- [ ] Cài và chạy Ollama nếu phần cứng đáp ứng; chọn một model nhỏ phù hợp. — Chưa cài Ollama trên máy này (ollama không tồn tại). Sẵn sàng tích hợp khi có.
- [x] Gọi mô hình từ Python qua API local. — `src/chatbot/llm.py` gọi Ollama `/api/chat` qua `requests`.
- [x] Tách cấu hình model và URL khỏi logic chính. — `src/chatbot/config.py` đọc từ `.env` (OLLAMA_BASE_URL/MODEL/TIMEOUT), `.env.example` đã cập nhật.
- [x] Xử lý timeout, lỗi kết nối và câu trả lời rỗng. — `LLMConnectionError`/`LLMTimeoutError`/`LLMResponseError`, thông báo thân thiện trong `responder.py`.
- [x] Viết prompt yêu cầu trả lời bằng ngôn ngữ phù hợp và thừa nhận khi không biết. — `LLM_SYSTEM_PROMPT` trong config, yêu cầu trả lời đúng ngôn ngữ và nói rõ nếu không chắc chắn.
- [x] Không lưu secret hoặc thông tin riêng tư trong Git. — `.env` đã có trong `.gitignore`, chỉ có `.env.example`.

**Bài tập kiểm tra:** so sánh cùng một câu hỏi với 2 prompt; ghi lại khác biệt và giải thích vì sao prompt không đảm bảo sự thật tuyệt đối.

**Definition of Done:** chatbot nhận input, gọi được LLM và xử lý lỗi cơ bản.

### Tuần 3 — PDF, chunking và embeddings

**Mục tiêu:** chuẩn bị dữ liệu để truy xuất.

- [x] Chọn 3–5 tài liệu PDF phù hợp và được phép sử dụng. — 3 tài liệu tự sinh bằng `scripts/make_sample_docs.py` + 1 PDF scan để test lỗi.
- [x] Trích xuất text theo trang; ghi nhận file và số trang. — `ingest.py` dùng PyMuPDF, `PageText(source, page, text)`.
- [x] Kiểm tra PDF scan không có text và xử lý bằng thông báo phù hợp. — `scan_khong_text.pdf` sinh cảnh báo, không làm hỏng index.
- [x] Viết hàm chunking có kích thước và overlap cấu hình được. — `chunking.py`, cắt ưu tiên ranh giới đoạn/câu, ID sha1 ổn định.
- [x] Kiểm tra chunk rỗng, chunk quá ngắn và mất metadata. — Có `min_chars`, bỏ trang rỗng, giữ `source/page/chunk_index/start_char/end_char`.
- [x] Tạo embeddings bằng mô hình hỗ trợ ngôn ngữ của tài liệu. — Mặc định `HashingEmbedder` offline; tùy chọn `OllamaEmbedder` (`nomic-embed-text`).
- [x] Thử tính tương đồng giữa các đoạn có liên quan và không liên quan. — Test trong `tests/test_embeddings.py` và `tests/test_retrieval.py`.

**Bài tập kiểm tra:** in ra 5 chunks, kiểm tra trang nguồn; thay đổi chunk size/overlap và ghi nhận tác động.

**Definition of Done:** pipeline đọc PDF tạo được chunks có metadata và embeddings tái lập được.

**Tài liệu học:** `docs/BAI_HOC_TUAN_3.md` — PyMuPDF, lỗi PDF scan/ký tự ẩn, chunk size/overlap, hashing embedding, chuẩn hoá L2.

### Tuần 4 — Retrieval và RAG end-to-end

**Mục tiêu:** chatbot trả lời dựa trên tài liệu, có nguồn.

- [x] Chọn một vector store: Chroma hoặc FAISS. — Chọn **tự viết** `VectorStore` (numpy + JSON) thay vì Chroma/FAISS để hiểu cơ chế; interface `add/search/save/load` đủ để thay backend sau.
- [x] Lưu embeddings cùng metadata. — `data/index/store.json` + `manifest.json` (provider, dimension, cấu hình retrieval).
- [x] Với mỗi câu hỏi, truy xuất top-k chunks. — `retrieval.py`: vector cosine + lexical coverage + IDF (hybrid).
- [x] Hiển thị nội dung truy xuất để debug trước khi gọi LLM. — `python src/main.py ask --retrieve-only "..."`.
- [x] Viết prompt chứa câu hỏi và ngữ cảnh truy xuất. — `build_rag_prompt`, `RAG_SYSTEM_PROMPT`.
- [x] Yêu cầu câu trả lời dựa trên ngữ cảnh, không tự thêm dữ kiện. — Prompt ràng buộc; generator `extractive` chỉ trích câu từ ngữ cảnh.
- [x] Trả về nguồn file/trang tương ứng. — `Source(file, page, score, chunk_id)`, chỉ trích dẫn nguồn thực sự dùng.
- [x] Xử lý trường hợp retrieval không đủ liên quan. — Ngưỡng `MIN_RELEVANCE_SCORE`, `NO_EVIDENCE_MESSAGE`, kiểm tra thực thể lạ.
- [x] Tạo ít nhất 20 câu hỏi đánh giá ban đầu. — `data/eval_questions.json` có 40 câu (36 có đáp án + 4 ngoài tài liệu).

**Bài tập kiểm tra:** hỏi một câu có câu trả lời trong tài liệu, một câu ngoài tài liệu và một câu mơ hồ; xem nguồn truy xuất có phù hợp không.

**Definition of Done:** chạy được toàn bộ luồng PDF → chunks → embeddings → retrieval → LLM → answer + sources.

**Tài liệu học:** `docs/BAI_HOC_TUAN_4.md` — cosine, hybrid retrieval, IDF, grounding, kiểm tra thực thể lạ, generator extractive.

### Tuần 5 — Đánh giá và cải thiện

**Mục tiêu:** chứng minh chất lượng bằng dữ liệu thay vì cảm giác.

- [x] Tạo bộ đánh giá khoảng 30–50 câu hỏi, có câu trả lời tham chiếu và nguồn mong đợi. — 40 câu: factual, reasoning, paraphrase (Anh), unanswerable.
- [x] Đo Hit Rate@k hoặc Recall@k cho retrieval. — Có Hit Rate@k, Recall@k, MRR trong `evaluation.py`.
- [x] Kiểm tra answer correctness bằng tiêu chí rõ ràng; có thể chấm thủ công nếu dữ liệu nhỏ. — Tự động theo keyword coverage ≥ 0.5; báo cáo ghi rõ đây là thước đo thô.
- [x] Đánh giá source/grounding: câu trả lời có được hỗ trợ bởi chunks không? — Đo `citation_rate`; trích dẫn chỉ lấy nguồn thực sự dùng.
- [x] Thử các chunk size và top-k khác nhau. — `scripts/run_experiments.py`: 400/700/1000 × top-k 2/4/6.
- [x] Ghi lại baseline, thay đổi, kết quả và giới hạn. — `reports/evaluation.md`, `reports/experiments.md`, `reports/*.json`.
- [x] Thêm unit tests cho chunking và retrieval. — Thêm test cho toàn bộ pipeline, API, evaluation; tổng 84 test.
- [x] Không chọn cấu hình chỉ vì một vài ví dụ tốt. — Cấu hình mặc định giữ 700/120/top-4; số liệu thực nghiệm ghi riêng, không tuyên bố vượt trội.

**Definition of Done:** có báo cáo đánh giá, giải thích được cách đo, kết quả và các trường hợp thất bại. Không tự đặt số liệu trước khi chạy.

**Tài liệu học:** `docs/BAI_HOC_TUAN_5.md` — thiết kế bộ eval, Hit Rate/Recall/MRR, answer accuracy, abstention, đọc thực nghiệm.

### Tuần 6 — API, giao diện và portfolio

**Mục tiêu:** đóng gói thành sản phẩm demo được.

- [x] Tạo API FastAPI cho health check, ingest tài liệu và hỏi đáp. — `chatbot/api.py`: `/`, `/health`, `/stats`, `/ask`, `/documents`.
- [x] Tạo giao diện Streamlit tải PDF, nhập câu hỏi và xem nguồn. — `chatbot/streamlit_app.py`, có build index, upload, chat, expander nguồn.
- [x] Thêm trạng thái đang xử lý và thông báo lỗi dễ hiểu. — `st.spinner`, `st.error`, HTTP 415/422/413 với `detail` tiếng Việt.
- [x] Kiểm tra file không hợp lệ và giới hạn kích thước file. — Chỉ `.pdf/.txt/.md`, tối đa `MAX_UPLOAD_BYTES` (15 MB).
- [x] Thêm logging không ghi nội dung nhạy cảm không cần thiết. — Log lỗi ingest/LLM, không log nội dung câu hỏi/tài liệu.
- [x] Viết tests cho luồng chính và các lỗi phổ biến. — `tests/test_api.py`, `test_rag_pipeline.py`, `test_evaluation.py`.
- [x] Cập nhật `.gitignore`, `.env.example`, `requirements.txt`. — Đã thêm `data/index/`, `reports/` (trừ file mẫu), biến môi trường mới.
- [x] Viết README hoàn chỉnh và sơ đồ kiến trúc. — README 12 mục, sơ đồ luồng dữ liệu + bảng module.
- [ ] Chụp ảnh/gif hoặc quay video demo 1–2 phút. — **Chưa làm**; cần chạy UI thật và ghi màn hình.
- [ ] Triển khai demo nếu có phương án miễn phí phù hợp; nếu không, hướng dẫn chạy local đầy đủ. — Chưa deploy; README có hướng dẫn chạy local.
- [ ] Viết phần mô tả project trong CV dựa trên tính năng và số liệu đã xác minh. — Chưa viết; số liệu đã có trong `reports/`.

**Definition of Done:** người khác có thể làm theo README để chạy project; có demo, tests, kết quả đánh giá và giới hạn được ghi rõ.

**Tài liệu học:** `docs/BAI_HOC_TUAN_6.md` — FastAPI app factory, validation/upload an toàn, Streamlit session state, đóng gói demo.

---

## 7. Lịch học mỗi ngày (1–2 giờ)

Mẫu một buổi học:

1. **15–20 phút:** học một khái niệm và ghi lại bằng lời của mình.
2. **40–60 phút:** tự viết code cho một nhiệm vụ nhỏ.
3. **10–20 phút:** chạy test, đọc lỗi và sửa.
4. **5 phút:** cập nhật checklist, ghi điều đã hiểu và câu hỏi còn vướng.

Nếu chỉ có 30 phút, hãy hoàn thành một nhiệm vụ nhỏ thay vì cố làm cả module.

---

## 8. Nguyên tắc làm việc với AI coding assistant

Có thể dùng AI để hỗ trợ, nhưng cần tránh để AI viết toàn bộ project mà bạn không hiểu.

- [ ] Trước khi yêu cầu code, mô tả mục tiêu và đầu vào/đầu ra.
- [ ] Yêu cầu giải thích lựa chọn thiết kế và thư viện.
- [ ] Đọc code, tự dự đoán kết quả rồi mới chạy.
- [ ] Yêu cầu test cho hành vi bình thường và trường hợp lỗi.
- [ ] Không chấp nhận code có API key hoặc thông tin bí mật.
- [ ] Không đưa tài liệu riêng tư lên dịch vụ bên ngoài nếu chưa được phép.
- [ ] Mỗi tính năng nên có một Git commit nhỏ, có thông điệp rõ.
- [ ] Nếu không giải thích được một đoạn code, hãy thu nhỏ vấn đề và học lại trước khi thêm tính năng.

---

## 9. Bộ câu hỏi phỏng vấn cần trả lời được

### Python và backend
- [ ] Khác nhau giữa list, dict, tuple và set?
- [ ] Hàm trả về giá trị bằng `return` khác gì `print()`?
- [ ] Vì sao dùng virtual environment?
- [ ] API, HTTP request và JSON là gì?
- [ ] Xử lý exception và logging để làm gì?

### LLM và RAG
- [ ] LLM là gì? Vì sao LLM có thể tạo thông tin sai?
- [ ] Token và context window là gì?
- [ ] Embedding là gì? Vì sao cần vector?
- [ ] Chunking là gì? Overlap có tác dụng gì?
- [ ] Retrieval khác generation như thế nào?
- [ ] Vì sao RAG có thể trả lời sai dù tài liệu có câu trả lời?
- [ ] Top-k ảnh hưởng thế nào đến chất lượng và độ trễ?
- [ ] Làm sao trích dẫn đúng trang nguồn?
- [ ] Khi nào cần fine-tuning, khi nào RAG phù hợp hơn?

### Đánh giá và bảo mật
- [ ] Hit Rate@k/Recall@k đo điều gì?
- [ ] Làm sao xây dựng tập đánh giá ít thiên lệch?
- [ ] Làm sao kiểm tra câu trả lời có được nguồn hỗ trợ?
- [ ] Prompt injection trong tài liệu là gì?
- [ ] Vì sao không nên cho LLM thực thi code hoặc SQL tùy ý?
- [ ] Làm sao xử lý file độc hại, dữ liệu riêng tư và API key?

---

## 10. Checklist phát hành CV

- [ ] Project chạy được theo README trên môi trường sạch.
- [ ] Có ảnh/video demo và ví dụ câu hỏi.
- [ ] Có sơ đồ kiến trúc và mô tả các quyết định kỹ thuật.
- [ ] Có bộ câu hỏi đánh giá và báo cáo kết quả thực tế.
- [ ] Có unit tests và kết quả chạy tests.
- [ ] Nêu rõ hạn chế: PDF scan, bảng phức tạp, tài liệu dài, câu hỏi mơ hồ hoặc truy xuất sai.
- [ ] Không tuyên bố độ chính xác, tốc độ hoặc mức tiết kiệm nếu chưa đo.
- [ ] GitHub không chứa `.env`, API keys, dữ liệu riêng tư hoặc file lớn không cần thiết.
- [ ] Có thể trình bày project trong 3–5 phút và trả lời câu hỏi kỹ thuật.

### Mẫu mô tả CV sau khi hoàn thành

> **AI Knowledge Base Chatbot | Python, FastAPI, RAG, Vector Search**
>
> - Built a document question-answering chatbot using RAG to answer questions from PDF files with source citations.
> - Implemented document ingestion, chunking, embedding generation, semantic retrieval, and grounded answer generation.
> - Evaluated retrieval and answer quality on a curated question set; documented failure cases and experiments.
> - Added API endpoints, automated tests, and reproducible setup instructions.

Chỉ giữ các gạch đầu dòng phản ánh đúng những gì đã làm. Bổ sung số liệu sau khi chạy thực nghiệm.

---

## 11. Nhật ký tiến độ

Cập nhật sau mỗi buổi:

| Ngày | Việc đã làm | Kết quả/test | Vướng mắc | Việc tiếp theo |
|---|---|---|---|---|
| YYYY-MM-DD | | | | |
| 2026-10-22 | Tuần 1: tạo `.venv`, nâng pip, cài pytest, dựng package `src/chatbot` (config / intents / main), launcher `src/main.py`, 30 test, `.gitignore`, `.gitattributes`, `requirements.txt`, `pyproject.toml`, `.env.example`, README 12 mục, git init + commit đầu tiên | `pytest` → 30 passed; chạy tay `python src/main.py` xử lý đúng 5 intent + fallback + lệnh thoát; commit `2b6f5c8` | Console Windows dùng `cp1252` in tiếng Việt lỗi → đã ép UTF-8 bằng `configure_console_encoding()`; khớp từ khóa dạng substring gây nhầm "hi" trong "khi nào" → chuyển sang khớp cụm từ theo token | Làm bài tập Tuần 1 (thêm intent `feedback`, giải thích `if __name__ == "__main__":`), sau đó bắt đầu Tuần 2: cài Ollama + gọi LLM từ Python |
| 2026-10-23 | Tuần 2: thêm tích hợp Ollama (`src/chatbot/llm.py`), `responder` phân tách rule-based/llm, cập nhật config/system prompt, xử lý lỗi timeout/kết nối/rỗng, thêm test lỗi LLM, cập nhật requirements/.env.example | `pytest` → 21 passed (toàn bộ test suite); imports OK; chế độ mặc định vẫn rule-based | Ollama chưa cài trên máy; đã thêm xử lý lỗi thân thiện và hướng dẫn khi không kết nối được | Cài Ollama + tải model nhỏ (vd qwen2.5:3b) nếu muốn test chế độ llm; hoặc chuyển sang Tuần 3 (PDF/chunking) theo kế hoạch |
| 2026-10-23 | Tuần 3–6: thêm `chunking`, `ingest` (PyMuPDF, cảnh báo PDF scan), `embeddings` (hashing offline + Ollama tùy chọn), `vector_store` (numpy + JSON), `retrieval` (hybrid cosine + lexical + IDF, chặn thực thể lạ), `generation` (extractive + Ollama), `rag_pipeline`, `evaluation`, `cli` (chat/ingest/ask/eval/serve), `api` FastAPI, `streamlit_app`; sinh 3 PDF mẫu + 1 PDF scan; tạo 40 câu hỏi đánh giá; viết test cho toàn bộ module | `pytest` → **84 passed**; `eval` top_k=4 → Hit Rate **86.11%**, Recall **86.11%**, MRR **0.806**, answer accuracy **72.5%**, abstention **100%**, latency TB **2.79 ms**; thực nghiệm 400/700/1000 × top-k 2/4/6 → MRR tốt nhất 0.833 ở chunk 400 | Generator mặc định là extractive nên câu trả lời còn cụt/ghép nhiều câu; câu hỏi paraphrase tiếng Anh bị hụt vì embedding hashing không hiểu ngữ nghĩa; Ollama chưa cài nên chưa đo chất lượng LLM thật | Bổ sung ảnh/video demo, viết mô tả CV, và nếu có Ollama thì chạy lại `eval` với `LLM_PROVIDER=ollama` |

### Ghi chú quyết định Tuần 3–6

| Vấn đề | Quyết định | Lý do |
|---|---|---|
| Vector store | Tự viết bằng numpy + JSON thay vì Chroma/FAISS | Dữ liệu nhỏ, cần thấy rõ cosine similarity và cấu trúc lưu; interface đủ để thay backend sau |
| Embedding mặc định | `HashingEmbedder` offline, 1024 chiều | Chạy không cần model, tất định, test ổn định; chấp nhận không hiểu đồng nghĩa |
| Embedding tùy chọn | `OllamaEmbedder` (`nomic-embed-text`) | Có đường nâng cấp ngữ nghĩa khi máy đủ tài nguyên |
| Generator mặc định | `extractive` | Không cần LLM, dùng để test retrieval/grounding và làm baseline |
| Generator tùy chọn | `OllamaGenerator` + `RAG_SYSTEM_PROMPT` | Trả lời tự nhiên hơn nhưng cần Ollama và kiểm tra hallucination |
| Retrieval | Hybrid: 0.5 cosine + 0.5 lexical coverage có IDF | Hashing embedding yếu về ngữ nghĩa; IDF giúp thuật ngữ hiếm ("OTD") thắng từ phổ biến |
| Chống bịa | Ngưỡng `MIN_RELEVANCE_SCORE=0.25` + kiểm tra thực thể lạ | Từ chối câu hỏi ngoài tài liệu rõ ràng hơn chỉ dựa vào cosine |
| Chunk mặc định | 700 ký tự, overlap 120 | Cân bằng ngữ cảnh và nhiễu; số liệu thực nghiệm không đủ để khẳng định cấu hình khác tốt hơn |

### Trạng thái hiện tại

- [x] Tuần 1 — Python project và chatbot CLI *(hoàn thành 2026-10-22)*
- [x] Tuần 2 — LLM và prompt *(tích hợp xong; chưa test với Ollama thật)*
- [x] Tuần 3 — PDF, chunking và embeddings *(hoàn thành 2026-10-23)*
- [x] Tuần 4 — Retrieval và RAG *(hoàn thành 2026-10-23)*
- [x] Tuần 5 — Đánh giá và cải thiện *(hoàn thành 2026-10-23; báo cáo trong `reports/`)*
- [x] Tuần 6 — API, UI và portfolio *(API/UI/README xong; còn ảnh/video demo và deploy)*

**Bước tiếp theo:** chạy `streamlit run src/chatbot/streamlit_app.py` và `uvicorn chatbot.api:app --reload` để kiểm tra giao diện; chụp ảnh/quay video demo; nếu cài Ollama, chạy lại `eval` với `LLM_PROVIDER=ollama` để so sánh với baseline extractive.
