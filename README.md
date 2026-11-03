# AI Knowledge Base Chatbot

Chatbot hỏi đáp tài liệu dựa trên RAG (Retrieval-Augmented Generation).
Dự án xây dựng theo lộ trình 6 tuần trong [`PROJECT_PLAN.md`](PROJECT_PLAN.md),
ưu tiên **hiểu cơ chế**, **đo chất lượng** và **không tuyên bố quá mức**.

> **Trạng thái: v0.6.0 — hoàn thành Tuần 1–6 ở mức demo local.**
> Có PDF ingestion, chunking, embedding, vector store, retrieval hybrid,
> trích dẫn nguồn, CLI, FastAPI, Streamlit, bộ 40 câu đánh giá và báo cáo thực nghiệm.
> Chưa có ảnh/video demo, chưa deploy, chưa đo chất lượng với Ollama thật.

---

## 1. Mục tiêu sản phẩm

Người dùng tải tài liệu PDF/TXT/MD lên, đặt câu hỏi bằng tiếng Việt hoặc tiếng Anh
và nhận câu trả lời **dựa trên nội dung tài liệu**, kèm **trích dẫn file + trang**.
Khi không có bằng chứng phù hợp, chatbot nói rõ *chưa tìm thấy bằng chứng*
thay vì tự bịa.

---

## 2. Yêu cầu môi trường

| Công cụ | Phiên bản | Ghi chú |
|---|---|---|
| Python | 3.11 hoặc 3.12 | Đã kiểm tra với 3.12.7 |
| pip | kèm Python | Kiểm tra `python -m pip --version` |
| Git | bất kỳ | Lưu lịch sử, làm portfolio |
| Ollama | tùy chọn | Chỉ cần khi muốn embedding/LLM ngữ nghĩa |

Không cần Ollama để chạy mặc định: project dùng `HashingEmbedder` + generator
`extractive`, hoàn toàn offline.

---

## 3. Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn script kích hoạt:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Kiểm tra đúng môi trường: `python -c "import sys; print(sys.prefix)"` phải trỏ vào
`...AI_Knowledge_Base\.venv`.

---

## 4. Cách chạy

### 4.1 Tạo index từ tài liệu mẫu

```powershell
.\.venv\Scripts\python.exe src\main.py ingest
```

Kết quả mong đợi: đọc 4 file (3 có text), tạo 12 chunk, cảnh báo
`scan_khong_text.pdf` không có text. Index nằm ở `data/index/store.json`.

### 4.2 Hỏi một câu

```powershell
.\.venv\Scripts\python.exe src\main.py ask "Top-k ảnh hưởng gì tới chất lượng trả lời?"
```

Thêm `--retrieve-only` để xem chunk trước khi sinh câu trả lời:

```powershell
.\.venv\Scripts\python.exe src\main.py ask --retrieve-only "OTD được tính thế nào?"
```

### 4.3 Chat trong terminal

```powershell
.\.venv\Scripts\python.exe src\main.py chat          # chat rule-based Tuần 1–2
.\.venv\Scripts\python.exe src\main.py chat --rag    # chat dựa trên index
```

Lệnh thoát: `exit`, `quit`, `q`, `thoát`.

### 4.4 API FastAPI

```powershell
.\.venv\Scripts\python.exe src\main.py serve --reload
```

Mở http://127.0.0.1:8000/docs. Các endpoint:

| Method | Path | Chức năng |
|---|---|---|
| GET | `/health` | Trạng thái + số chunk |
| GET | `/stats` | Thống kê index |
| POST | `/ask` | `{"question": "...", "top_k": 4}` |
| POST | `/documents` | Upload `.pdf/.txt/.md` (tối đa 15 MB) |

### 4.5 Giao diện Streamlit

```powershell
.\.venv\Scripts\streamlit.exe run src\chatbot\streamlit_app.py
```

Giao diện có build index, upload tài liệu, chat và xem nguồn.

### 4.6 Đánh giá và thực nghiệm

```powershell
.\.venv\Scripts\python.exe src\main.py eval --top-k 4
.\.venv\Scripts\python.exe scripts\run_experiments.py
```

Báo cáo ghi vào `reports/evaluation.md` và `reports/experiments.md`.

### Về tiếng Việt trên Windows

Code tự ép console về UTF-8 trong `configure_console_encoding()`. Nếu vẫn lỗi
khi paste từ file, chạy `chcp 65001`.

---

## 5. Kiến trúc và cấu trúc thư mục

```text
PDF/TXT/MD
   │  ingest.py (PyMuPDF, chuẩn hoá text, cảnh báo PDF scan)
   ▼
PageText ──► chunking.py (700 ký tự, overlap 120, ID ổn định)
   ▼
embeddings.py (hashing offline hoặc Ollama)
   ▼
vector_store.py (numpy + JSON, cosine)
   ▼
retrieval.py (0.5 cosine + 0.5 lexical IDF, ngưỡng liên quan)
   ▼
generation.py (extractive mặc định, Ollama tùy chọn)
   ▼
Answer + Source(file, page, score, chunk_id)
```

```text
AI_Knowledge_Base/
├── data/
│   ├── sample_docs/            # 3 PDF mẫu + 1 PDF scan để test lỗi
│   ├── eval_questions.json     # 40 câu đánh giá
│   └── index/                  # store.json + manifest.json (sinh tự động)
├── docs/
│   ├── BAI_HOC_TUAN_1.md ... BAI_HOC_TUAN_6.md
├── reports/
│   ├── evaluation.md / evaluation_results.json
│   └── experiments.md / experiments_results.json
├── scripts/
│   ├── make_sample_docs.py
│   └── run_experiments.py
├── src/
│   ├── main.py                 # launcher
│   └── chatbot/
│       ├── config.py           # hằng số + biến môi trường
│       ├── ingest.py           # đọc PDF/TXT/MD
│       ├── chunking.py         # chia đoạn + metadata
│       ├── embeddings.py       # hashing / Ollama
│       ├── vector_store.py     # lưu và tìm vector
│       ├── retrieval.py        # truy xuất hybrid
│       ├── generation.py       # extractive / Ollama + trích dẫn
│       ├── rag_pipeline.py     # ghép toàn bộ luồng
│       ├── evaluation.py       # Hit Rate, Recall, MRR, abstention
│       ├── cli.py              # chat/ingest/ask/eval/serve
│       ├── api.py              # FastAPI
│       ├── streamlit_app.py    # giao diện web
│       ├── llm.py              # client Ollama (Tuần 2)
│       ├── responder.py        # rule-based / LLM
│       ├── intents.py          # intent Tuần 1
│       └── main.py             # vòng lặp chat
├── tests/                      # 84 test
├── .env.example
├── pyproject.toml
├── requirements.txt
├── README.md
└── PROJECT_PLAN.md
```

---

## 6. Nguyên tắc hoạt động

1. **Ingest theo trang** để giữ `source + page`; PDF scan bị cảnh báo, không im lặng bỏ qua.
2. **Chunk theo ranh giới câu/đoạn**, có overlap để không cắt đôi ý.
3. **Embedding offline** bằng hashing trick: từ + char 3-gram + bigram, chuẩn hoá L2.
   Cách này tất định và không cần model, nhưng **không hiểu đồng nghĩa**.
4. **Retrieval hybrid**: cosine bắt tương đồng bề mặt, lexical coverage có IDF giúp
   thuật ngữ hiếm (`OTD`, `FastAPI`) thắng từ phổ biến (`tính`, `hệ thống`).
5. **Chống bịa**: ngưỡng `MIN_RELEVANCE_SCORE`, kiểm tra tên riêng/viết tắt lạ,
   và generator chỉ được dùng ngữ cảnh đã truy xuất.
6. **Trích dẫn** chỉ lấy nguồn thực sự được dùng để tạo câu trả lời.

Đổi backend bằng biến môi trường trong `.env`:

```dotenv
EMBEDDING_PROVIDER=hashing   # hoặc ollama
LLM_PROVIDER=extractive      # hoặc ollama
```

---

## 7. Kết quả đã đo

Lần chạy `eval --top-k 4`, chunk 700/overlap 120, `HashingEmbedder`,
generator `extractive`, 40 câu hỏi:

| Thước đo | Kết quả |
|---|---|
| Hit Rate@4 | 86.1% |
| Recall@4 | 86.1% |
| MRR | 0.806 |
| Answer accuracy (keyword, tự động) | 72.5% |
| Citation rate | 86.1% |
| Abstention accuracy (4 câu ngoài tài liệu) | 100% |
| Độ trễ trung bình | 2.79 ms |

Thực nghiệm chunk size × top-k (`reports/experiments.md`): MRR cao nhất 0.833 ở
chunk 400; top-k không đổi Hit Rate trên bộ tài liệu nhỏ vì đáp án đúng đã ở hạng 1.

**Cách đọc số liệu:** đây là baseline extractive trên 3 tài liệu ngắn do chính
project sinh ra. Không dùng các con số này để suy rộng cho tài liệu dài, tiếng Anh
hoặc miền tri thức khác.

---

## 8. Chạy test

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Kết quả hiện tại: **84 passed**.

| Tệp | Tầng | Kiểm tra gì |
|---|---|---|
| `test_intents.py` | unit | Chuẩn hoá, intent, fallback |
| `test_cli.py` | unit + giả lập terminal | Vòng lặp chat, lệnh thoát |
| `test_cli_entrypoint.py` | integration | Chạy thật `src/main.py` |
| `test_llm_errors.py` | unit | Lỗi timeout/kết nối/rỗng của Ollama |
| `test_chunking.py` | unit | Chunk, overlap, ID ổn định, metadata |
| `test_ingest.py` | unit | Đọc TXT/MD, chuẩn hoá PDF, định dạng lỗi |
| `test_embeddings.py` | unit | Vector tất định, chuẩn hoá, tương đồng |
| `test_vector_store.py` | unit | Add/search/save/load, dedupe, ngưỡng |
| `test_retrieval.py` | unit | Hybrid ranking, từ chối thực thể lạ |
| `test_rag_pipeline.py` | integration | Build index, ask, persist/reload, upload |
| `test_evaluation.py` | unit | Hit Rate, abstention, JSON lỗi |
| `test_api.py` | integration | `/health`, `/ask`, upload, lỗi 415 |

Kỹ thuật đáng học: `RAGPipeline` là điểm vào duy nhất nên CLI, API và UI dùng
chung logic; test không cần chạy server thật.

---

## 9. Bài học theo tuần

| Tuần | Tài liệu | Nội dung chính |
|---|---|---|
| 1 | [`docs/BAI_HOC_TUAN_1.md`](docs/BAI_HOC_TUAN_1.md) | Python, CLI, test, Git |
| 2 | [`docs/BAI_HOC_TUAN_2.md`](docs/BAI_HOC_TUAN_2.md) | LLM, prompt, Ollama, lỗi mạng |
| 3 | [`docs/BAI_HOC_TUAN_3.md`](docs/BAI_HOC_TUAN_3.md) | PDF, chunking, embedding |
| 4 | [`docs/BAI_HOC_TUAN_4.md`](docs/BAI_HOC_TUAN_4.md) | Vector store, retrieval, grounding |
| 5 | [`docs/BAI_HOC_TUAN_5.md`](docs/BAI_HOC_TUAN_5.md) | Hit Rate, Recall, MRR, thực nghiệm |
| 6 | [`docs/BAI_HOC_TUAN_6.md`](docs/BAI_HOC_TUAN_6.md) | FastAPI, Streamlit, đóng gói demo |

---

## 10. Giới hạn hiện tại

- **PDF scan** không có text: chỉ cảnh báo, chưa OCR.
- **Bảng biểu phức tạp** có thể bị mất cấu trúc khi trích text.
- **Hashing embedding** không hiểu đồng nghĩa/ngoại ngữ; 6 câu paraphrase tiếng Anh
  trong bộ eval chỉ đạt 1/6 Hit Rate. Muốn cải thiện phải dùng `OllamaEmbedder`
  hoặc model đa ngôn ngữ.
- **Extractive generator** trích câu nên câu trả lời có thể cụt hoặc ghép nhiều ý;
  chưa đo chất lượng với LLM thật vì máy chưa cài Ollama.
- **Bộ eval nhỏ** (40 câu, 3 tài liệu ngắn) và do người làm project soạn → có thiên lệch.
- **Answer accuracy** là kiểm tra từ khóa tự động, không thay thế chấm thủ công.
- **Chưa có ảnh/video demo, chưa deploy.**
- Không tuyên bố chatbot "không bao giờ hallucinate".

---

## 11. Git

```powershell
git add .
git commit -m "feat(week-6): RAG pipeline, evaluation, FastAPI and Streamlit demo"
```

Quy tắc: một tính năng = một commit nhỏ; không commit `.env`, `.venv/`,
`__pycache__/`, index sinh tự động.

---

## 12. Bản quyền

Chưa chọn license. Sẽ thêm khi hoàn thiện dự án.
