# AI Knowledge Base Chatbot — Project Plan

> **Mục tiêu:** Tự xây dựng một chatbot hỏi đáp tài liệu có sử dụng LLM và RAG, đủ hoàn thiện để đưa vào CV khi ứng tuyển **AI Engineer Intern**.  
> **Thời gian dự kiến:** 6 tuần, 1–2 giờ/ngày, khoảng 5 ngày/tuần. Có thể kéo dài tùy lịch học.  
> **Nguyên tắc:** Hiểu rồi mới dùng framework; đo chất lượng thay vì chỉ làm giao diện; không tuyên bố tính năng hoặc số liệu chưa kiểm chứng.

---

## 1. Tầm nhìn sản phẩm

Người dùng tải tài liệu lên (bắt đầu với PDF), đặt câu hỏi bằng tiếng Việt hoặc tiếng Anh và nhận câu trả lời dựa trên nội dung tài liệu, kèm trích dẫn nguồn. Khi tài liệu không có thông tin phù hợp, chatbot phải nói rõ chưa tìm thấy bằng chứng thay vì tự bịa.

### Phạm vi MVP

- [ ] Tải và đọc file PDF.
- [ ] Chia văn bản thành các đoạn (chunking).
- [ ] Tạo embeddings và lưu vector.
- [ ] Tìm các đoạn liên quan đến câu hỏi (retrieval).
- [ ] Gửi câu hỏi cùng ngữ cảnh tìm được tới LLM.
- [ ] Trả lời kèm tên file và trang/đoạn nguồn nếu có thể xác định.
- [ ] Xử lý câu hỏi không có bằng chứng phù hợp.
- [ ] Có giao diện đơn giản và hướng dẫn chạy local.
- [ ] Có bộ câu hỏi đánh giá và báo cáo kết quả thực nghiệm.
- [ ] Có README, sơ đồ kiến trúc, kiểm thử và ảnh/video demo.

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

- [ ] Tạo thư mục project và môi trường ảo `.venv`.
- [ ] Kiểm tra phiên bản Python và pip.
- [ ] Ôn hàm, chuỗi, điều kiện, vòng lặp, `return`, import, exception.
- [ ] Viết chatbot rule-based CLI phiên bản 0.1.
- [ ] Tách logic trả lời khỏi hàm `main()`.
- [ ] Thêm xử lý input rỗng và lệnh thoát.
- [ ] Khởi tạo Git repository và commit đầu tiên.
- [ ] Viết README sơ bộ: mục tiêu, cài đặt, cách chạy.

**Bài tập kiểm tra:** tự thêm một intent mới; giải thích `if __name__ == "__main__":`; chạy chương trình mà không cần chép lại từng bước.

**Definition of Done:** chạy được trên máy local, có code dễ đọc, Git commit và README sơ bộ.

### Tuần 2 — Làm quen LLM và prompt

**Mục tiêu:** thay luật cố định bằng mô hình ngôn ngữ.

- [ ] Tìm hiểu LLM, prompt, system message, user message, token, context window.
- [ ] Cài và chạy Ollama nếu phần cứng đáp ứng; chọn một model nhỏ phù hợp.
- [ ] Gọi mô hình từ Python qua API local.
- [ ] Tách cấu hình model và URL khỏi logic chính.
- [ ] Xử lý timeout, lỗi kết nối và câu trả lời rỗng.
- [ ] Viết prompt yêu cầu trả lời bằng ngôn ngữ phù hợp và thừa nhận khi không biết.
- [ ] Không lưu secret hoặc thông tin riêng tư trong Git.

**Bài tập kiểm tra:** so sánh cùng một câu hỏi với 2 prompt; ghi lại khác biệt và giải thích vì sao prompt không đảm bảo sự thật tuyệt đối.

**Definition of Done:** chatbot nhận input, gọi được LLM và xử lý lỗi cơ bản.

### Tuần 3 — PDF, chunking và embeddings

**Mục tiêu:** chuẩn bị dữ liệu để truy xuất.

- [ ] Chọn 3–5 tài liệu PDF phù hợp và được phép sử dụng.
- [ ] Trích xuất text theo trang; ghi nhận file và số trang.
- [ ] Kiểm tra PDF scan không có text và xử lý bằng thông báo phù hợp.
- [ ] Viết hàm chunking có kích thước và overlap cấu hình được.
- [ ] Kiểm tra chunk rỗng, chunk quá ngắn và mất metadata.
- [ ] Tạo embeddings bằng mô hình hỗ trợ ngôn ngữ của tài liệu.
- [ ] Thử tính tương đồng giữa các đoạn có liên quan và không liên quan.

**Bài tập kiểm tra:** in ra 5 chunks, kiểm tra trang nguồn; thay đổi chunk size/overlap và ghi nhận tác động.

**Definition of Done:** pipeline đọc PDF tạo được chunks có metadata và embeddings tái lập được.

### Tuần 4 — Retrieval và RAG end-to-end

**Mục tiêu:** chatbot trả lời dựa trên tài liệu, có nguồn.

- [ ] Chọn một vector store: Chroma hoặc FAISS.
- [ ] Lưu embeddings cùng metadata.
- [ ] Với mỗi câu hỏi, truy xuất top-k chunks.
- [ ] Hiển thị nội dung truy xuất để debug trước khi gọi LLM.
- [ ] Viết prompt chứa câu hỏi và ngữ cảnh truy xuất.
- [ ] Yêu cầu câu trả lời dựa trên ngữ cảnh, không tự thêm dữ kiện.
- [ ] Trả về nguồn file/trang tương ứng.
- [ ] Xử lý trường hợp retrieval không đủ liên quan.
- [ ] Tạo ít nhất 20 câu hỏi đánh giá ban đầu.

**Bài tập kiểm tra:** hỏi một câu có câu trả lời trong tài liệu, một câu ngoài tài liệu và một câu mơ hồ; xem nguồn truy xuất có phù hợp không.

**Definition of Done:** chạy được toàn bộ luồng PDF → chunks → embeddings → retrieval → LLM → answer + sources.

### Tuần 5 — Đánh giá và cải thiện

**Mục tiêu:** chứng minh chất lượng bằng dữ liệu thay vì cảm giác.

- [ ] Tạo bộ đánh giá khoảng 30–50 câu hỏi, có câu trả lời tham chiếu và nguồn mong đợi.
- [ ] Đo Hit Rate@k hoặc Recall@k cho retrieval.
- [ ] Kiểm tra answer correctness bằng tiêu chí rõ ràng; có thể chấm thủ công nếu dữ liệu nhỏ.
- [ ] Đánh giá source/grounding: câu trả lời có được hỗ trợ bởi chunks không?
- [ ] Thử các chunk size và top-k khác nhau.
- [ ] Ghi lại baseline, thay đổi, kết quả và giới hạn.
- [ ] Thêm unit tests cho chunking và retrieval.
- [ ] Không chọn cấu hình chỉ vì một vài ví dụ tốt.

**Definition of Done:** có báo cáo đánh giá, giải thích được cách đo, kết quả và các trường hợp thất bại. Không tự đặt số liệu trước khi chạy.

### Tuần 6 — API, giao diện và portfolio

**Mục tiêu:** đóng gói thành sản phẩm demo được.

- [ ] Tạo API FastAPI cho health check, ingest tài liệu và hỏi đáp.
- [ ] Tạo giao diện Streamlit tải PDF, nhập câu hỏi và xem nguồn.
- [ ] Thêm trạng thái đang xử lý và thông báo lỗi dễ hiểu.
- [ ] Kiểm tra file không hợp lệ và giới hạn kích thước file.
- [ ] Thêm logging không ghi nội dung nhạy cảm không cần thiết.
- [ ] Viết tests cho luồng chính và các lỗi phổ biến.
- [ ] Cập nhật `.gitignore`, `.env.example`, `requirements.txt`.
- [ ] Viết README hoàn chỉnh và sơ đồ kiến trúc.
- [ ] Chụp ảnh/gif hoặc quay video demo 1–2 phút.
- [ ] Triển khai demo nếu có phương án miễn phí phù hợp; nếu không, hướng dẫn chạy local đầy đủ.
- [ ] Viết phần mô tả project trong CV dựa trên tính năng và số liệu đã xác minh.

**Definition of Done:** người khác có thể làm theo README để chạy project; có demo, tests, kết quả đánh giá và giới hạn được ghi rõ.

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

### Trạng thái hiện tại

- [ ] Tuần 1 — Python project và chatbot CLI
- [ ] Tuần 2 — LLM và prompt
- [ ] Tuần 3 — PDF, chunking và embeddings
- [ ] Tuần 4 — Retrieval và RAG
- [ ] Tuần 5 — Đánh giá và cải thiện
- [ ] Tuần 6 — API, UI và portfolio

**Bước tiếp theo:** hoàn thành chatbot CLI đầu tiên, chạy `python src/main.py`, rồi tự thêm một intent mới và giải thích được từng phần của chương trình.
