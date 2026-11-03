# Bài học Tuần 3 — PDF, chunking và embeddings

> Mục tiêu: biến tài liệu thô thành các đoạn nhỏ có metadata và vector, để Tuần 4
> có thể tìm đúng bằng chứng. Học xong phải trả lời được: **vì sao phải chunk,
> chunk ảnh hưởng gì, embedding là gì, và làm sao biết pipeline đọc đúng.**

---

## 1. Luồng dữ liệu của Tuần 3

```text
data/sample_docs/*.pdf
        │
        ▼
ingest.load_pages()          # mỗi trang → PageText(source, page, text, has_text)
        │
        ▼
chunking.chunk_pages()       # mỗi trang → nhiều Chunk có metadata
        │
        ▼
embeddings.HashingEmbedder   # mỗi chunk → vector 1024 chiều, chuẩn hoá L2
        │
        ▼
vector_store.VectorStore     # lưu chunk + vector, tìm bằng cosine
```

Ba file tương ứng ba câu hỏi khác nhau:

| File | Câu hỏi nó trả lời |
|---|---|
| `ingest.py` | Đọc được chữ từ PDF chưa? Có giữ số trang không? |
| `chunking.py` | Cắt ở đâu để không mất nghĩa? |
| `embeddings.py` | Làm sao so sánh nghĩa của hai đoạn văn? |

---

## 2. Vì sao không nhét cả tài liệu vào LLM?

LLM có **context window** hữu hạn. Nhét cả tài liệu 100 trang gây ba vấn đề:

1. **Tốn token** → chậm và có thể vượt giới hạn.
2. **Loãng thông tin** → đoạn không liên quan làm model dễ trả lời sai.
3. **Không trích dẫn được** → không biết câu trả lời lấy từ trang nào.

Chunking biến bài toán "đọc cả cuốn sách" thành "tìm đúng 3–5 đoạn rồi đọc".
Đây là ý tưởng cốt lõi của RAG.

---

## 3. Đọc PDF bằng PyMuPDF

### 3.1 Vì sao giữ số trang?

Người dùng cần kiểm chứng câu trả lời. Nếu chỉ trả về `"rag_co_ban.pdf"` mà không
có trang, họ phải lật cả file. `PageText` giữ `page` ngay từ khâu đọc:

```python
PageText(source=path.name, page=page_number, text=text, has_text=bool(text.strip()))
```

Metadata này phải đi theo chunk đến tận câu trả lời. Mất metadata ở bất kỳ bước
nào là mất khả năng trích dẫn.

### 3.2 PDF scan không có text

PyMuPDF trả `page.get_text()` rỗng với PDF chỉ chứa ảnh. Nếu bỏ qua im lặng,
người dùng sẽ thấy chatbot "không biết gì" mà không hiểu vì sao.

Cách xử lý trong project:

```python
has_text = bool(text.strip())
```

`build_index` gom các trang không có text và trả cảnh báo:

```text
'scan_khong_text.pdf' không có text (có thể là PDF scan); cần OCR trước.
```

**Bài học:** lỗi dữ liệu phải nổi lên trên, không được nuốt.

### 3.3 Ký tự ẩn trong PDF

PDF sinh ra non-breaking space (`\xa0`) và soft hyphen (`\xad`). Nếu không xử lý:

- `Top\xadk` thành `Topk` (mất dấu gạch) nếu xoá `\xad`.
- Tìm kiếm `"Top-k"` không khớp.

Project thay `\xad` bằng `-`, `\xa0` bằng space trong `clean_text()`. Đây là bug
thật gặp khi build index lần đầu: `Top-k` biến thành `Topk`, làm hỏng cả hiển thị
lẫn truy vết.

---

## 4. Chunking: cắt ở đâu và cắt bao nhiêu?

### 4.1 Kích thước

- Chunk quá nhỏ: mất ngữ cảnh, câu trả lời cụt.
- Chunk quá lớn: chứa nhiều thông tin nhiễu, tốn token, embedding bị "pha loãng".

Mặc định project: `CHUNK_SIZE=700` ký tự, `CHUNK_OVERLAP=120`. Đây là điểm bắt
đầu, **không phải chân lý**; Tuần 5 đo lại với 400/700/1000.

### 4.2 Overlap

Overlap là phần lặp lại giữa hai chunk liền nhau. Nó giúp câu nằm đúng ranh giới
vẫn xuất hiện trọn vẹn ở ít nhất một chunk.

Ví dụ:

```text
Chunk 1: ... Top-k nhỏ thì nhanh nhưng dễ bỏ sót bằng chứng. Top-k lớn thì
Chunk 2: Top-k lớn thì tăng recall nhưng prompt dài và dễ nhiễu...
```

Nếu không overlap, chunk 1 có thể kết thúc giữa câu.

### 4.3 Cắt ưu tiên ranh giới tự nhiên

`_find_break()` thử lần lượt: đoạn văn (`\n\n`), kết câu (`. `, `!\n`), dòng,
khoảng trắng. Cắt ở khoảng trắng tệ hơn cắt ở cuối câu, nhưng vẫn tốt hơn cắt
giữa từ.

### 4.4 ID ổn định

```python
raw = f"{source}|{page}|{index}|{start}"
chunk_id = sha1(raw)[:16]
```

Cùng tài liệu + cùng tham số chunk → cùng ID. Nhờ vậy:

- Build lại index không tạo bản trùng.
- `add_document` gọi hai lần không nhân đôi chunk.
- Test so sánh ID giữa hai lần chạy.

### 4.5 metadata của Chunk

```text
id, text, source, page, chunk_index, start_char, end_char
```

`start_char/end_char` cho biết vị trí trong trang gốc — hữu ích khi cần debug
hoặc highlight.

---

## 5. Embedding: biến chữ thành vector

### 5.1 Ý tưởng

Embedding ánh xạ văn bản sang vector số sao cho **văn bản giống nghĩa thì vector
gần nhau**. Khoảng cách thường dùng là cosine:

```text
cosine(a, b) = (a · b) / (|a| × |b|)
```

Nếu đã chuẩn hoá L2 (`|a| = |b| = 1`) thì cosine = tích vô hướng.

### 5.2 Vì sao project dùng hashing thay vì model?

`HashingEmbedder` băm từng feature vào vector cố định:

- Từ nguyên vẹn: trọng số 1.0.
- Cụm 3 ký tự: trọng số 0.15.
- Bigram hai từ: trọng số 0.8.

Ưu điểm: offline, tất định, không tải model, test nhanh.
Nhược điểm: **không hiểu đồng nghĩa**. `"ô tô"` và `"xe hơi"` gần như không liên
quan; câu hỏi tiếng Anh không khớp tài liệu tiếng Việt.

Đây là lựa chọn có ý thức: học cơ chế trước, thay `OllamaEmbedder`
(`nomic-embed-text`) khi cần chất lượng ngữ nghĩa.

### 5.3 Chuẩn hoá L2

```python
norms = np.linalg.norm(matrix, axis=1, keepdims=True)
norms[norms == 0] = 1.0
return matrix / norms
```

Chia cho norm để mọi vector dài 1. Nếu không, chunk dài sẽ có tích vô hướng lớn
hơn chỉ vì nhiều từ hơn, không phải vì liên quan hơn.

---

## 6. Vector store tối giản

Tuần 3 chưa cần truy vấn, nhưng đã lưu được:

```python
store.add(chunks, vectors, embedding_provider="HashingEmbedder")
store.save()  # data/index/store.json
```

Vì sao JSON + numpy thay vì Chroma/FAISS?

- Vài nghìn chunk: numpy đủ nhanh.
- Nhìn được toàn bộ dữ liệu, dễ debug.
- Interface `add/search/save/load` đủ để thay backend sau mà không đổi retrieval.

---

## 7. Kiểm thử Tuần 3

| Test | Bảo vệ điều gì |
|---|---|
| `test_chunk_page_keeps_source_and_page` | Không mất metadata |
| `test_chunk_ids_are_stable_between_runs` | Build lại không sinh ID mới |
| `test_chunk_page_ignores_blank_page` | Trang rỗng không tạo chunk rỗng |
| `test_load_pages_reads_utf8_text` | Đọc được tiếng Việt |
| `test_load_pages_rejects_unsupported_extension` | Báo lỗi rõ, không đoán |
| `test_embedding_is_deterministic_and_normalized` | Vector ổn định, norm = 1 |
| `test_similar_text_scores_higher_than_unrelated_text` | Embedding có ý nghĩa |

Chạy:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_chunking.py tests/test_ingest.py tests/test_embeddings.py -q
```

---

## 8. Bài tập

1. Chạy `ingest` rồi mở `data/index/store.json`, tìm 1 chunk và đối chiếu
   `page`, `start_char` với PDF gốc.
2. Đổi `CHUNK_SIZE` thành 300 và 1200, chạy `ingest`, ghi lại số chunk. Giải thích
   vì sao số chunk không tỉ lệ nghịch tuyệt đối với kích thước.
3. Tạo một file TXT tiếng Việt có từ "ô tô" và một câu hỏi dùng "xe hơi".
   Quan sát điểm tương đồng. Giải thích vì sao hashing thất bại.
4. Tự tạo PDF scan (hoặc dùng `scan_khong_text.pdf`) và kiểm tra cảnh báo.
5. Viết thêm test cho trường hợp `chunk_size=0` và `overlap >= chunk_size`.

---

## 9. Câu hỏi tự kiểm tra

1. Vì sao phải giữ `page` trong `Chunk`?
2. Overlap giải quyết vấn đề gì? Nó có làm tăng số token không?
3. Vì sao chunk quá lớn lại làm retrieval kém?
4. Hashing embedding khác embedding ngữ nghĩa ở điểm nào?
5. Vì sao phải chuẩn hoá L2 trước khi tính cosine?
6. PDF scan sẽ được xử lý thế nào trong project? Muốn hỗ trợ thật thì cần gì?
7. Chunk ID ổn định mang lại lợi ích gì cho `add_document`?

---

## 10. Tóm tắt một dòng

**Tuần 3 biến tài liệu thành các đoạn có địa chỉ (file, trang) và vector đại diện,
để Tuần 4 có thể tìm đúng chỗ thay vì đọc lại toàn bộ tài liệu.**
