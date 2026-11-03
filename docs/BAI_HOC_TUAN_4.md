# Bài học Tuần 4 — Retrieval và RAG end-to-end

> Mục tiêu: từ câu hỏi, tìm đúng đoạn tài liệu, rồi trả lời **chỉ dựa trên đoạn đó**
> và trích dẫn nguồn. Học xong phải phân biệt được lỗi retrieval với lỗi generation.

---

## 1. Hai bước không được nhầm

```text
Câu hỏi
  │
  ▼
RETRIEVAL  → tìm top-k chunk liên quan        (retrieval.py)
  │
  ▼
GENERATION → diễn đạt câu trả lời từ chunk     (generation.py)
  │
  ▼
Answer + Source(file, page)
```

Nếu câu trả lời sai, câu hỏi đầu tiên luôn là: **retrieval có tìm đúng chunk không?**

- Retrieval sai → generation không có cơ hội đúng.
- Retrieval đúng nhưng trả lời sai → lỗi prompt/generator.

Vì vậy project có lệnh debug riêng:

```powershell
python src\main.py ask --retrieve-only "OTD được tính thế nào?"
```

---

## 2. Vector store tự viết

### 2.1 Lưu gì?

Mỗi bản ghi gồm `Chunk` + vector + metadata:

```json
{
  "id": "a1b2...",
  "text": "...",
  "source": "logistics_kpi.pdf",
  "page": 1,
  "chunk_index": 0,
  "start_char": 0,
  "end_char": 680,
  "vector": [0.01, -0.02, ...]
}
```

`manifest.json` lưu thêm provider, dimension và cấu hình retrieval. Nhờ vậy biết
index được tạo bằng embedder nào, chunk bao nhiêu — tránh dùng vector 1024 chiều
với embedder 768 chiều.

### 2.2 Cosine similarity

```python
scores = self._vectors @ query   # đã chuẩn hoá L2
order = np.argsort(-scores)[:top_k]
```

Tích vô hướng của hai vector đơn vị chính là cosine. Điểm càng cao càng gần.

### 2.3 Vì sao chuẩn hoá lại lần nữa?

Store không tin embedder đã làm đúng. Chuẩn hoá trong `add()` và `search()` là
lớp phòng thủ: nếu ai đó truyền vector chưa chuẩn hoá, cosine vẫn đúng.

### 2.4 Ghi file an toàn

`save()` ghi ra `.tmp` rồi `replace()`. Nếu tiến trình chết giữa chừng, file cũ
vẫn còn nguyên thay vì bị cắt cụt.

---

## 3. Retrieval hybrid: vì sao cần hai tín hiệu?

### 3.1 Điểm vector

Cosine bắt được tương đồng bề mặt (và ngữ nghĩa nếu có model tốt). Nhưng với
hashing, một câu hỏi ngắn như `"OTD được tính thế nào?"` có ít feature, điểm
cosine rất thấp và không ổn định.

### 3.2 Điểm từ khóa có IDF

```text
coverage = tổng IDF của các từ khóa câu hỏi có trong chunk / tổng IDF
```

IDF = `log((N + 1) / (df + 1)) + 1`, trong đó `df` là số chunk chứa từ đó.

- `"OTD"` xuất hiện ở 1 chunk → IDF cao.
- `"tính"` xuất hiện ở nhiều chunk → IDF thấp.

Nhờ IDF, chunk chứa thuật ngữ hiếm thắng chunk chỉ chứa từ phổ biến. Đây là bug
thật đã gặp: câu hỏi OTD ban đầu trả về đoạn "Chênh lệch lớn thường do nhập sai
đơn vị tính" chỉ vì khớp từ `"tính"`.

### 3.3 Công thức cuối

```text
score = 0.5 × cosine + 0.5 × lexical_coverage
```

Hai trọng số này là lựa chọn có ý thức cho baseline hashing. Khi thay bằng
embedding ngữ nghĩa thật, có thể tăng trọng số cosine.

### 3.4 Lấy nhiều ứng viên rồi mới lọc

```python
candidates = store.search(vector, top_k=max(k * 3, 12), min_score=None)
```

Nếu lọc ngưỡng ngay ở vector store, chunk đúng nhưng cosine thấp sẽ bị loại trước
khi lexical kịp "cứu". Vì vậy lấy rộng rồi rerank.

---

## 4. Chống bịa (grounding)

### 4.1 Ngưỡng liên quan

`MIN_RELEVANCE_SCORE=0.25`. Dưới ngưỡng → trả `NO_EVIDENCE_MESSAGE`, không gọi
generator. Ngưỡng quá thấp → trả lời cả câu ngoài tài liệu; quá cao → bỏ sót câu
trả lời đúng.

### 4.2 Chặn thực thể lạ

Hỏi `"Hệ thống dùng mô hình GPT-5 nào?"` trong khi tài liệu không nhắc GPT-5:
các từ chung (`hệ thống`, `mô hình`) vẫn khớp, nên nếu chỉ dựa vào coverage sẽ
trả lời sai.

`Retriever._has_unknown_entity()` tìm tên riêng/viết tắt (chữ hoa, chữ số) không
xuất hiện trong bất kỳ chunk nào. Nếu có → trả rỗng ngay.

Đây là heuristic, không phải giải pháp hoàn chỉnh: nó có thể bỏ sót thực thể viết
thường hoặc bắt nhầm từ đầu câu. Cần ghi rõ giới hạn.

### 4.3 Generator chỉ được dùng ngữ cảnh

`RAG_SYSTEM_PROMPT` quy định:

1. Không có trong ngữ cảnh → nói chưa tìm thấy.
2. Không bịa số liệu/tên riêng.
3. Trả lời đúng ngôn ngữ câu hỏi.
4. Trích dẫn `[file, trang N]`.
5. Mâu thuẫn giữa các đoạn → nói rõ.

`ExtractiveGenerator` làm đúng tinh thần này ở mức tối giản: **chỉ chọn câu từ
chunk**, không tự sinh chữ.

---

## 5. Generator extractive hoạt động thế nào?

### 5.1 Tách câu đúng cách

PDF ngắt dòng giữa câu. Nếu tách theo `\n`, ta được các mẩu cụt:

```text
TestClient của FastAPI cho phép gọi endpoint mà không cần chạy
server thật.
```

Project gộp các dòng chưa kết thúc câu thành **logical line**, rồi mới tách câu.
Nhờ vậy câu trả lời không bị cắt ở `"không cần chạy"`.

### 5.2 Gộp tiêu đề với nội dung

Dòng `"Top-k ảnh hưởng trực tiếp:"` một mình vô nghĩa. `_passages()` gộp tiêu đề
kết thúc bằng `:` với tối đa 2 câu tiếp theo.

### 5.3 Chọn câu

Điểm một câu gồm:

```text
coverage từ khóa
+ 0.5 × điểm retrieval
+ 0.6 nếu là câu định nghĩa đúng chủ thể
+ 0.25 nếu chứa chủ thể
- 0.08 × thứ hạng chunk
```

Sau đó:

- Ưu tiên câu chứa chủ thể câu hỏi (`OTD`, `FastAPI`).
- Chọn tối đa 2 câu.
- Chỉ trích dẫn nguồn thực sự được dùng.

### 5.4 Vì sao chỉ 2 câu?

Câu trả lời dài dễ chứa thông tin không được hỏi. Baseline thà ngắn và đúng
nguồn còn hơn dài mà loãng. Khi dùng LLM, prompt có thể yêu cầu tổng hợp tốt hơn.

---

## 6. Luồng end-to-end trong `rag_pipeline.py`

```python
pipeline = RAGPipeline.empty()
pipeline.build_index()                 # ingest + chunk + embed + save

answer = pipeline.ask("RAG là gì?")
answer.text                            # câu trả lời
answer.sources                         # Source(file, page, score, chunk_id)
answer.retrieved                       # toàn bộ chunk đã truy xuất
answer.evidence_found                  # có bằng chứng hay không
```

`RAGPipeline` là **điểm vào duy nhất**. CLI, FastAPI và Streamlit đều gọi nó,
nên ba giao diện không thể lệch logic nhau.

---

## 7. Kiểm thử Tuần 4

| Test | Bảo vệ điều gì |
|---|---|
| `test_add_and_search_returns_best_match_first` | Cosine xếp hạng đúng |
| `test_add_ignores_duplicate_ids` | Không nhân đôi chunk |
| `test_save_and_load_roundtrip` | Index đọc lại được |
| `test_retrieve_finds_chunk_with_rare_term` | IDF giúp thuật ngữ hiếm thắng |
| `test_unknown_entity_abstains` | Không trả lời GPT-5 |
| `test_ask_returns_answer_with_sources` | Answer có nguồn |
| `test_ask_abstains_for_unknown_topic` | Từ chối đúng |
| `test_persist_and_reload_index` | Pipeline load lại vẫn trả lời |

---

## 8. Bài tập

1. Chạy `ask --retrieve-only` với 3 câu: một câu có đáp án, một câu ngoài tài liệu,
   một câu mơ hồ. Ghi lại top-3 và nhận xét.
2. Đổi `vector_weight`/`lexical_weight` trong `RetrievalConfig` thành 0.8/0.2 và
   0.2/0.8. Chạy `eval`, so sánh MRR. Giải thích kết quả.
3. Hạ `MIN_RELEVANCE_SCORE` xuống 0.05. Chạy `eval`, xem abstention accuracy.
4. Tự tạo một câu hỏi chứa tên riêng không có trong tài liệu. Kiểm tra hệ thống
   từ chối.
5. Đọc `_passages()` và tạo một chunk giả có tiêu đề kết thúc `:`. Kiểm tra câu
   trả lời có gộp đúng không.

---

## 9. Câu hỏi tự kiểm tra

1. Retrieval khác generation ở đâu? Lỗi nào dễ chẩn đoán hơn?
2. Vì sao lấy `top_k * 3` ứng viên trước khi rerank?
3. IDF là gì và vì sao nó giúp câu hỏi có thuật ngữ hiếm?
4. Vì sao chỉ dựa vào cosine lại yếu với hashing embedding?
5. Ngưỡng liên quan ảnh hưởng thế nào tới hai loại lỗi: bỏ sót và bịa?
6. Vì sao phải kiểm tra thực thể lạ thay vì chỉ dựa vào coverage?
7. Vì sao chỉ trích dẫn nguồn thực sự dùng, không trích dẫn mọi nguồn truy xuất?
8. Nếu retrieval đúng mà câu trả lời vẫn sai, bạn kiểm tra gì tiếp theo?

---

## 10. Tóm tắt một dòng

**Retrieval tìm bằng chứng, generation chỉ được nói lại bằng chứng; tách hai bước
là điều kiện để debug và để trích dẫn đúng.**
