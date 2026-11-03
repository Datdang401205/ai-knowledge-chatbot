# Bài học Tuần 5 — Đánh giá và cải thiện

> Mục tiêu: chứng minh chất lượng bằng số liệu đo được, không bằng cảm giác.
> Học xong phải tự thiết kế được bộ câu hỏi, chọn thước đo, chạy thực nghiệm và
> đọc đúng điểm yếu của hệ thống.

---

## 1. Vì sao phải có bộ đánh giá?

Một chatbot trả lời hay 2–3 câu bạn thử **không chứng minh** nó hoạt động tốt.
Bộ đánh giá cố định giúp:

- So sánh trước/sau khi đổi chunk size, top-k, embedder.
- Phát hiện hồi quy khi sửa code.
- Biết mình yếu ở đâu: retrieval, generation hay khả năng từ chối.
- Tránh tuyên bố "chính xác 90%" mà không có dữ liệu.

Nguyên tắc: **không đặt số liệu trước khi chạy**. Mọi con số trong `reports/`
được sinh từ lần chạy thật, kèm cấu hình.

---

## 2. Thiết kế bộ câu hỏi

File: `data/eval_questions.json`, 40 câu.

Mỗi câu có:

```json
{
  "id": "q01",
  "category": "factual",
  "question": "RAG là gì?",
  "expected_sources": [{"file": "rag_co_ban.pdf", "page": 1}],
  "reference_answer": "RAG là kiến trúc kết hợp truy xuất tài liệu với mô hình ngôn ngữ."
}
```

### 2.1 Bốn loại câu

| Loại | Số câu | Mục đích |
|---|---|---|
| `factual` | 25 | Hỏi trực tiếp thông tin có trong tài liệu |
| `reasoning` | 5 | Cần suy luận/nối hai ý |
| `paraphrase` | 6 | Hỏi cùng ý bằng tiếng Anh |
| `unanswerable` | 4 | Không có trong tài liệu, phải từ chối |

### 2.2 Vì sao cần câu `unanswerable`?

Nếu chỉ hỏi câu có đáp án, hệ thống luôn trả lời sẽ trông "giỏi". Câu ngoài tài
liệu kiểm tra khả năng **không bịa** — kỹ năng quan trọng nhất của RAG.

### 2.3 Thiên lệch cần ghi nhận

Bộ câu hỏi do chính người làm project soạn, dựa trên tài liệu tự sinh. Nó có thể:

- Dùng đúng từ ngữ trong tài liệu → dễ hơn câu hỏi thật.
- Bỏ sót cách diễn đạt mà người dùng thật sẽ dùng.
- Quá ít câu (40) để kết luận thống kê.

Phải ghi rõ trong báo cáo, không giấu.

---

## 3. Các thước đo

### 3.1 Hit Rate@k

Câu hỏi có **ít nhất một** nguồn đúng trong top-k không?

```text
Hit Rate@k = số câu có ít nhất 1 nguồn đúng / tổng số câu có đáp án
```

Ưu điểm: dễ hiểu. Nhược điểm: không phân biệt đúng ở hạng 1 hay hạng 4.

### 3.2 Recall@k

Bao nhiêu phần trăm **các nguồn đúng** được tìm thấy?

```text
Recall@k = số nguồn đúng tìm thấy / tổng số nguồn đúng
```

Nếu một câu cần 2 trang mà chỉ tìm được 1, Hit Rate vẫn 1 nhưng Recall = 0.5.

### 3.3 MRR (Mean Reciprocal Rank)

Nguồn đúng đầu tiên xuất hiện ở hạng nào?

```text
RR = 1 / rank
MRR = trung bình RR
```

- Hạng 1 → 1.0
- Hạng 2 → 0.5
- Hạng 4 → 0.25
- Không thấy → 0

MRR phạt việc đẩy đáp án xuống cuối danh sách, nên phản ánh chất lượng xếp hạng
tốt hơn Hit Rate.

### 3.4 Answer accuracy

Với mỗi câu, so từ khóa của `reference_answer` với câu trả lời:

```text
coverage = |từ khóa tham chiếu ∩ từ khóa câu trả lời| / |từ khóa tham chiếu|
correct = coverage >= 0.5
```

Đây là **kiểm tra tự động thô**:

- Có thể đánh giá thấp câu trả lời đúng nhưng diễn đạt khác.
- Có thể đánh giá cao câu trả lời chứa từ khóa nhưng sai ý.

Vì vậy báo cáo luôn nhắc: cần chấm thủ công một mẫu.

### 3.5 Citation rate

Tỷ lệ câu trả lời có chứa `[...]` (trích dẫn). Không đo trích dẫn có **đúng**
hay không — muốn đo phải so `sources` với `expected_sources`, đó là Hit Rate.

### 3.6 Abstention accuracy

Với câu `unanswerable`, đúng khi `evidence_found == False`.

```text
Abstention accuracy = số câu từ chối đúng / tổng số câu ngoài tài liệu
```

---

## 4. Chạy đánh giá

```powershell
.\.venv\Scripts\python.exe src\main.py eval --top-k 4
```

Kết quả lần chạy thật (chunk 700/overlap 120, hashing + extractive):

| Thước đo | Kết quả |
|---|---|
| Hit Rate@4 | 86.1% |
| Recall@4 | 86.1% |
| MRR | 0.806 |
| Answer accuracy | 72.5% |
| Citation rate | 86.1% |
| Abstention accuracy | 100% |
| Độ trễ trung bình | 2.79 ms |

Báo cáo chi tiết: `reports/evaluation.md`; dữ liệu thô: `reports/evaluation_results.json`.

### 4.1 Đọc kết quả theo loại câu

Lần chạy đầu cho thấy:

- `factual`: Hit 25/25.
- `reasoning`: Hit 5/5 nhưng answer accuracy thấp.
- `paraphrase`: Hit **1/6** — điểm yếu rõ nhất.
- `unanswerable`: 4/4 từ chối đúng sau khi thêm kiểm tra thực thể lạ.

Kết luận: retrieval hoạt động tốt với câu tiếng Việt dùng từ giống tài liệu;
embedding hashing không chuyển được ngữ nghĩa Anh → Việt.

### 4.2 Đừng chỉ nhìn một con số

Hit Rate 86% nghe tốt, nhưng:

- 6 câu paraphrase gần như fail hết.
- Answer accuracy 72.5% nghĩa là khoảng 1/4 câu trả lời thiếu từ khóa.
- Bộ eval nhỏ, tài liệu ngắn.

Một báo cáo trung thực phải nêu cả ba.

---

## 5. Thực nghiệm chunk size × top-k

Script: `scripts/run_experiments.py`.

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py
```

Chạy 9 cấu hình: chunk 400/700/1000 × top-k 2/4/6, cùng bộ 40 câu.
Kết quả ghi vào `reports/experiments.md`.

### 5.1 Kết quả thật

- Chunk 400 → 24 chunk, MRR 0.833.
- Chunk 700 → 12 chunk, MRR 0.806.
- Chunk 1000 → 10 chunk, MRR 0.806.
- top-k 2/4/6 không đổi Hit Rate vì đáp án đúng đã ở hạng 1.

### 5.2 Cách đọc

- Chunk 400 tốt hơn về MRR trên bộ tài liệu này, nhưng tạo nhiều chunk hơn → tốn
  dung lượng và có thể tăng nhiễu khi tài liệu lớn.
- top-k không tạo khác biệt vì bộ dữ liệu nhỏ; không được kết luận "top-k không
  quan trọng" cho mọi trường hợp.
- Chênh lệch 0.027 MRR không đủ để khẳng định cấu hình 400 tốt hơn về mọi mặt.

**Không chọn cấu hình chỉ vì một bảng số nhỏ.** Giữ mặc định 700/120/top-4 và
ghi lại số liệu để người khác tự đánh giá.

---

## 6. Vòng lặp cải thiện

```text
Đo baseline
   │
   ▼
Xem failure cases
   │
   ├─ retrieval sai  → sửa chunking / embedding / top-k / ngưỡng
   ├─ retrieval đúng, answer sai → sửa generator / prompt
   └─ câu ngoài tài liệu vẫn trả lời → sửa ngưỡng / grounding
   │
   ▼
Chạy lại eval, so sánh cùng bộ câu
```

### 6.1 Ba lỗi thật đã sửa trong Tuần 5

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| Câu hỏi OTD trả về đoạn "đơn vị tính" | Từ phổ biến `tính` chiếm điểm | Thêm IDF vào lexical coverage |
| Trả lời "GPT-5" dù tài liệu không có | Chỉ dựa vào cosine/coverage | Chặn thực thể lạ trước khi retrieval |
| Câu trả lời bắt đầu bằng "trước." | Chunk cắt giữa câu | Bỏ mẩu câu cụt ở đầu passage |

### 6.2 Điều không nên làm

- Chỉnh threshold cho đến khi đúng vài câu mình thích.
- Chấm điểm bằng cách nhìn đáp án rồi sửa bộ eval.
- Bỏ câu khó khỏi bộ đánh giá.
- Báo cáo Hit Rate mà không nói bộ eval nhỏ.

---

## 7. Kiểm thử cho evaluation

| Test | Bảo vệ điều gì |
|---|---|
| `test_load_questions_rejects_empty_list` | Không chạy với bộ rỗng |
| `test_load_questions_rejects_missing_field` | Phát hiện JSON thiếu trường |
| `test_run_evaluation_computes_hit_and_abstention` | Công thức Hit/Abstention đúng |

Điểm quan trọng: test dùng pipeline thật trên tài liệu tạm, không mock số liệu.

---

## 8. Bài tập

1. Thêm 5 câu `factual` mới và 2 câu `unanswerable`, chạy lại `eval`. Hit Rate
   thay đổi thế nào? Có câu nào làm hệ thống bịa không?
2. Chấm thủ công 10 câu trả lời, so với `correct` tự động. Tìm ít nhất 1 trường
   hợp tự động chấm sai.
3. Đổi `CHUNK_SIZE=400`, chạy `eval`, so MRR với mặc định. Ghi lại cả số chunk.
4. Hạ `MIN_RELEVANCE_SCORE` xuống 0.1. Abstention accuracy còn bao nhiêu? Giải thích.
5. Viết thêm một thước đo: `citation_precision` = tỷ lệ nguồn trích dẫn nằm trong
   `expected_sources`. Đây là bài tập mở, cần định nghĩa rõ trước khi code.

---

## 9. Câu hỏi tự kiểm tra

1. Hit Rate@k và Recall@k khác nhau ở đâu? Khi nào cùng giá trị?
2. Vì sao MRR phản ánh xếp hạng tốt hơn Hit Rate?
3. Answer accuracy tự động có thể sai theo hai chiều nào?
4. Vì sao cần câu `unanswerable` trong bộ eval?
5. Bộ eval 40 câu do mình soạn có thiên lệch gì?
6. Vì sao top-k không thay đổi kết quả trên bộ tài liệu nhỏ?
7. Khi retrieval đúng mà answer sai, bạn sửa gì trước?
8. Vì sao không được chọn cấu hình dựa trên một vài ví dụ tốt?

---

## 10. Tóm tắt một dòng

**Đánh giá không phải để khoe điểm, mà để biết mình sai ở đâu và chứng minh rằng
thay đổi tiếp theo thực sự tốt hơn trên cùng một thước đo.**
