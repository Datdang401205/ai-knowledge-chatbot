# Báo cáo đánh giá

- Thời điểm chạy: `2026-11-03T07:56:54+00:00`
- Cấu hình: top_k=4, chunk_size=700, overlap=120
- Embedding: `HashingEmbedder` — Generator: `extractive`
- Index: 12 chunk từ 3 tài liệu

## 1. Kết quả tổng hợp

| Thước đo | Kết quả |
|---|---|
| Số câu hỏi | 40 (36 có đáp án, 4 không có) |
| Hit Rate@k | 86.1% |
| Recall@k | 86.1% |
| MRR | 0.806 |
| Answer accuracy (tự động, theo từ khóa) | 72.5% |
| Tỷ lệ câu trả lời có trích dẫn | 86.1% |
| Abstention accuracy (câu ngoài tài liệu) | 100.0% |
| Độ trễ trung bình mỗi câu | 2.87 ms |
| Số câu lỗi | 0 |

## 2. Kết quả theo loại câu hỏi

| Loại | Số câu | Hit | Recall TB | Đúng (auto) |
|---|---|---|---|---|
| factual | 25 | 25/25 | 100.0% | 23/25 |
| paraphrase | 6 | 1/6 | 16.7% | 0/6 |
| reasoning | 5 | 5/5 | 100.0% | 2/5 |
| unanswerable | 4 | 0/0 | 0.0% | 4/4 |

## 3. Các trường hợp thất bại nổi bật

| ID | Câu hỏi | Nguồn mong đợi | Nguồn truy xuất | Vấn đề |
|---|---|---|---|---|
| q06 | Chunk size và overlap mặc định trong project là bao nhiêu? | rag_co_ban.pdf p3 | rag_co_ban.pdf p2, rag_co_ban.pdf p3 | Câu trả lời thiếu từ khóa / trích dẫn |
| q10 | RAG có loại bỏ hoàn toàn hallucination không? | rag_co_ban.pdf p4 | rag_co_ban.pdf p4, rag_co_ban.pdf p1 | Câu trả lời thiếu từ khóa / trích dẫn |
| q12 | List khác dict như thế nào? | python_fastapi.pdf p1 | python_fastapi.pdf p1 | Câu trả lời thiếu từ khóa / trích dẫn |
| q14 | Vì sao nên dùng virtual environment? | python_fastapi.pdf p1 | python_fastapi.pdf p1 | Câu trả lời thiếu từ khóa / trích dẫn |
| q20 | Dependency injection giúp gì khi kiểm thử? | python_fastapi.pdf p3 | python_fastapi.pdf p3 | Câu trả lời thiếu từ khóa / trích dẫn |
| q31 | How does the system find relevant passages before answering? | rag_co_ban.pdf p2 | — | Không truy xuất đúng nguồn |
| q32 | Why is overlap used between chunks? | rag_co_ban.pdf p3 | — | Không truy xuất đúng nguồn |
| q33 | Which metric tells whether at least one correct source appears in the top-k results? | rag_co_ban.pdf p4 | — | Không truy xuất đúng nguồn |
| q34 | What does FastAPI use to validate input data? | python_fastapi.pdf p2 | — | Không truy xuất đúng nguồn |
| q35 | Which warehouse method ships batches that expire first? | logistics_kpi.pdf p2 | — | Không truy xuất đúng nguồn |
| q36 | Làm sao biết câu trả lời có được nguồn hỗ trợ? | rag_co_ban.pdf p4 | rag_co_ban.pdf p4 | Câu trả lời thiếu từ khóa / trích dẫn |

## 4. Cách đọc kết quả

- **Hit Rate@k** cao nhưng **answer accuracy** thấp: retrieval đúng, khâu sinh câu trả lời kém.
- **Hit Rate@k** thấp: vấn đề nằm ở chunking/embedding/top-k, không phải LLM.
- **Abstention accuracy** thấp: ngưỡng liên quan quá thấp hoặc embedding quá nhạy.
- Answer accuracy ở đây là kiểm tra từ khóa tự động, có thể đánh giá thấp các câu trả lời diễn đạt khác đáp án tham chiếu. Cần chấm thủ công một mẫu để xác nhận.

## 5. Giới hạn của lần đánh giá này

- Bộ câu hỏi do chính người làm project soạn, có thể thiên lệch theo cách diễn đạt của tài liệu.
- Chỉ có 4 câu không có đáp án; chưa đủ để kết luận về khả năng từ chối ngoài phân phối.
- Generator `extractive` chỉ trích câu, không diễn giải; điểm answer accuracy vì vậy thấp hơn so với khi dùng LLM thật.
- Chưa đo chi phí token, chưa đo trên tài liệu dài hàng trăm trang.
