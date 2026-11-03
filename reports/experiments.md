# Thực nghiệm chunk size × top-k

Bộ câu hỏi: `data/eval_questions.json` (40 câu). Embedding: `HashingEmbedder`, generator: `extractive`, overlap=120, min_score=0.05 (ngưỡng thấp để quan sát nhiều ứng viên hơn).

| Chunk size | Số chunk | top_k | Hit Rate | Recall | MRR | Answer acc | Abstain |
|---|---|---|---|---|---|---|---|
| 400 | 24 | 2 | 86.1% | 86.1% | 0.833 | 70.0% | 50.0% |
| 400 | 24 | 4 | 86.1% | 86.1% | 0.833 | 70.0% | 50.0% |
| 400 | 24 | 6 | 86.1% | 86.1% | 0.833 | 70.0% | 50.0% |
| 700 | 12 | 2 | 86.1% | 86.1% | 0.806 | 70.0% | 50.0% |
| 700 | 12 | 4 | 86.1% | 86.1% | 0.806 | 70.0% | 50.0% |
| 700 | 12 | 6 | 86.1% | 86.1% | 0.806 | 70.0% | 50.0% |
| 1000 | 10 | 2 | 86.1% | 86.1% | 0.806 | 70.0% | 50.0% |
| 1000 | 10 | 4 | 86.1% | 86.1% | 0.806 | 70.0% | 50.0% |
| 1000 | 10 | 6 | 86.1% | 86.1% | 0.806 | 67.5% | 50.0% |

## Cấu hình tốt nhất theo MRR

- chunk_size = **400**, overlap = 120, top_k = **2**
- Hit Rate = 86.1%, Recall = 86.1%, MRR = 0.833

## Nhận xét

- Chunk nhỏ (400) tạo nhiều chunk hơn, recall có thể tăng nhưng mỗi chunk ít ngữ cảnh.
- Chunk lớn (1000) giữ ngữ cảnh tốt hơn nhưng dễ chứa nhiều thông tin nhiễu.
- top_k không làm thay đổi Hit Rate ở bộ tài liệu nhỏ này vì chunk đúng đã ở hạng 1; top_k chỉ có ích khi đáp án nằm rải rác ở nhiều chunk.
- Với bộ tài liệu nhỏ (10 trang), khác biệt giữa các cấu hình không đủ để kết luận cho tài liệu lớn. Cần lặp lại trên bộ tài liệu đa dạng hơn.
- Abstain chỉ đạt 50% vì thực nghiệm hạ `min_score` xuống 0.05 để quan sát nhiều ứng viên; đây không phải cấu hình dùng khi vận hành.

> Số liệu trên được sinh tự động bởi `scripts/run_experiments.py`; không chỉnh tay.
