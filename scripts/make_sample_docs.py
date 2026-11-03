"""Tạo bộ tài liệu PDF mẫu (nội dung tổng hợp) cho project.

Chạy:
    .\\.venv\\Scripts\\python.exe scripts\\make_sample_docs.py

Vì sao sinh PDF bằng script thay vì commit file mờ ám?
- Người đọc thấy rõ nội dung và số trang.
- Có thể tái tạo (reproducible) sau khi xoá data/sample_docs.
- Không dùng tài liệu có bản quyền.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "sample_docs"
FONT_PATH = Path("C:/Windows/Fonts/arial.ttf")

DOCUMENTS: dict[str, list[str]] = {
    "rag_co_ban.pdf": [
        """RAG là gì?

RAG (Retrieval-Augmented Generation) là kiến trúc kết hợp truy xuất tài liệu với mô hình ngôn ngữ lớn. Thay vì để LLM trả lời hoàn toàn từ tham số đã học, hệ thống tìm các đoạn văn bản liên quan trong kho tri thức rồi đưa chúng vào ngữ cảnh của prompt. Nhờ vậy câu trả lời bám sát tài liệu và có thể trích dẫn nguồn.

Ba thành phần chính của RAG:
1. Kho tri thức: tài liệu đã được chia nhỏ thành chunk và chuyển thành embedding.
2. Bộ truy xuất: tìm top-k chunk gần câu hỏi nhất.
3. Mô hình sinh: tạo câu trả lời chỉ từ ngữ cảnh được cung cấp.

RAG phù hợp khi kiến thức thay đổi thường xuyên hoặc khi cần dẫn nguồn. Fine-tuning phù hợp hơn khi cần thay đổi phong cách, định dạng đầu ra, hoặc dạy một kỹ năng mới cho mô hình.""",
        """Pipeline RAG gồm hai pha

Pha ngoại tuyến (ingest):
- Trích xuất text từ PDF theo từng trang.
- Chia văn bản thành chunk có overlap.
- Tạo embedding cho từng chunk.
- Lưu vector và metadata (tên file, số trang) vào vector store.

Pha truy vấn (query):
- Chuyển câu hỏi thành embedding bằng cùng mô hình.
- Tìm top-k chunk có cosine similarity cao nhất.
- Lọc theo ngưỡng liên quan tối thiểu.
- Ghép ngữ cảnh và câu hỏi thành prompt có ràng buộc.
- Gọi LLM, nhận câu trả lời và gắn trích dẫn nguồn.

Trong project này, top-k mặc định là 4 và ngưỡng liên quan mặc định là 0.18. Hai tham số này được đo ở Tuần 5, không chọn theo cảm tính.""",
        """Chunking và các tham số quan trọng

Chunk là một đoạn văn bản nhỏ được truy xuất độc lập. Chunk quá nhỏ làm mất ngữ cảnh; chunk quá lớn chứa nhiều thông tin nhiễu và tốn token.

Kích thước mặc định trong project là 700 ký tự với overlap 120 ký tự. Overlap giúp câu trả lời không bị cắt đôi ở ranh giới hai chunk.

Top-k ảnh hưởng trực tiếp:
- Top-k nhỏ (1-2): nhanh, ít nhiễu, nhưng dễ bỏ sót bằng chứng.
- Top-k lớn (8-10): tăng recall, nhưng prompt dài, chậm và dễ làm LLM lạc đề.

Chiến lược tốt là đo Hit Rate@k và Recall@k trên bộ câu hỏi cố định trước khi tăng k.""",
        """Giới hạn của RAG và cách đánh giá

RAG không loại bỏ hoàn toàn hallucination. LLM vẫn có thể diễn giải sai ngữ cảnh, bỏ qua bằng chứng, hoặc trả lời tự tin khi ngữ cảnh mơ hồ.

Các giới hạn thường gặp:
- PDF scan không có lớp text: cần OCR trước.
- Bảng biểu phức tạp bị trích xuất sai thứ tự.
- Câu hỏi mơ hồ khiến retrieval lấy sai chunk.
- Tài liệu dài vượt context window.

Thước đo retrieval:
- Hit Rate@k: tỷ lệ câu hỏi có ít nhất một nguồn đúng trong top-k.
- Recall@k: tỷ lệ nguồn đúng được tìm thấy trên tổng số nguồn đúng.
- MRR: nghịch đảo thứ hạng của nguồn đúng đầu tiên.

Thước đo câu trả lời: độ đúng so với đáp án tham chiếu, tỷ lệ trích dẫn đúng, và tỷ lệ từ chối đúng khi không có bằng chứng.""",
    ],
    "python_fastapi.pdf": [
        """Python nền tảng cho backend

List là dãy có thứ tự và thay đổi được. Dict lưu cặp khóa-giá trị, tra cứu nhanh theo khóa. Tuple có thứ tự nhưng không thay đổi được, phù hợp làm bản ghi bất biến. Set lưu phần tử duy nhất, không có thứ tự, kiểm tra thành viên nhanh.

Hàm trả về giá trị bằng return; print chỉ in ra màn hình. Muốn test tự động thì phải dùng return vì test cần nhận giá trị, không đọc màn hình.

Virtual environment tách thư viện của từng project, tránh xung đột phiên bản giữa các dự án.

Xử lý exception giúp chương trình báo lỗi rõ ràng thay vì crash. Logging ghi lại sự kiện để debug về sau, nhưng không được ghi dữ liệu nhạy cảm như mật khẩu hay nội dung riêng tư.""",
        """FastAPI và API HTTP

FastAPI là framework Python xây dựng API dựa trên type hint và Pydantic. Mỗi endpoint nhận request, kiểm tra dữ liệu đầu vào, gọi logic và trả JSON kèm HTTP status code.

Các method phổ biến:
- GET: đọc dữ liệu, không thay đổi trạng thái.
- POST: tạo mới hoặc thực hiện hành động.
- PUT/PATCH: cập nhật.
- DELETE: xoá.

Mã trạng thái hay dùng: 200 thành công, 201 tạo mới thành công, 400 request sai, 404 không tìm thấy, 413 file quá lớn, 422 dữ liệu không hợp lệ, 500 lỗi phía server.

Tài liệu API tự động có tại /docs khi chạy uvicorn.""",
        """Kiểm thử, logging và an toàn

pytest là framework kiểm thử. TestClient của FastAPI cho phép gọi endpoint mà không cần chạy server thật. Dependency injection giúp thay hàm thật bằng hàm giả trong test.

Ba tầng test nên có:
1. Unit test: hàm thuần như chunking, normalize.
2. Integration test: pipeline từ tài liệu tới câu trả lời.
3. API test: status code, schema JSON, trường hợp lỗi.

Nguyên tắc an toàn: không commit file .env, không đưa API key lên Git, giới hạn kích thước upload, kiểm tra loại file trước khi đọc, và không cho LLM thực thi SQL hoặc code tuỳ ý.""",
    ],
    "logistics_kpi.pdf": [
        """Các chỉ số vận hành chuỗi cung ứng

OTD (On-Time Delivery) là tỷ lệ đơn giao đúng hạn trên tổng đơn. Công thức: số đơn giao đúng hạn chia cho tổng số đơn, nhân 100%.

Fill Rate là tỷ lệ số lượng hàng giao được trên số lượng khách đặt. Fill Rate thấp thường do thiếu tồn kho hoặc lỗi phân bổ.

Inventory Turnover (vòng quay tồn kho) bằng giá vốn hàng bán chia cho tồn kho trung bình. Chỉ số cao nghĩa là hàng luân chuyển nhanh; quá cao có thể dẫn tới thiếu hàng.

Dwell Time là thời gian hàng nằm chờ tại kho hoặc cảng. Lead Time là tổng thời gian từ lúc đặt hàng đến khi nhận hàng.""",
        """Quy trình kho và nguyên tắc xuất hàng

Quy trình kho cơ bản: receiving (nhận hàng), putaway (cất hàng), storage (lưu trữ), picking (lấy hàng), packing (đóng gói), shipping (xuất hàng).

FIFO (First In First Out) xuất lô nhập trước trước. FEFO (First Expired First Out) xuất lô gần hết hạn trước. FEFO bắt buộc với hàng có hạn sử dụng như thực phẩm và dược phẩm.

Kiểm kê định kỳ giúp phát hiện chênh lệch giữa sổ sách và thực tế. Chênh lệch lớn thường do nhập sai đơn vị tính, thất thoát, hoặc ghi nhận trùng lô.""",
        """Control Tower và chi phí

Control Tower là trung tâm giám sát vận hành theo thời gian thực, tổng hợp dữ liệu từ nhiều hệ thống như TMS, WMS và ERP.

Các chỉ số thường hiển thị: số đơn đang vận chuyển, tỷ lệ giao trễ, tồn kho dưới mức tối thiểu, chi phí trên mỗi đơn, và cảnh báo SLA.

Cost per Order bằng tổng chi phí vận hành chia cho số đơn. Muốn giảm chi phí trên mỗi đơn cần tách chi phí cố định và biến đổi, tránh cắt giảm làm tăng tỷ lệ giao trễ.

SLA (Service Level Agreement) là cam kết mức dịch vụ, ví dụ giao trong 24 giờ đối với 95% đơn nội thành.""",
    ],
}


def build_pdf(path: Path, pages: list[str], font_path: Path) -> None:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page(width=595, height=842)  # A4
        if font_path.exists():
            page.insert_font(fontname="F0", fontfile=str(font_path))
            fontname = "F0"
        else:  # pragma: no cover - fallback cho máy không có Arial
            fontname = "helv"
        rect = pymupdf.Rect(50, 50, 545, 792)
        remaining = page.insert_textbox(
            rect,
            text.strip(),
            fontname=fontname,
            fontsize=11,
            lineheight=1.45,
            align=pymupdf.TEXT_ALIGN_LEFT,
        )
        if remaining < 0:
            raise RuntimeError(f"Nội dung trang quá dài trong {path.name}")
    document.set_metadata({"title": path.stem, "author": "AI Knowledge Base sample"})
    document.save(path, garbage=3, deflate=True)
    document.close()


def build_scanned_like_pdf(path: Path) -> None:
    """PDF chỉ có hình vẽ, không có text — mô phỏng bản scan chưa OCR."""
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    page.draw_rect(pymupdf.Rect(80, 80, 515, 300), color=(0, 0, 0), width=2)
    for y in range(120, 280, 30):
        page.draw_line((110, y), (480, y), color=(0.2, 0.2, 0.2), width=1)
    document.save(path, garbage=3, deflate=True)
    document.close()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, pages in DOCUMENTS.items():
        path = OUTPUT / filename
        build_pdf(path, pages, FONT_PATH)
        print(f"created {path.relative_to(ROOT)} ({len(pages)} pages)")

    scanned = OUTPUT / "scan_khong_text.pdf"
    build_scanned_like_pdf(scanned)
    print(f"created {scanned.relative_to(ROOT)} (scan-like, no text)")


if __name__ == "__main__":
    main()
