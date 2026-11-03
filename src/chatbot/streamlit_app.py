"""Giao diện web tối giản cho pipeline RAG bằng Streamlit.

Chạy:
    streamlit run src/chatbot/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from chatbot.config import MAX_UPLOAD_BYTES, SAMPLE_DOCS_DIR, VERSION
from chatbot.ingest import SUPPORTED_EXTENSIONS, IngestError
from chatbot.rag_pipeline import RAGPipeline

st.set_page_config(page_title="AI Knowledge Base Chatbot", layout="wide")


def get_pipeline() -> RAGPipeline:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline.from_index()
    return st.session_state.pipeline


st.title("AI Knowledge Base Chatbot")
st.caption(f"Phiên bản {VERSION} — hỏi đáp tài liệu, trả lời có trích dẫn nguồn.")

pipeline = get_pipeline()
stats = pipeline.stats()

with st.sidebar:
    st.header("Kho tri thức")
    st.metric("Số chunk", stats["chunks"])
    st.metric("Số tài liệu", stats["source_count"])
    st.caption(f"Embedding: `{stats['embedding_provider'] or 'chưa build'}`")

    if st.button("Build lại index từ data/sample_docs", use_container_width=True):
        with st.spinner("Đang đọc tài liệu, chunk và tạo embedding..."):
            report = pipeline.build_index(SAMPLE_DOCS_DIR)
        st.session_state.pipeline = pipeline
        st.success(f"Đã tạo {report.chunks} chunk từ {report.documents} tài liệu.")
        for warning in report.warnings:
            st.warning(warning)
        st.rerun()

    uploaded = st.file_uploader(
        "Thêm tài liệu",
        type=[suffix.lstrip(".") for suffix in sorted(SUPPORTED_EXTENSIONS)],
    )
    if uploaded is not None and st.button("Thêm vào kho", use_container_width=True):
        data = uploaded.getvalue()
        if len(data) > MAX_UPLOAD_BYTES:
            st.error(f"File vượt quá {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        elif not data:
            st.error("File rỗng.")
        else:
            suffix = Path(uploaded.name).suffix.lower()
            try:
                with TemporaryDirectory(prefix="rag_upload_") as directory:
                    path = Path(directory) / Path(uploaded.name).name
                    path.write_bytes(data)
                    report = pipeline.add_document(path)
                st.session_state.pipeline = pipeline
                st.success(f"Đã thêm {report.chunks} chunk từ `{uploaded.name}`.")
                for warning in report.warnings:
                    st.warning(warning)
                st.rerun()
            except IngestError as exc:
                st.error(str(exc))

    if st.button("Xoá hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("Hãy hỏi một câu về tài liệu. Ví dụ: “Top-k ảnh hưởng gì tới chất lượng trả lời?”")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Nguồn tham khảo"):
                for source in message["sources"]:
                    st.markdown(
                        f"- `{source['file']}` — trang {source['page']} "
                        f"(điểm {source['score']:.3f})"
                    )

question = st.chat_input("Nhập câu hỏi...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm bằng chứng và soạn câu trả lời..."):
            if len(pipeline.store) == 0:
                st.warning("Kho tri thức đang rỗng. Hãy build index ở thanh bên trái.")
            else:
                answer = pipeline.ask(question)
                st.markdown(answer.text)
                payload = answer.to_dict()
                if payload["sources"]:
                    with st.expander("Nguồn tham khảo"):
                        for source in payload["sources"]:
                            st.markdown(
                                f"- `{source['file']}` — trang {source['page']} "
                                f"(điểm {source['score']:.3f})"
                            )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer.text,
                        "sources": payload["sources"],
                    }
                )
