import streamlit as st
from dotenv import load_dotenv
load_dotenv(override = True)
from rag.utils import load_pipeline, change_model, handle_upload, reset_chat_history, cleanup, init_chat
import atexit


st.set_page_config(
    page_title="Q-A with RAG",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="auto"
)

# Header
st.markdown(
    """
    <h1 style='text-align: center;'>🤖📚🔍 Chatbot hỏi đáp tài liệu</h1>
    """,
    unsafe_allow_html=True
)

pipe = load_pipeline()

with st.sidebar:
    selected_model = st.selectbox(
        "🤖 Chọn một model",
        options=pipe.cfg.model_list,
        key="model",
        on_change=lambda: change_model(pipe, st.session_state["model"])
    )

    st.button(label = "🧹Clear Chat",
              help = 'Xoá lịch sử chat',
              on_click=lambda: reset_chat_history(),
              type="primary"
    )

    st.header('Tài liệu: ')
    st.file_uploader(
        label = "Upload các tài liệu",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
        on_change=handle_upload,
        args=(pipe,),
        key="rag_docs",
        help="Sử dụng các tài liệu để truy vấn thông tin.",
        label_visibility="visible"
    )

    with st.expander(f"📚 Các tài liệu trong DB: {len(st.session_state.rag_docs)}"):
        if len(st.session_state.rag_docs) == 0:
            pass
        else:
            for f in st.session_state.rag_docs:
                st.write(f"- {f.name}")

init_chat()

if st.session_state.pop("need_rerun", False):
    st.rerun()

for role, content in st.session_state.chat_history:
    if role == 'user':
        st.chat_message('user').markdown(content)
    else:
        st.chat_message('assistant').markdown(content)

if query := st.chat_input('Nhập câu hỏi...'):
    st.session_state.chat_history.append(('user', query))
    st.chat_message('user').markdown(query)

    answer, sources = pipe.answer(query)

    with st.chat_message('assistant'):
        full_answer = st.write_stream((chunk.text for chunk in answer if chunk.text))
    st.session_state.chat_history.append(('assistant', full_answer))

    hidden = """
    st.subheader("Sources")
        
    for i, src in enumerate(sources, 1):
        with st.expander(f"Source {i}: {src.get('metadata', {}).get('source', 'document')}", expanded=False):
            st.write(src.get("text", "")[:250])
    """

file_list = [file.name for file in st.session_state.get("rag_docs", [])]

atexit.register(lambda: cleanup(file_list, pipe))