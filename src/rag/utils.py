from .pipeline import RAGPipeline
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_pipeline():
    return RAGPipeline()

def change_model(pipeline: RAGPipeline, model_name: str):
    pipeline.cfg.llm_cfg.change_model(model_name)

def handle_upload(pipeline: RAGPipeline):
    current_files = [file.name for file in st.session_state.rag_docs]
    prev_files = st.session_state.get("prev_files", [])

    added_files = [file for file in st.session_state.rag_docs if file.name not in prev_files]
    removed_files = list(set(prev_files) - set(current_files))

    st.session_state.prev_files = list(current_files)

    for file in added_files:
        pipeline.rebuild_index(file)

    for file in removed_files:
        pipeline.vec_store.delete_by_source(file)

def init_chat():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    greeting = "Xin chào! 👋 Mình là trợ lý RAG. Bạn hãy upload tài liệu và hỏi mình bất cứ điều gì về chúng nhé."
    if not st.session_state.chat_history:
        st.session_state.chat_history.append(("assistant", greeting))

def reset_chat_history():
    st.session_state.chat_history.clear()
    init_chat()
    st.session_state.need_rerun = True

def cleanup(files, pipeline: RAGPipeline):
    for file in files:
        pipeline.vec_store.delete_by_source(file)