# app.py
import streamlit as st
from dotenv import load_dotenv

from document_loader import MAX_FILE_SIZE_MB
from rag_pipeline import ask_question, process_uploaded_documents
from vector_store import clear_saved_store, list_indexed_sources, load_vector_store

load_dotenv()

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="centered")

# ---------- Session state (per user, not shared between users) ----------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_vector_store()  # reuse a saved index if one exists
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

st.title("Document Q&A Assistant")
st.write("Upload your documents in the sidebar, process them, and ask questions based on their content.")

# ---------- Sidebar: document management ----------
with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader(
        f"Upload PDF or TXT files (max {MAX_FILE_SIZE_MB} MB each)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploaded_files:
        st.caption("Selected files:")
        for f in uploaded_files:
            st.write(f"• {f.name} ({f.size / 1024:.0f} KB)")

        if st.button("Process Documents", type="primary"):
            with st.spinner("Extracting, chunking and embedding documents..."):
                try:
                    store, info = process_uploaded_documents(uploaded_files)
                    st.session_state.vector_store = store  # replaces any previous documents
                    st.success(
                        f"Processed {info['files']} file(s), {info['pages']} page(s), {info['chunks']} chunks."
                    )
                    for w in info["warnings"]:
                        st.warning(w)
                except Exception as e:
                    st.error(f"Error processing files: {e}")

    indexed = list_indexed_sources(st.session_state.vector_store)
    st.divider()
    if indexed:
        st.caption("Documents currently loaded:")
        for name in indexed:
            st.write(f"{name}")
        if st.button("Clear Documents"):
            st.session_state.vector_store = None
            clear_saved_store()
            st.session_state.uploader_key += 1
            st.rerun()
    else:
        st.info("No documents loaded yet.")

    if st.button("Clear Chat"):
        st.session_state.messages = []

    st.caption(
         "in the original document. Do not upload confidential files without permission."
    )

# ---------- Chat history ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Sources"):
                for src in message["sources"]:
                    st.write(src)

# ---------- New question ----------
if user_query := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating response..."):
            try:
                answer, sources = ask_question(user_query, st.session_state.vector_store)
                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for src in sources:
                            st.write(src)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            except Exception as e:
                st.error(f"An error occurred: {e}")
