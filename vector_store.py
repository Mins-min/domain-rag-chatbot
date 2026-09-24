# vector_store.py
"""Chunking, embeddings (Sentence Transformers) and FAISS storage."""
import os
import shutil
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_loader import load_pdf_documents

DB_PATH = "vector_store/saved_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800      # guidance: 700-1000 characters
CHUNK_OVERLAP = 150   # guidance: 100-150 characters
TOP_K = 4             # guidance: retrieve the top 3-5 chunks


@lru_cache(maxsize=1)
def get_embeddings():
    """Load the embedding model once. Normalised vectors make FAISS ranking equal to cosine similarity."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )
    return splitter.split_documents(documents)


def build_vector_store(documents, embeddings=None, save=True):
    """Chunk documents, embed them, store in FAISS and optionally save the index locally."""
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks could be created from the documents.")

    store = FAISS.from_documents(chunks, embeddings or get_embeddings())
    if save:
        os.makedirs(DB_PATH, exist_ok=True)
        store.save_local(DB_PATH)
    return store


def create_vector_store(directory="documents/"):
    """Index every PDF in `directory` and save the FAISS index (command-line use)."""
    documents = load_pdf_documents(directory)
    if not documents:
        print("No PDF documents found to process.")
        return None
    store = build_vector_store(documents)
    print(f"Indexed {store.index.ntotal} chunks. Saved to {DB_PATH}")
    return store


def load_vector_store(embeddings=None):
    """Load a previously saved FAISS index, or return None if there is none / it is unreadable."""
    if not os.path.exists(os.path.join(DB_PATH, "index.faiss")):
        return None
    try:
        # The index is written by this app itself, so deserialising it is safe.
        return FAISS.load_local(DB_PATH, embeddings or get_embeddings(), allow_dangerous_deserialization=True)
    except Exception:
        return None


def clear_saved_store():
    """Delete the saved index from disk."""
    shutil.rmtree(DB_PATH, ignore_errors=True)


def list_indexed_sources(store):
    """Names of the documents contained in a vector store."""
    if store is None:
        return []
    return sorted({d.metadata.get("source", "Unknown") for d in store.docstore._dict.values()})


if __name__ == "__main__":
    create_vector_store()
