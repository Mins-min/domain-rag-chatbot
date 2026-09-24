# rag_pipeline.py
"""RAG workflow: load -> chunk -> embed -> FAISS -> retrieve -> Gemini -> answer + sources."""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from document_loader import load_uploaded_files
from prompt import FALLBACK_MESSAGE, HUMAN_PROMPT, SYSTEM_PROMPT
from vector_store import TOP_K, build_vector_store

load_dotenv()

# Override in .env (GEMINI_MODEL=...) if Google retires or renames a model.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")


def _get_api_key():
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("API key not found. Set GEMINI_API_KEY in your .env file.")
    return key


def process_uploaded_documents(uploaded_files):
    """
    Extract text from uploaded PDF/TXT files, chunk, embed and store in FAISS.
    Returns (vector_store, info) where info has file/page/chunk counts and warnings.
    """
    documents, warnings = load_uploaded_files(uploaded_files)
    if not documents:
        raise ValueError("No readable text found. " + " ".join(warnings))

    store = build_vector_store(documents)
    info = {
        "files": len({d.metadata["source"] for d in documents}),
        "pages": len(documents),
        "chunks": store.index.ntotal,
        "warnings": warnings,
    }
    return store, info


def _format_context(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'Unknown')}, Page {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
        for d in docs
    )


def _format_sources(docs):
    labels = [f"{d.metadata.get('source', 'Unknown')} (Page {d.metadata.get('page', 'N/A')})" for d in docs]
    return list(dict.fromkeys(labels))  # unique, keeps relevance order


def ask_question(user_query: str, vector_store, k: int = TOP_K):
    """Retrieve the top-k chunks for the question and generate a grounded answer. Returns (answer, sources)."""
    if vector_store is None:
        return "Please upload and process documents first using the sidebar.", []

    api_key = _get_api_key()

    docs = vector_store.as_retriever(search_kwargs={"k": k}).invoke(user_query)
    if not docs:
        return FALLBACK_MESSAGE, []

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0, google_api_key=api_key, max_retries=3)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"context": _format_context(docs), "question": user_query}).strip()

    # Don't show sources when the model says the answer is not in the documents.
    if FALLBACK_MESSAGE.lower().rstrip(".") in answer.lower():
        return answer, []
    return answer, _format_sources(docs)
