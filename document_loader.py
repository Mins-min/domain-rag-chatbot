# document_loader.py
"""Text extraction from PDF / TXT files. Keeps document name and page number as metadata."""
import io
import os

from langchain_core.documents import Document
from pypdf import PdfReader

MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = (".pdf", ".txt")


def validate_file(name: str, size_bytes: int):
    """Return an error message if the file is not allowed, otherwise None."""
    if not name.lower().endswith(ALLOWED_EXTENSIONS):
        return f"'{name}': unsupported file type (only PDF and TXT are allowed)."
    if size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
        return f"'{name}': file is larger than {MAX_FILE_SIZE_MB} MB."
    if size_bytes == 0:
        return f"'{name}': file is empty."
    return None


def _pdf_to_documents(reader: PdfReader, source: str):
    """Read every page with pypdf. Empty pages are skipped safely. Page numbers are 1-based."""
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("the PDF is password-protected.")

    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            continue  # unreadable page: skip it instead of crashing
        if not text:
            continue  # blank / scanned page (OCR is an optional extension)
        documents.append(Document(page_content=text, metadata={"source": source, "page": page_number}))
    return documents


def load_pdf_file(file_path: str, display_name: str = None):
    """Load a PDF from disk."""
    return _pdf_to_documents(PdfReader(file_path), display_name or os.path.basename(file_path))


def load_uploaded_files(uploaded_files):
    """
    Load Streamlit UploadedFile objects (PDF or TXT) directly from memory.
    Returns (documents, warnings). One bad file never stops the others.
    """
    documents, warnings = [], []
    for uploaded in uploaded_files:
        name = uploaded.name
        data = uploaded.getvalue()

        error = validate_file(name, len(data))
        if error:
            warnings.append(error)
            continue

        try:
            if name.lower().endswith(".pdf"):
                docs = _pdf_to_documents(PdfReader(io.BytesIO(data)), name)
            else:
                text = data.decode("utf-8", errors="ignore").strip()
                docs = [Document(page_content=text, metadata={"source": name, "page": 1})] if text else []
        except Exception as e:
            warnings.append(f"'{name}': could not be read ({e}).")
            continue

        if not docs:
            warnings.append(f"'{name}': no extractable text found (scanned PDFs need OCR).")
            continue
        documents.extend(docs)

    return documents, warnings


def load_pdf_documents(directory: str = "documents/"):
    """Load all PDFs from a folder (used for sample documents / command-line indexing)."""
    documents = []
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return documents

    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(".pdf"):
            continue
        try:
            documents.extend(load_pdf_file(os.path.join(directory, filename), filename))
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return documents
