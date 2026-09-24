# Domain-Specific RAG Chatbot for PDF Question Answering

A Streamlit chatbot that answers questions **only from your uploaded PDF/TXT documents** and shows the
source document and page number for every answer. If the answer is not in the documents, it says so
instead of inventing one.

## Workflow

```
Upload PDF/TXT files
        |
Extract text page by page (pypdf, keeps file name + page number)
        |
Split into chunks (800 characters, 150 overlap)
        |
Embed chunks (Sentence Transformers: all-MiniLM-L6-v2)
        |
Store in FAISS (saved locally in vector_store/saved_index)
        |
User asks a question -> retrieve top 4 chunks
        |
Send context + question to Gemini with a strict prompt
        |
Display answer + source document and page
```

## Tech stack

Python, pypdf, LangChain, Sentence Transformers (all-MiniLM-L6-v2), FAISS, Google Gemini, Streamlit, python-dotenv.

## Project structure

```
domain_rag_chatbot/
|-- app.py              Streamlit interface
|-- rag_pipeline.py     Processing + retrieval + answer generation
|-- document_loader.py  PDF/TXT text extraction, file validation
|-- vector_store.py     Chunking, embeddings, FAISS save/load
|-- prompt.py           Guardrail prompt
|-- requirements.txt
|-- .env.example        Copy to .env and add your key
|-- documents/          Sample PDFs
|-- vector_store/       Saved FAISS index
|-- tests/test_questions.csv   Evaluation sheet (add at least 15 questions)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then put your Gemini API key in .env
streamlit run app.py
```

Get a free Gemini API key from Google AI Studio. The first run downloads the embedding model (~90 MB).

## Usage

1. Upload one or more PDF/TXT files in the sidebar (max 20 MB each).
2. Click **Process Documents**.
3. Ask questions in the chat box. Open **Sources** under an answer to see the document and page.
4. Use **Clear Chat** to reset the conversation or **Clear Documents** to remove the loaded documents.

Optional: index the PDFs in `documents/` from the command line with `python vector_store.py`.

## Safety and limitations

- API keys live in `.env` (git-ignored); never commit them.
- Only PDF/TXT files up to 20 MB are accepted.
- The prompt tells the model to ignore instructions found inside documents.
- Answers can still be wrong: verify high-stakes information in the original document.
- Scanned (image-only) PDFs are not supported unless OCR is added.
- Do not upload confidential documents without permission.

## Testing

Fill in `tests/test_questions.csv` with at least 15 questions: correct, incorrect and unavailable ones.
Record the retrieved source and whether the answer was correct.
