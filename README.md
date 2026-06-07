# RAG Document Q&A System

Upload any PDF and ask questions in natural language. Get answers with page citations powered by Groq LLM.

---

## How It Works

```
PDF Upload → Text Extraction → Chunking → TF-IDF Indexing
                                                ↓
User Question → TF-IDF Retrieval → Context Building → Groq LLM → Answer with Citations
```

---

## Features

- Upload any PDF via REST API
- Automatic text chunking and TF-IDF indexing
- Retrieval of relevant chunks using cosine similarity
- Groq LLM (Llama 3.1) generates answers with page citations
- FastAPI with auto-generated Swagger docs

---

## Tech Stack

| Component | Tool |
|---|---|
| PDF Parsing | LangChain PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| Retrieval | TF-IDF + Cosine Similarity |
| LLM | Groq (Llama 3.1 8b) |
| API | FastAPI + Uvicorn |

---

## Quick Start

```bash
git clone https://github.com/Jyoshika3012/rag-document-qa.git
cd rag-document-qa
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Add your Groq API key to `.env`:
```
GROQ_API_KEY=your_key_here
```

Run the API:
```bash
uvicorn api.app:app --reload
```

Open `http://127.0.0.1:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Upload and index a PDF |
| POST | `/ask` | Ask a question about the document |
| GET | `/` | Health check |

---

## Example

**Request:**
```json
POST /ask
{
  "question": "What is the main objective of this project?"
}
```

**Response:**
```json
{
  "answer": "The main goal is to reduce manual effort and provide a contactless solution. (This comes from Page 1)"
}
```

---

## Key Engineering Decisions

- **TF-IDF over vector embeddings** — no GPU needed, no model download required, fast and effective for document Q&A
- **Direct Groq SDK** over LangChain wrapper — avoids timeout issues, more reliable for production use
- **Chunking with overlap** — 500 char chunks with 50 char overlap prevents context loss at boundaries