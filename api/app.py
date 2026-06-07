from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ingest import load_and_index
from generator import answer_question
import shutil

app = FastAPI(title="RAG Document Q&A")

class QuestionRequest(BaseModel):
    question: str
    index_path: str = "data/faiss_index"

@app.get("/")
def home():
    return {"message": "RAG Document Q&A API is running"}

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    load_and_index(file_path)
    return {"message": f"{file.filename} uploaded and indexed successfully"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    answer = answer_question(request.question, request.index_path)
    return {"answer": answer}