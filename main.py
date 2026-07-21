import os
import shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from ingest import extract_text_from_pdf, chunk_text
from vector_store import VectorStore
from rag_pipeline import generate_answer

app = FastAPI(title="RAG Document Q&A API")

# Keep the vector store in memory for the running session.
# (Simple approach for a learning project — real production systems
# would persist this to disk/a database instead of memory.)
vector_store = VectorStore()
is_ready = False  # tracks whether a document has been indexed yet


class Question(BaseModel):
    question: str


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global is_ready

    # Save the uploaded file to disk
    save_path = f"data/uploads/{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Process it: extract → chunk → embed → index
    text = extract_text_from_pdf(save_path)
    chunks = chunk_text(text)
    vector_store.build_index(chunks)
    is_ready = True

    return {
        "filename": file.filename,
        "chunks_created": len(chunks),
        "status": "Document indexed successfully"
    }


@app.post("/ask")
async def ask_question(payload: Question):
    if not is_ready:
        return {"error": "No document has been uploaded yet. Upload one first via /upload."}

    retrieved_chunks = vector_store.search(payload.question, top_k=3)
    answer = generate_answer(payload.question, retrieved_chunks)

    return {
        "question": payload.question,
        "answer": answer,
        "sources_used": len(retrieved_chunks)
    }


@app.get("/")
async def root():
    return {"message": "RAG Document Q&A API is running. Visit /docs for interactive testing."}