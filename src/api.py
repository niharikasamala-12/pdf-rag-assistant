import csv
import json
from pathlib import Path
import hashlib

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingest import ingest_pdf, collection
from retriever import retrieve
from generator import generate_answer


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:5501",
        "http://127.0.0.1:5501"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class QuestionRequest(BaseModel):
    question: str
    document_id: str | None = None


@app.get("/")
def home():
    return {
        "message": "Multi-PDF RAG API is running!"
    }


# --------------------------------
# UPLOAD PDF
# --------------------------------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF is empty."
        )

    # Create a unique filename on disk
    unique_name = f"{hashlib.sha256(content).hexdigest()}.pdf"
    pdf_path = UPLOAD_DIR / unique_name

    try:
        # Reuse an existing file if the same PDF was uploaded
        if not pdf_path.exists():
            pdf_path.write_bytes(content)

        result = ingest_pdf(
            pdf_path,
            original_filename=Path(file.filename).name
        )

        return {
            "message": "PDF uploaded and processed successfully!",
            **result
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# --------------------------------
# LIST DOCUMENTS
# --------------------------------

@app.get("/documents")
def list_documents():

    results = collection.get(
        include=["metadatas"]
    )

    documents = {}

    for metadata in results["metadatas"]:

        if not metadata:
            continue

        document_id = metadata.get("document_id")

        if not document_id:
            continue

        if document_id not in documents:

            documents[document_id] = {
                "document_id": document_id,
                "filename": metadata.get("source", "Unknown"),
                "chunks": 0
            }

        documents[document_id]["chunks"] += 1

    return {
        "documents": list(documents.values()),
        "total_documents": len(documents)
    }


# --------------------------------
# DELETE DOCUMENT
# --------------------------------

@app.delete("/documents/{document_id}")
def delete_document(document_id: str):

    results = collection.get(
        where={"document_id": document_id},
        include=["metadatas"]
    )

    if not results["ids"]:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    filename = (
        results["metadatas"][0].get("source", "Unknown")
        if results["metadatas"] else "Unknown"
    )

    collection.delete(ids=results["ids"])

    return {
        "message": "Document deleted successfully!",
        "document_id": document_id,
        "filename": filename,
        "deleted_chunks": len(results["ids"])
    }


# --------------------------------
# ASK QUESTION
# --------------------------------

@app.post("/ask")
def ask_question(request: QuestionRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if request.document_id:

        selected = collection.get(
            where={"document_id": request.document_id},
            include=["metadatas"]
        )

        if not selected["ids"]:
            raise HTTPException(
                status_code=404,
                detail="Selected document not found."
            )

    elif collection.count() == 0:

        raise HTTPException(
            status_code=400,
            detail="Upload a PDF before asking questions."
        )

    try:

        chunks = retrieve(
            request.question,
            top_k=3,
            document_id=request.document_id
        )

        result = generate_answer(
            request.question,
            retrieved_chunks=chunks
        )

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
@app.get("/evaluation")
def get_evaluation_report():

    summary_file = BASE_DIR / "data" / "evaluation_summary.json"
    results_file = BASE_DIR / "data" / "evaluation_results.csv"

    if not summary_file.exists() or not results_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation reports not found. Run evaluate.py first."
        )

    with open(summary_file, "r", encoding="utf-8") as file:
        summary = json.load(file)

    with open(results_file, "r", encoding="utf-8") as file:
        results = list(csv.DictReader(file))

    return {
        "summary": summary,
        "results": results
    }