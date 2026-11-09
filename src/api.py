
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingest import ingest_pdf
from generator import generate_answer

app = FastAPI()

# Allow requests from the local frontend
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

active_pdf = None


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "Dynamic PDF RAG API is running!"
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global active_pdf

    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    pdf_path = UPLOAD_DIR / "uploaded_document.pdf"

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF is empty."
            )

        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(content)

        chunk_count = ingest_pdf(pdf_path)
        active_pdf = file.filename

        return {
            "message": "PDF uploaded and processed successfully!",
            "filename": active_pdf,
            "chunks": chunk_count
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not active_pdf:
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF before asking questions."
        )

    try:
        result = generate_answer(request.question)

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
