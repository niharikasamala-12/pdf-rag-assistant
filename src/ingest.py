
from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "chroma_db"

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_or_create_collection(name="documents")


def create_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def ingest_pdf(pdf_path, original_filename=None):

    pdf_path = Path(pdf_path)

    reader = PdfReader(str(pdf_path))

    document_id = str(uuid4())

    filename = original_filename or pdf_path.name

    documents = []
    metadatas = []
    ids = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        chunks = create_chunks(text)

        for chunk_number, chunk in enumerate(chunks):

            documents.append(chunk)

            metadatas.append({
                "document_id": document_id,
                "source": filename,
                "page": page_number
            })

            ids.append(
                f"{document_id}_{page_number}_{chunk_number}"
            )

    if not documents:
        raise ValueError(
            "No extractable text found in this PDF."
        )

    embeddings = model.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return {
        "document_id": document_id,
        "filename": filename,
        "chunks": len(documents)
    }
