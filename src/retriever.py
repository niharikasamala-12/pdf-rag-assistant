
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "chroma_db"

# Load the same embedding model used during ingestion
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Connect to the existing vector database
client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_or_create_collection(name="documents")


def retrieve(query, top_k=5):
    # Convert the user's question into an embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # Find the most relevant stored chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved_chunks = []

    for i, document in enumerate(results["documents"][0]):
        retrieved_chunks.append({
            "text": document,
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "distance": results["distances"][0][i],
        })

    return retrieved_chunks


if __name__ == "__main__":
    question = input("Ask a question about your PDF: ")

    chunks = retrieve(question)

    if not chunks:
        print("No chunks found. Check whether your PDF was indexed.")
    else:
        for i, chunk in enumerate(chunks, start=1):
            print(f"\n--- Result {i} ---")
            print(f"Source: {chunk['source']}, Page: {chunk['page']}")
            print(f"Distance: {chunk['distance']:.4f}")
            print(chunk["text"])
