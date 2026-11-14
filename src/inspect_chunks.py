
import json
from pathlib import Path

from retriever import collection

BASE_DIR = Path(__file__).resolve().parent.parent

documents = collection.get(
    include=["documents", "metadatas"]
)

chunks = []

for chunk_id, text, metadata in zip(
    documents["ids"],
    documents["documents"],
    documents["metadatas"]
):
    metadata = metadata or {}

    chunks.append({
        "chunk_id": chunk_id,
        "source": metadata.get("source", "Unknown"),
        "page": metadata.get("page", "Unknown"),
        "text": text
    })

output_file = BASE_DIR / "data" / "all_chunks.json"

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(chunks, file, indent=4, ensure_ascii=False)

print("Total chunks:", len(chunks))
print("Saved to:", output_file)
