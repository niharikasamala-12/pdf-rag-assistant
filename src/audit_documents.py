
from pathlib import Path
from collections import Counter

import chromadb

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "chroma_db"

client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection(name="documents")

results = collection.get(include=["metadatas"])

document_counts = Counter()
document_sources = {}

for metadata in results["metadatas"]:
    metadata = metadata or {}

    document_id = metadata.get("document_id", "NO_DOCUMENT_ID")
    source = metadata.get("source", "UNKNOWN")

    document_counts[document_id] += 1
    document_sources[document_id] = source

print("\n========== DOCUMENT METADATA AUDIT ==========")
print("Total chunks:", len(results["ids"]))
print()

for document_id, count in document_counts.items():
    print("Document ID:", document_id)
    print("Source:", document_sources[document_id])
    print("Chunks:", count)
    print("-" * 50)
 