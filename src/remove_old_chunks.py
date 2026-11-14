
from pathlib import Path
import chromadb

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "chroma_db"

client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection(name="documents")

results = collection.get(include=["metadatas"])

old_ids = []

for chunk_id, metadata in zip(results["ids"], results["metadatas"]):
    metadata = metadata or {}

    if (
        metadata.get("source") == "uploaded_document.pdf"
        and "document_id" not in metadata
    ):
        old_ids.append(chunk_id)

print("Old chunks found:", len(old_ids))

if len(old_ids) != 38:
    print("Unexpected chunk count. Nothing deleted.")
else:
    confirm = input("Type DELETE to remove these old chunks: ")

    if confirm == "DELETE":
        collection.delete(ids=old_ids)
        print("Old duplicate chunks deleted successfully!")
    else:
        print("Deletion cancelled.")

print("Remaining chunks:", collection.count())
