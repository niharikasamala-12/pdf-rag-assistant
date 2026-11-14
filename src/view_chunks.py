
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_FILE = BASE_DIR / "data" / "all_chunks.json"

with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

for chunk in chunks:
    print("\n" + "=" * 80)
    print("CHUNK ID:", chunk["chunk_id"])
    print("SOURCE:", chunk["source"])
    print("PAGE:", chunk["page"])
    print("TEXT:")
    print(chunk["text"][:700])
