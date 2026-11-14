
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "all_chunks.json"

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

# Count chunks by PDF filename
source_counts = Counter(chunk["source"] for chunk in chunks)

# Group identical text across chunk IDs and sources
text_groups = defaultdict(list)

for chunk in chunks:
    normalized_text = " ".join(chunk["text"].lower().split())

    text_hash = hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()

    text_groups[text_hash].append({
        "chunk_id": chunk["chunk_id"],
        "source": chunk["source"],
        "page": chunk["page"]
    })

duplicates = {
    key: value
    for key, value in text_groups.items()
    if len(value) > 1
}

print("\n========== CHUNK AUDIT ==========")

print("\nTotal chunks:", len(chunks))

print("\nChunks by source:")
for source, count in source_counts.items():
    print(f"{source}: {count}")

print("\nUnique chunk texts:", len(text_groups))
print("Duplicate text groups:", len(duplicates))

print("\nDuplicate examples:")

for index, group in enumerate(duplicates.values(), start=1):
    print(f"\nDuplicate group {index}:")
    for item in group:
        print(
            f"  Source: {item['source']} | "
            f"Page: {item['page']} | "
            f"ID: {item['chunk_id']}"
        )

    if index == 10:
        break
