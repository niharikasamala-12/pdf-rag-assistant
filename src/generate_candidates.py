
import json
from pathlib import Path

from retriever import retrieve

BASE_DIR = Path(__file__).resolve().parent.parent

EVALUATION_FILE = BASE_DIR / "data" / "evaluation.json"
OUTPUT_FILE = BASE_DIR / "data" / "chunk_candidates.json"

with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
    questions = json.load(file)

all_candidates = []

for index, item in enumerate(questions, start=1):
    question = item["question"]

    print(f"\nProcessing question {index}/{len(questions)}")

    chunks = retrieve(question, top_k=10)

    candidate = {
        "question_id": index,
        "question": question,
        "expected_source": item["expected_source"],
        "candidates": []
    }

    for rank, chunk in enumerate(chunks, start=1):
        candidate["candidates"].append({
            "rank": rank,
            "chunk_id": chunk["id"],
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"],
            "relevant": None
        })

    all_candidates.append(candidate)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(all_candidates, file, indent=4, ensure_ascii=False)

print("\nCandidate generation completed!")
print("Questions processed:", len(all_candidates))
print("Output:", OUTPUT_FILE)
