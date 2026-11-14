
import json
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "chunk_candidates.json"
OUTPUT_FILE = BASE_DIR / "data" / "chunk_candidates_labeled.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"


def judge_relevance(question, chunk_text):
    prompt = f"""
You are evaluating a RAG retrieval system.

Determine whether the following chunk contains information
that is relevant to answering the question.

Question:
{question}

Chunk:
{chunk_text}

Return only JSON:
{{"relevant": true}}

Use true if the chunk contains useful information for answering
the question. Otherwise use false.
Do not mark a chunk relevant merely because it discusses
the same broad topic.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0}
        },
        timeout=300
    )

    response.raise_for_status()

    result = json.loads(response.json()["response"])

    if not isinstance(result.get("relevant"), bool):
        raise ValueError("Invalid relevance judgment")

    return result["relevant"]


with open(INPUT_FILE, "r", encoding="utf-8") as file:
    questions = json.load(file)

total = 0
labeled = 0

for question_item in questions:
    print(
        f"\nQuestion {question_item['question_id']}: "
        f"{question_item['question']}"
    )

    for candidate in question_item["candidates"]:
        total += 1

        try:
            candidate["relevant"] = judge_relevance(
                question_item["question"],
                candidate["text"]
            )

            labeled += 1

            print(
                f"  Rank {candidate['rank']}: "
                f"{candidate['relevant']}"
            )

        except Exception as error:
            candidate["relevant"] = None
            print(f"  Rank {candidate['rank']}: ERROR - {error}")

    # Save progress after each question
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(questions, file, indent=4, ensure_ascii=False)

print("\n========== LABELING SUMMARY ==========")
print("Total candidates:", total)
print("Successfully labeled:", labeled)
print("Unlabeled:", total - labeled)
print("Saved to:", OUTPUT_FILE)
