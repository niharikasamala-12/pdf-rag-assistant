
import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "chunk_candidates_labeled.json"
CSV_FILE = BASE_DIR / "data" / "chunk_level_results.csv"
SUMMARY_FILE = BASE_DIR / "data" / "chunk_level_summary.json"

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    questions = json.load(file)

precision_scores = []
recall_scores = []
mrr_scores = []

results = []

for item in questions:
    candidates = item["candidates"]

    # Stop if any relevance label is missing
    if any(c["relevant"] is None for c in candidates):
        raise ValueError(
            f"Question {item['question_id']} has unlabeled chunks. "
            "Please label all candidates first."
        )

    # Candidates are already ordered by retrieval rank
    top_3 = candidates[:3]

    relevant_in_top_3 = sum(
        1 for chunk in top_3 if chunk["relevant"] is True
    )

    total_relevant = sum(
        1 for chunk in candidates if chunk["relevant"] is True
    )

    # Precision@3
    precision = relevant_in_top_3 / len(top_3) if top_3 else 0

    # Recall@3 within the 10 candidate chunks
    recall = (
        relevant_in_top_3 / total_relevant
        if total_relevant > 0 else 0
    )

    # Reciprocal Rank
    reciprocal_rank = 0

    for rank, chunk in enumerate(candidates, start=1):
        if chunk["relevant"] is True:
            reciprocal_rank = 1 / rank
            break

    precision_scores.append(precision)
    recall_scores.append(recall)
    mrr_scores.append(reciprocal_rank)

    results.append({
        "question_id": item["question_id"],
        "question": item["question"],
        "relevant_chunks_in_top_3": relevant_in_top_3,
        "total_relevant_in_candidates": total_relevant,
        "precision_at_3": round(precision, 4),
        "recall_at_3": round(recall, 4),
        "reciprocal_rank": round(reciprocal_rank, 4)
    })

# Overall averages
total_questions = len(results)

summary = {
    "total_questions": total_questions,
    "total_candidates": sum(
        len(item["candidates"]) for item in questions
    ),
    "mean_precision_at_3": round(
        sum(precision_scores) / total_questions, 4
    ),
    "mean_recall_at_3": round(
        sum(recall_scores) / total_questions, 4
    ),
    "mean_reciprocal_rank": round(
        sum(mrr_scores) / total_questions, 4
    )
}

# Save detailed results
with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)

# Save summary
with open(SUMMARY_FILE, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

print("\n========== CHUNK-LEVEL EVALUATION ==========")

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nDetailed results:", CSV_FILE)
print("Summary:", SUMMARY_FILE)
# Show weakest questions by Precision@3
print("\n========== WEAKEST QUESTIONS ==========")

weakest_questions = sorted(
    results,
    key=lambda item: (
        item["precision_at_3"],
        item["reciprocal_rank"]
    )
)

for item in weakest_questions[:10]:
    print(f"\nQuestion ID: {item['question_id']}")
    print(f"Question: {item['question']}")
    print(f"Relevant chunks in Top 3: {item['relevant_chunks_in_top_3']}/3")
    print(f"Precision@3: {item['precision_at_3']}")
    print(f"Recall@3: {item['recall_at_3']}")
    print(f"MRR: {item['reciprocal_rank']}")