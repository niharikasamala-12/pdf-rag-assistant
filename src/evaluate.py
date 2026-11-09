import csv
import json
import time
from pathlib import Path
from retriever import retrieve
from generator import generate_answer

# Find evaluation.json from the project root
BASE_DIR = Path(__file__).resolve().parent.parent
EVALUATION_FILE = BASE_DIR / "data" / "evaluation.json"


def normalize_source(source):
    """Compare source filenames consistently."""
    return Path(str(source)).name.lower()


def evaluate():
    if not EVALUATION_FILE.exists():
        print(f"Evaluation file not found: {EVALUATION_FILE}")
        return

    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    # Start with two questions to test the evaluator.
    # Remove this line to evaluate all 10 questions.
    #test_cases = test_cases[:2]
    test_cases = test_cases
    retrieval_hits = 0
    retrieval_times = []
    generation_times = []

    print("\n========== RAG EVALUATION ==========\n")
    print(f"Questions to evaluate: {len(test_cases)}")

    quality_scores = []
    results = []
    for index, test in enumerate(test_cases, start=1):
        question = test["question"]
        expected_source = test.get("expected_source", "")

        print(f"\n{'=' * 60}")
        print(f"Question {index}: {question}")

        # 1. Evaluate retrieval
        start = time.perf_counter()
        retrieved_chunks = retrieve(question, top_k=3)
        retrieval_time = time.perf_counter() - start
        retrieval_times.append(retrieval_time)

        retrieved_sources = {
            normalize_source(chunk.get("source", ""))
            for chunk in retrieved_chunks
        }

        hit = (
            normalize_source(expected_source) in retrieved_sources
            if expected_source
            else False
        )

        if hit:
            retrieval_hits += 1

        print(f"\nExpected source: {expected_source}")
        print(f"Retrieved sources: {sorted(retrieved_sources)}")
        print(f"Retrieval hit: {'YES' if hit else 'NO'}")
        print(f"Retrieval time: {retrieval_time:.2f} seconds")

        # 2. Generate the answer
        start = time.perf_counter()
        result = generate_answer(question,retrieved_chunks=retrieved_chunks)
        generation_time = time.perf_counter() - start
        generation_times.append(generation_time)

        # Support either a dictionary or a plain-text answer.
        if isinstance(result, dict):
            answer = result.get("answer", str(result))
        else:
            answer = str(result)

        print("\nExpected answer:")
        print(test.get("expected_answer", "Not provided"))

        print("\nGenerated answer:")
        print(answer)

        # Manually evaluate the generated answer
        while True:
            try:
                score = int(input(
                    "\nRate answer quality (0 = Poor, 1 = Partial, 2 = Good): "
                ))

                if score in [0, 1, 2]:
                    quality_scores.append(score)
                    break
                print("Enter only 0, 1, or 2.")

            except ValueError:
                print("Please enter a valid number.")
        

        print(f"\nAnswer generation time: {generation_time:.2f} seconds")
        results.append({
            "question": question,
            "expected_answer": test.get("expected_answer", ""),
            "generated_answer": answer,
            "retrieved_sources": ", ".join(sorted(retrieved_sources)),
            "retrieval_hit": hit,
            "retrieval_time_seconds": round(retrieval_time, 2),
            "generation_time_seconds": round(generation_time, 2),
        })
    output_file = BASE_DIR / "data" / "evaluation_results.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nEvaluation report saved to: {output_file}")

    # 3. Print summary metrics
    total = len(test_cases)

    if total == 0:
        print("No evaluation questions found.")
        return

    hit_rate = (retrieval_hits / total) * 100
    avg_retrieval = sum(retrieval_times) / len(retrieval_times)
    avg_generation = sum(generation_times) / len(generation_times)

    print(f"\n{'=' * 60}")
    print("========== EVALUATION SUMMARY ==========")
    print(f"Total questions: {total}")
    print(f"Retrieval hits: {retrieval_hits}/{total}")
    print(f"Retrieval hit rate: {hit_rate:.2f}%")
    print(f"Average retrieval time: {avg_retrieval:.2f} seconds")
    print(f"Average answer generation time: {avg_generation:.2f} seconds")
    print("\nReview each generated answer against its expected answer.")
    # Answer quality metrics
    
    if quality_scores:
        total_score = sum(quality_scores)
        max_score = len(quality_scores) * 2
        quality_percentage = (total_score / max_score) * 100
        good_answers = quality_scores.count(2)
        partial_answers = quality_scores.count(1)
        poor_answers = quality_scores.count(0)

        print("\n========== ANSWER QUALITY ==========")
        print(f"Good answers: {good_answers}")
        print(f"Partial answers: {partial_answers}")
        print(f"Poor answers: {poor_answers}")
        print(f"Answer quality score: {quality_percentage:.2f}%")
        print("========================================")


if __name__ == "__main__":
    evaluate()
