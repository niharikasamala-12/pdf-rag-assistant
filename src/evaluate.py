from answer_evaluator import evaluate_answer
import csv
import json
import time
from pathlib import Path

from retriever import retrieve
from generator import generate_answer


BASE_DIR = Path(__file__).resolve().parent.parent
EVALUATION_FILE = BASE_DIR / "data" / "evaluation.json"

TOP_K = 3


def normalize_source(source):
    """Normalize filenames for consistent comparison."""
    return Path(str(source)).name.lower()


def get_quality_score():
    """Get manual answer-quality rating."""
    while True:
        try:
            score = int(input(
                "\nRate answer quality (0 = Poor, 1 = Partial, 2 = Good): "
            ))

            if score in [0, 1, 2]:
                return score

            print("Enter only 0, 1, or 2.")

        except ValueError:
            print("Please enter a valid number.")


def evaluate():

    if not EVALUATION_FILE.exists():
        print(f"Evaluation file not found: {EVALUATION_FILE}")
        return

    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    if not test_cases:
        print("No evaluation questions found.")
        return

    results = []
    quality_scores = []
    judge_scores = []

    retrieval_hits = 0
    precision_scores = []
    reciprocal_ranks = []

    retrieval_times = []
    generation_times = []
    total_times = []

    print("\n========== ADVANCED RAG EVALUATION ==========\n")
    print(f"Questions to evaluate: {len(test_cases)}")
    print(f"Top K: {TOP_K}")

    for index, test in enumerate(test_cases, start=1):

        question = test["question"]
        expected_source = test.get("expected_source", "")

        print("\n" + "=" * 60)
        print(f"Question {index}: {question}")

        total_start = time.perf_counter()

        # 1. Retrieval evaluation
        retrieval_start = time.perf_counter()

        retrieved_chunks = retrieve(question, top_k=TOP_K)

        retrieval_time = time.perf_counter() - retrieval_start
        retrieval_times.append(retrieval_time)

        expected = normalize_source(expected_source) if expected_source else ""

        retrieved_sources = [
            normalize_source(chunk.get("source", ""))
            for chunk in retrieved_chunks
        ]

        # Identify relevant retrieved chunks using expected source.
        relevant_ranks = [
            rank
            for rank, source in enumerate(retrieved_sources, start=1)
            if expected and source == expected
        ]

        relevant_count = len(relevant_ranks)

        # Source-level hit
        hit = relevant_count > 0

        if hit:
            retrieval_hits += 1

        # Precision@K proxy
        precision_at_k = relevant_count / TOP_K
        precision_scores.append(precision_at_k)

        # Mean Reciprocal Rank
        reciprocal_rank = (
            1 / relevant_ranks[0]
            if relevant_ranks
            else 0
        )
        reciprocal_ranks.append(reciprocal_rank)

        print(f"\nExpected source: {expected_source}")
        print(f"Retrieved sources: {retrieved_sources}")
        print(f"Retrieval hit: {'YES' if hit else 'NO'}")
        print(f"Precision@{TOP_K}: {precision_at_k:.2f}")
        print(f"Reciprocal Rank: {reciprocal_rank:.2f}")
        print(f"Retrieval time: {retrieval_time:.2f} seconds")

        # 2. Answer generation
        generation_start = time.perf_counter()

        result = generate_answer(
            question,
            retrieved_chunks=retrieved_chunks
        )

        generation_time = time.perf_counter() - generation_start
        generation_times.append(generation_time)

        if isinstance(result, dict):
            answer = result.get("answer", str(result))
        else:
            answer = str(result)

        print("\nExpected answer:")
        print(test.get("expected_answer", "Not provided"))

        print("\nGenerated answer:")
        print(answer)
        # 3. Automated answer evaluation using Ollama
        print("\nRunning automated answer evaluation...")
        try:
            judge_result = evaluate_answer(
            question,
            answer,
            retrieved_chunks
            )

            faithfulness = judge_result["faithfulness"]
            answer_relevancy = judge_result["answer_relevancy"]
            context_relevance = judge_result["context_relevance"]

            judge_scores.append(judge_result)

            print("\nAutomated Evaluation:")
            print(f"Faithfulness: {faithfulness:.2f}")
            print(f"Answer Relevancy: {answer_relevancy:.2f}")
            print(f"Context Relevance: {context_relevance:.2f}")
            print(f"Reason: {judge_result.get('reason', '')}")

        except Exception as error:
            print(f"Automated evaluation failed: {error}")

            faithfulness = None
            answer_relevancy = None
            context_relevance = None

        # 3. Manual answer-quality evaluation
        total_time = retrieval_time + generation_time
        total_times.append(total_time)

        score = get_quality_score()
        quality_scores.append(score)


        print(f"\nGeneration time: {generation_time:.2f} seconds")
        print(f"Total response time: {total_time:.2f} seconds")

        results.append({
            "question": question,
            "expected_source": expected_source,
            "retrieved_sources": ", ".join(retrieved_sources),
            "retrieval_hit": hit,
            "precision_at_3": round(precision_at_k, 3),
            "reciprocal_rank": round(reciprocal_rank, 3),
            "expected_answer": test.get("expected_answer", ""),
            "generated_answer": answer,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_relevance": context_relevance,
            "judge_reason": (
                judge_result.get("reason", "")
                if faithfulness is not None
                else ""
            ),
            "manual_quality_score": score,
            "retrieval_time_seconds": round(retrieval_time, 3),
            "generation_time_seconds": round(generation_time, 3),
            "total_time_seconds": round(total_time, 3)
        })

    # 4. Calculate overall metrics
    total = len(test_cases)

    hit_rate = (retrieval_hits / total) * 100
    mean_precision = sum(precision_scores) / total
    mean_reciprocal_rank = sum(reciprocal_ranks) / total

    avg_retrieval = sum(retrieval_times) / total
    avg_generation = sum(generation_times) / total
    avg_total = sum(total_times) / total
    avg_faithfulness = (
        sum(item["faithfulness"] for item in judge_scores) / len(judge_scores)
        if judge_scores else 0
    )
    avg_answer_relevancy = (
        sum(item["answer_relevancy"] for item in judge_scores) / len(judge_scores)
        if judge_scores else 0
    )

    avg_context_relevance = (
        sum(item["context_relevance"] for item in judge_scores) / len(judge_scores)
        if judge_scores else 0
    )

    total_quality = sum(quality_scores)
    max_quality = total * 2
    quality_percentage = (total_quality / max_quality) * 100

    good_answers = quality_scores.count(2)
    partial_answers = quality_scores.count(1)
    poor_answers = quality_scores.count(0)

    # 5. Save detailed CSV report
    output_file = BASE_DIR / "data" / "evaluation_results.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    # 6. Save summary JSON report
    summary = {
        "total_questions": total,
        "retrieval_hit_rate_percent": round(hit_rate, 2),
        "mean_precision_at_3": round(mean_precision, 3),
        "mean_reciprocal_rank": round(mean_reciprocal_rank, 3),
        "average_retrieval_time_seconds": round(avg_retrieval, 3),
        "average_generation_time_seconds": round(avg_generation, 3),
        "average_total_time_seconds": round(avg_total, 3),
        "answer_quality_percent": round(quality_percentage, 2),
        "good_answers": good_answers,
        "partial_answers": partial_answers,
        "poor_answers": poor_answers,
        "automated_evaluations_completed": len(judge_scores),
        "average_faithfulness": round(avg_faithfulness, 3),
        "average_answer_relevancy": round(avg_answer_relevancy, 3),
        "average_context_relevance": round(avg_context_relevance, 3)
    }

    summary_file = BASE_DIR / "data" / "evaluation_summary.json"

    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    # 7. Print evaluation report
    print("\n" + "=" * 60)
    print("========== RAG EVALUATION SUMMARY ==========")

    print(f"Total questions: {total}")
    print(f"Retrieval hits: {retrieval_hits}/{total}")
    print(f"Retrieval hit rate: {hit_rate:.2f}%")

    print(f"\nMean Precision@{TOP_K}: {mean_precision:.3f}")
    print(f"Mean Reciprocal Rank (MRR): {mean_reciprocal_rank:.3f}")

    print("\n========== PERFORMANCE ==========")
    print(f"Average retrieval time: {avg_retrieval:.2f} seconds")
    print(f"Average generation time: {avg_generation:.2f} seconds")
    print(f"Average total response time: {avg_total:.2f} seconds")


    print("\n========== AUTOMATED ANSWER EVALUATION ==========")
    print(f"Evaluations completed: {len(judge_scores)}/{total}")
    print(f"Average faithfulness: {avg_faithfulness:.3f}")
    print(f"Average answer relevancy: {avg_answer_relevancy:.3f}")
    print(f"Average context relevance: {avg_context_relevance:.3f}")
    
    print("\n========== ANSWER QUALITY ==========")
    print(f"Good answers: {good_answers}")
    print(f"Partial answers: {partial_answers}")
    print(f"Poor answers: {poor_answers}")
    print(f"Answer quality: {quality_percentage:.2f}%")

    print("\n========== REPORTS SAVED ==========")
    print(f"Detailed CSV: {output_file}")
    print(f"Summary JSON: {summary_file}")

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    evaluate()