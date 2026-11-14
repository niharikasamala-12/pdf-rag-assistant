import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"


def extract_context(chunks):
    context_parts = []

    for chunk in chunks:
        if isinstance(chunk, dict):
            content = (
                chunk.get("text")
                or chunk.get("content")
                or chunk.get("page_content")
                or chunk.get("document")
            )

            # Support Chroma-style nested metadata if needed.
            if not content and isinstance(chunk.get("metadata"), dict):
                content = chunk["metadata"].get("text")

            if content:
                context_parts.append(str(content))

    return "\n\n".join(context_parts)


def evaluate_answer(question, answer, chunks):
    context = extract_context(chunks)

    if not context:
        raise ValueError(
            "Could not extract text from retrieved chunks. "
            "Check the chunk structure returned by retriever.py."
        )

    prompt = f"""
You are an evaluator for a Retrieval-Augmented Generation system.

Evaluate the answer using ONLY the supplied context.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Give three scores between 0.0 and 1.0:

1. faithfulness:
How much of the answer is supported by the retrieved context?

2. answer_relevancy:
How directly does the answer address the question?

3. context_relevance:
How useful is the retrieved context for answering the question?

Return ONLY valid JSON in this exact format:
{{
    "faithfulness": 0.0,
    "answer_relevancy": 0.0,
    "context_relevance": 0.0,
    "reason": "Short explanation"
}}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        },
        timeout=300
    )

    response.raise_for_status()

    result = response.json()["response"]
    scores = json.loads(result)

    # Validate and normalize the scores.
    for metric in [
        "faithfulness",
        "answer_relevancy",
        "context_relevance"
    ]:
        value = float(scores[metric])

        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid {metric} score: {value}")

        scores[metric] = value

    return scores
