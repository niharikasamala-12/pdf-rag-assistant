
from retriever import (
    collection,
    vector_search,
    bm25_search,
    reciprocal_rank_fusion,
    rerank_chunks,
)

QUERY = "Explain the different data types available in MySQL with examples."

print("\n========== DATABASE CHECK ==========")
print("Total indexed chunks:", collection.count())

# Check whether the expected MySQL pages exist in ChromaDB.
results = collection.get(
    include=["documents", "metadatas"]
)

print("\n========== PAGE 6 AND 7 CHECK ==========")

found = False

for chunk_id, document, metadata in zip(
    results["ids"],
    results["documents"],
    results["metadatas"]
):
    metadata = metadata or {}

    if (
        metadata.get("source") == "MySQL Handbook.pdf"
        and metadata.get("page") in [6, 7]
    ):
        found = True
        print("\nChunk ID:", chunk_id)
        print("Page:", metadata.get("page"))
        print("Text:", document[:500])

if not found:
    print("No chunks found for MySQL Handbook.pdf pages 6 and 7.")

# Retrieve more candidates for diagnosis only.
print("\n========== VECTOR SEARCH TOP 50 ==========")

vector_chunks = vector_search(QUERY, candidate_k=50)

for rank, chunk in enumerate(vector_chunks, 1):
    print(
        rank,
        chunk["source"],
        chunk["page"],
        chunk["id"]
    )

print("\n========== BM25 SEARCH TOP 50 ==========")

bm25_chunks = bm25_search(QUERY, candidate_k=50)

for rank, chunk in enumerate(bm25_chunks, 1):
    print(
        rank,
        chunk["source"],
        chunk["page"],
        chunk["id"],
        "BM25:", round(chunk["bm25_score"], 4)
    )

# Combine the expanded candidate lists.
print("\n========== RRF TOP 50 ==========")

fused = reciprocal_rank_fusion(
    vector_chunks,
    bm25_chunks
)

for rank, chunk in enumerate(fused[:50], 1):
    print(
        rank,
        chunk["source"],
        chunk["page"],
        chunk["id"],
        "RRF:", round(chunk["rrf_score"], 5)
    )

# Test reranking on the expanded candidate pool.
print("\n========== CROSS-ENCODER TOP 20 ==========")

reranked = rerank_chunks(QUERY, fused, top_k=20)

for rank, chunk in enumerate(reranked, 1):
    print(
        rank,
        chunk["source"],
        chunk["page"],
        chunk["id"],
        "Score:", round(chunk["rerank_score"], 4)
    )

print("\n========== DIAGNOSTIC COMPLETE ==========")
