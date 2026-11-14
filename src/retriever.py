from pathlib import Path
import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CANDIDATE_K = 10
RRF_K = 60


# --------------------------------------------------
# LOAD MODELS AND DATABASE
# --------------------------------------------------

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

reranker = CrossEncoder(RERANKER_MODEL)

client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(name="documents")


# --------------------------------------------------
# TEXT PREPROCESSING FOR BM25
# --------------------------------------------------

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# --------------------------------------------------
# VECTOR SEARCH
# --------------------------------------------------

def vector_search(query, candidate_k, document_id=None):

    count = collection.count()

    if count == 0:
        return []

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()
    query_filter = {}
    
    if document_id:
        query_filter["document_id"] = document_id
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(candidate_k, count),
        where=query_filter if query_filter else None,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []

    for i, document in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][i] or {}

        chunks.append({
            "id": results["ids"][0][i],
            "text": document,
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page", "Unknown"),
            "distance": results["distances"][0][i],
        })

    return chunks


# --------------------------------------------------
# BM25 KEYWORD SEARCH
# --------------------------------------------------

def bm25_search(query, candidate_k, document_id=None):

    query_filter = {}
    if document_id:
        query_filter["document_id"] = document_id
    results = collection.get(
        where=query_filter if query_filter else None,
        include=["documents", "metadatas"]
    )

    documents = results["documents"]
    metadatas = results["metadatas"]
    ids = results["ids"]

    if not documents:
        return []

    tokenized_documents = [
        tokenize(document) for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    chunks = []

    for index in ranked_indices[:candidate_k]:

        if scores[index] <= 0:
            continue

        metadata = metadatas[index] or {}

        chunks.append({
            "id": ids[index],
            "text": documents[index],
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page", "Unknown"),
            "bm25_score": float(scores[index]),
        })

    return chunks


# --------------------------------------------------
# RECIPROCAL RANK FUSION
# --------------------------------------------------

def reciprocal_rank_fusion(vector_chunks, bm25_chunks):

    fused_scores = {}
    chunk_lookup = {}

    for ranked_chunks in [vector_chunks, bm25_chunks]:

        for rank, chunk in enumerate(ranked_chunks, start=1):

            chunk_id = chunk["id"]

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0)
                + 1 / (RRF_K + rank)
            )

            chunk_lookup[chunk_id] = chunk

    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )

    fused_chunks = []

    for chunk_id in ranked_ids:

        chunk = chunk_lookup[chunk_id].copy()

        chunk["rrf_score"] = fused_scores[chunk_id]

        fused_chunks.append(chunk)

    return fused_chunks


# --------------------------------------------------
# CROSS-ENCODER RERANKING
# --------------------------------------------------

def rerank_chunks(query, chunks, top_k):

    if not chunks:
        return []

    pairs = [
        (query, chunk["text"])
        for chunk in chunks
    ]

    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    ranked_chunks = sorted(
        chunks,
        key=lambda chunk: chunk["rerank_score"],
        reverse=True
    )

    return ranked_chunks[:top_k]


# --------------------------------------------------
# HYBRID RETRIEVAL PIPELINE
# --------------------------------------------------

def retrieve(query, top_k=5, document_id=None):

    if collection.count() == 0:
        return []

    # 1. Semantic vector search
    vector_chunks = vector_search(
        query,
        CANDIDATE_K,
        document_id
    )

    # 2. Keyword search
    keyword_chunks = bm25_search(
        query,
        CANDIDATE_K,
        document_id
    )

    # 3. Combine rankings using RRF
    fused_chunks = reciprocal_rank_fusion(
        vector_chunks,
        keyword_chunks
    )

    # 4. Rerank using CrossEncoder
    final_chunks = rerank_chunks(
        query,
        fused_chunks,
        top_k
    )

    return final_chunks


# --------------------------------------------------
# TEST RETRIEVAL
# --------------------------------------------------

if __name__ == "__main__":

    question = input("Ask a question about your PDF: ")

    chunks = retrieve(question, top_k=3)

    if not chunks:

        print("No chunks found. Check whether your PDF was indexed.")

    else:

        for i, chunk in enumerate(chunks, start=1):

            print(f"\n--- Result {i} ---")

            print(f"Source: {chunk['source']}")
            print(f"Page: {chunk['page']}")
            print(f"RRF Score: {chunk['rrf_score']:.4f}")
            print(f"Reranker Score: {chunk['rerank_score']:.4f}")

            print(chunk["text"])
