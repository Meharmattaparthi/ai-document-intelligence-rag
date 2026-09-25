import json
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBEDDINGS_PATH = "data/processed/metformin_embeddings.npy"
METADATA_PATH = "data/processed/metformin_metadata.json"


def load_data():
    embeddings = np.load(EMBEDDINGS_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return embeddings, metadata


def search(
    query,
    embeddings,
    metadata,
    embedding_model,
    reranker,
    top_k=5,
    candidate_k=10
):
    """
    Two-stage retrieval:

    Stage 1:
        Use embeddings to retrieve candidate chunks.

    Stage 2:
        Use a CrossEncoder to rerank those candidates.
    """

    # ---------------------------------
    # Stage 1: Semantic retrieval
    # ---------------------------------

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    semantic_scores = np.dot(
        embeddings,
        query_embedding
    )

    candidate_indices = np.argsort(
        semantic_scores
    )[::-1][:candidate_k]

    # ---------------------------------
    # Stage 2: CrossEncoder reranking
    # ---------------------------------

    pairs = []

    for index in candidate_indices:
        pairs.append([
            query,
            metadata[index]["text"]
        ])

    reranker_scores = reranker.predict(pairs)

    # Sort candidates by reranker score
    ranked_results = sorted(
        zip(candidate_indices, reranker_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # ---------------------------------
    # Return final top-k results
    # ---------------------------------

    results = []

    for index, reranker_score in ranked_results[:top_k]:

        results.append({
            "score": float(reranker_score),
            "semantic_score": float(
                semantic_scores[index]
            ),
            "document": metadata[index]["document"],
            "page_number": metadata[index]["page_number"],
            "section": metadata[index].get(
                "section",
                "GENERAL"
            ),
            "text": metadata[index]["text"]
        })

    return results


if __name__ == "__main__":

    print("Loading embedding model...")

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Loading reranker model...")

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    embeddings, metadata = load_data()

    print(
        f"Loaded {len(metadata)} document chunks."
    )

    query = input("\nEnter your question: ")

    results = search(
        query,
        embeddings,
        metadata,
        embedding_model,
        reranker,
        top_k=5,
        candidate_k=20
    )

    print("\n===== RERANKED SEARCH RESULTS =====")

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")

        print(
            f"Reranker score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Semantic score: "
            f"{result['semantic_score']:.4f}"
        )

        print(
            f"Document: "
            f"{result['document']}"
        )

        print(
            f"Page: "
            f"{result['page_number']}"
        )

        print(
            f"Section: "
            f"{result['section']}"
        )

        print("\nText:")
        print(result["text"][:1000])

        print("\n" + "-" * 70)