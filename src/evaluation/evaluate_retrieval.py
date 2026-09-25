import json
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBEDDINGS_PATH = "data/processed/metformin_embeddings.npy"
METADATA_PATH = "data/processed/metformin_metadata.json"
QUESTIONS_PATH = "data/processed/evaluation_questions.json"


def load_data():
    embeddings = np.load(EMBEDDINGS_PATH)

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        questions = json.load(file)

    return embeddings, metadata, questions


def retrieve(
    query,
    embeddings,
    metadata,
    embedding_model,
    reranker,
    top_k=5,
    candidate_k=20
):
    """
    Two-stage retrieval:

    1. Retrieve candidate chunks using embeddings.
    2. Rerank candidates using a CrossEncoder.
    """

    # -----------------------------
    # Stage 1: Candidate retrieval
    # -----------------------------

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

    # -----------------------------
    # Stage 2: CrossEncoder
    # -----------------------------

    pairs = []

    for index in candidate_indices:
        pairs.append([
            query,
            metadata[index]["text"]
        ])

    reranker_scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(candidate_indices, reranker_scores),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for index, reranker_score in ranked_results[:top_k]:

        results.append({
            "score": float(reranker_score),
            "page_number": metadata[index]["page_number"],
            "section": metadata[index].get(
                "section",
                "GENERAL"
            ),
            "text": metadata[index]["text"]
        })

    return results


def evaluate_retrieval(
    questions,
    embeddings,
    metadata,
    embedding_model,
    reranker,
    top_k=5,
    candidate_k=20
):

    hit_count = 0
    recall_scores = []
    reciprocal_ranks = []

    print("\n===== RETRIEVAL EVALUATION =====")

    for item in questions:

        question = item["question"]
        expected_pages = item["expected_pages"]

        results = retrieve(
            question,
            embeddings,
            metadata,
            embedding_model,
            reranker,
            top_k=top_k,
            candidate_k=candidate_k
        )

        retrieved_pages = [
            result["page_number"]
            for result in results
        ]

        relevant_pages = set(expected_pages).intersection(
            retrieved_pages
        )

        recall = (
            len(relevant_pages)
            / len(set(expected_pages))
        )

        is_hit = recall > 0

        if is_hit:
            hit_count += 1

        recall_scores.append(recall)
        
        if relevant_pages:
            first_relevant_rank = next(
                rank
                for rank, page in enumerate(
                    retrieved_pages,
                    start=1
                )
                if page in relevant_pages
            )

            reciprocal_rank = 1 / first_relevant_rank
        else:
            reciprocal_rank = 0.0

        reciprocal_ranks.append(reciprocal_rank)

        print("\nQuestion:")
        print(question)

        print(
            f"Expected pages: "
            f"{expected_pages}"
        )

        print(
            f"Retrieved pages: "
            f"{retrieved_pages}"
        )
        
        print(f"Recall@{top_k}: {recall:.2%}")

        print(
            "Result: PASS"
            if is_hit
            else "Result: FAIL"
        )

    hit_rate = hit_count / len(questions)

    average_recall = (
        sum(recall_scores)
        / len(recall_scores)
    )

    mean_reciprocal_rank = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
    )

    print("\n" + "=" * 50)

    print(
        f"Hit Rate@{top_k}: "
        f"{hit_rate:.2%}"
    )

    print(
        f"Average Recall@{top_k}: "
        f"{average_recall:.2%}"
    )

    print(
        f"Mean Reciprocal Rank: "
        f"{mean_reciprocal_rank:.2f}"
    )

    print("=" * 50)


if __name__ == "__main__":

    print("Loading embedding model...")

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Loading reranker model...")

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    embeddings, metadata, questions = load_data()

    print(
        f"Loaded {len(metadata)} document chunks."
    )

    print(
        f"Loaded {len(questions)} evaluation questions."
    )

    evaluate_retrieval(
        questions,
        embeddings,
        metadata,
        embedding_model,
        reranker,
        top_k=5,
        candidate_k=20
    )