import sys
import os

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, project_root)

from sentence_transformers import SentenceTransformer, CrossEncoder

from src.rag.rag_pipeline import (
    load_data,
    retrieve_chunks,
    build_prompt
)


if __name__ == "__main__":

    query = "What are the contraindications of metformin?"

    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading reranker model...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("Loading document data...")
    embeddings, metadata = load_data()

    print(f"Loaded {len(metadata)} document chunks.")

    print("\nRunning retrieval and reranking...")

    retrieved_chunks = retrieve_chunks(
        query,
        embeddings,
        metadata,
        embedding_model,
        reranker,
        top_k=5,
        candidate_k=20
    )

    print("\n===== LOCAL RETRIEVAL TEST =====")

    for i, chunk in enumerate(retrieved_chunks, start=1):

        print(f"\nResult {i}")
        print(f"Reranker score: {chunk['score']:.4f}")
        print(f"Semantic score: {chunk['semantic_score']:.4f}")
        print(f"Page: {chunk['page_number']}")
        print(f"Section: {chunk['section']}")

        print("\nText:")
        print(chunk["text"][:500])

        print("-" * 70)

    prompt = build_prompt(
        query,
        retrieved_chunks
    )

    print("\n===== PROMPT PREVIEW =====")
    print(prompt[:3000])

    print("\n===== TEST COMPLETED =====")
    print("Retrieval: OK")
    print("Reranking: OK")
    print("Section metadata: OK")
    print("Prompt construction: OK")
    print("Gemini API: NOT CALLED")