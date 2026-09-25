import json
import numpy as np
import time
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from google.genai import types
from dotenv import load_dotenv


EMBEDDINGS_PATH = "data/processed/metformin_embeddings.npy"
METADATA_PATH = "data/processed/metformin_metadata.json"


def load_data():
    """Load document embeddings and metadata."""

    embeddings = np.load(EMBEDDINGS_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return embeddings, metadata


def retrieve_chunks(
    query,
    embeddings,
    metadata,
    model,
    reranker,
    top_k=5,
    candidate_k=20
):
    """
    Two-stage retrieval:

    Stage 1:
        Use embeddings to retrieve candidate chunks.

    Stage 2:
        Use a CrossEncoder to rerank those candidates.
    """

    # -----------------------------
    # Stage 1: Semantic retrieval
    # -----------------------------

    query_embedding = model.encode(
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
    # Stage 2: CrossEncoder reranking
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

    # -----------------------------
    # Return final top-k chunks
    # -----------------------------

    results = []

    for index, reranker_score in ranked_results[:top_k]:

        results.append({
            "score": float(reranker_score),
            "semantic_score": float(
                semantic_scores[index]
            ),
            "document": metadata[index]["document"],
            "page_number": metadata[index]["page_number"],
            "section": metadata[index].get("section", "GENERAL"),
            "text": metadata[index]["text"]
        })

    return results


def build_prompt(query, retrieved_chunks):
    """Build a grounded prompt for the LLM."""

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"[Source: {chunk['document']}, "
            f"Page {chunk['page_number']}, "
            f"Section: {chunk['section']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided document context.

Rules:
1. Do not invent information.
2. Use all relevant retrieved context needed to answer the question completely.
3. Prefer specific facts, names, categories, values, and conditions from the document.
4. Do not omit important information that is directly relevant to the question.
5. If the answer is not present in the context, say that the information was not found in the provided document.
6. Cite the relevant document page numbers in your answer.
7. Keep the answer clear and concise.
8. This is a technical document QA demonstration, not medical advice.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{query}
"""

    return prompt


def generate_answer(client, prompt):
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    )
                )
            )

            return response.text

        except Exception as error:
            error_message = str(error)

            # Do not retry if the daily quota is exhausted.
            if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                print("\nGemini API quota has been exhausted.")
                print("Stopping without retrying.")
                return None

            # Retry temporary server/service errors.
            if "503" in error_message or "UNAVAILABLE" in error_message:
                if attempt < max_retries:
                    wait_time = 5 * (2 ** (attempt - 1))

                    print(
                        f"\nGemini service temporarily unavailable "
                        f"(attempt {attempt}/{max_retries})."
                    )
                    print(f"Retrying in {wait_time} seconds...")

                    time.sleep(wait_time)
                    continue

                print("\nGemini service remained unavailable after retries.")
                return None

            # Retry other unexpected temporary failures.
            print(
                f"\nGemini request failed "
                f"(attempt {attempt}/{max_retries}): {error}"
            )

            if attempt == max_retries:
                raise

            wait_time = 5 * (2 ** (attempt - 1))
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    return None

if __name__ == "__main__":

    load_dotenv()

    client = genai.Client()

    print("Loading embedding model...")

    model = SentenceTransformer(
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

    query = input(
        "\nEnter your question: "
    )

    retrieved_chunks = retrieve_chunks(
        query,
        embeddings,
        metadata,
        model,
        reranker,
        top_k=5,
        candidate_k=20
    )

    print("\nRetrieved relevant chunks.")

    prompt = build_prompt(
        query,
        retrieved_chunks
    )

    print("\nGenerating answer...\n")

    answer = generate_answer(
        client,
        prompt
    )

    if answer is None:
        print("\nRAG answer generation was skipped because the Gemini API quota is exhausted.")
        print("Retrieved chunks and source information are still available.")
        raise SystemExit

    print("===== RAG ANSWER =====")
    print(answer)

    print("\n===== SOURCES =====")

    for chunk in retrieved_chunks:

        print(
            f"- {chunk['document']} "
            f"(Page {chunk['page_number']}, "
            f"Section: {chunk['section']}) "
            f"[Score: {chunk['score']:.4f}]"
        )