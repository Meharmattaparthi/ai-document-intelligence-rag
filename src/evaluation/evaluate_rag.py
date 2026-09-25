import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from google.genai import types
from dotenv import load_dotenv

EMBEDDINGS_PATH = "data/processed/metformin_embeddings.npy"
METADATA_PATH = "data/processed/metformin_metadata.json"
QUESTIONS_PATH = "data/processed/rag_evaluation_questions.json"


def load_data():

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

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


def retrieve_chunks(
    query,
    embeddings,
    metadata,
    embedding_model,
    reranker,
    top_k=5,
    candidate_k=20
):

    # Stage 1: semantic retrieval

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

    # Stage 2: CrossEncoder reranking

    pairs = []

    for index in candidate_indices:

        pairs.append([
            query,
            metadata[index]["text"]
        ])

    reranker_scores = reranker.predict(
        pairs
    )

    ranked_results = sorted(
        zip(
            candidate_indices,
            reranker_scores
        ),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for index, reranker_score in ranked_results[:top_k]:

        results.append({
            "score": float(reranker_score),
            "document": metadata[index]["document"],
            "page_number": metadata[index]["page_number"],
            "text": metadata[index]["text"]
        })

    return results


def build_prompt(
    query,
    retrieved_chunks
):

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"[Source: {chunk['document']}, "
            f"Page {chunk['page_number']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

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


def generate_answer(
    client,
    prompt
):

    import time

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

            # Daily quota exhausted.
            if (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
            ):

                print(
                    "\nGemini API quota has been exhausted."
                )

                print(
                    "Skipping this question."
                )

                return None

            # Temporary Gemini service unavailability.
            if (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            ):

                if attempt < max_retries:

                    wait_time = 5 * (
                        2 ** (attempt - 1)
                    )

                    print(
                        f"\nGemini service temporarily unavailable "
                        f"(attempt {attempt}/{max_retries})."
                    )

                    print(
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                    continue

                print(
                    "\nGemini service remained unavailable "
                    "after retries."
                )

                return None

            # Other unexpected errors.
            print(
                f"\nGemini request failed "
                f"(attempt {attempt}/{max_retries}): {error}"
            )

            if attempt == max_retries:
                raise

            wait_time = 5 * (
                2 ** (attempt - 1)
            )

            print(
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

    return None

def calculate_keyword_coverage(reference_answer, generated_answer):
    reference_words = set(
        re.findall(r"\b[a-zA-Z0-9]+\b", reference_answer.lower())
    )
    generated_words = set(
        re.findall(r"\b[a-zA-Z0-9]+\b", generated_answer.lower())
    )

    if not reference_words:
        return 0.0

    matched_words = reference_words.intersection(generated_words)
    coverage = (len(matched_words) / len(reference_words)) * 100

    return coverage


def calculate_groundedness(generated_answer, retrieved_chunks):
    context_text = " ".join(
        chunk["text"] for chunk in retrieved_chunks
    ).lower()

    sentences = re.split(r"(?<=[.!?])\s+", generated_answer.strip())

    valid_sentences = [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip().split()) >= 3
    ]

    if not valid_sentences:
        return 0.0

    grounded_sentences = 0

    for sentence in valid_sentences:
        words = re.findall(r"\b[a-zA-Z0-9]+\b", sentence.lower())

        meaningful_words = [
            word
            for word in words
            if len(word) >= 4
        ]

        if not meaningful_words:
            continue

        matched_words = [
            word
            for word in meaningful_words
            if word in context_text
        ]

        match_ratio = len(matched_words) / len(meaningful_words)

        if match_ratio >= 0.30:
            grounded_sentences += 1

    groundedness = (
        grounded_sentences / len(valid_sentences)
    ) * 100

    return groundedness

if __name__ == "__main__":

    load_dotenv()

    client = genai.Client()

    print(
        "Loading embedding model..."
    )

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print(
        "Loading reranker model..."
    )

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

    print(
        "\n===== RAG ANSWER EVALUATION ====="
    )

    for number, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]

        reference_answer = item[
            "reference_answer"
        ]

        retrieved_chunks = retrieve_chunks(
            question,
            embeddings,
            metadata,
            embedding_model,
            reranker,
            top_k=5,
            candidate_k=20
        )

        prompt = build_prompt(
            question,
            retrieved_chunks
        )

        answer = generate_answer(
            client,
            prompt
        )

        if answer is None:

            print(
                "\nGenerated RAG answer:"
            )

            print(
                "Skipped because Gemini API quota is exhausted."
            )

            continue

        coverage = calculate_keyword_coverage(
            reference_answer,
            answer
        )

        groundedness = calculate_groundedness(
            answer,
            retrieved_chunks
        )

        print(
            "\n" + "=" * 70
        )

        print(
            f"Question {number}:"
        )

        print(question)

        print(
            "\nReference answer:"
        )

        print(reference_answer)

        print(
            "\nGenerated RAG answer:"
        )

        print(answer)
        
        print(
            f"\nKeyword coverage: {coverage:.2f}%"
        )

        print(
            f"Groundedness: {groundedness:.2f}%"
        )

        print(
            "\nRetrieved pages:"
        )

        print([
            chunk["page_number"]
            for chunk in retrieved_chunks
        ])

    print(
        "\n" + "=" * 70
    )

    print(
        "RAG evaluation completed."
    )