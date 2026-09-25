import sys
import os

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.insert(0, project_root)

from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from dotenv import load_dotenv

from src.rag.rag_pipeline import (
    load_data,
    retrieve_chunks,
    build_prompt,
    generate_answer
)


class RAGService:
    """Reusable local RAG retrieval service."""

    def __init__(self):
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Loading reranker model...")
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        print("Loading document data...")
        self.embeddings, self.metadata = load_data()

        print(
            f"Loaded {len(self.metadata)} document chunks."
        )

    def retrieve(self, query, top_k=5, candidate_k=20):
        """Retrieve and rerank relevant document chunks."""

        retrieved_chunks = retrieve_chunks(
            query,
            self.embeddings,
            self.metadata,
            self.embedding_model,
            self.reranker,
            top_k=top_k,
            candidate_k=candidate_k
        )

        return retrieved_chunks

    def prepare_query(self, query, top_k=5, candidate_k=20):
        """
        Retrieve relevant chunks and build the grounded
        prompt that will later be sent to the LLM.
        """

        retrieved_chunks = self.retrieve(
            query,
            top_k=top_k,
            candidate_k=candidate_k
        )

        prompt = build_prompt(
            query,
            retrieved_chunks
        )

        return {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "prompt": prompt
        }

    def answer_query(self, query, top_k=5, candidate_k=20):
        """
        Run the complete RAG pipeline:
        retrieval -> reranking -> prompt construction -> LLM answer.
        """

        result = self.prepare_query(
            query,
            top_k=top_k,
            candidate_k=candidate_k
        )

        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            return {
                **result,
                "answer": None,
                "error": "GEMINI_API_KEY was not found."
            }

        try:
            client = genai.Client(api_key=api_key)

            answer = generate_answer(
                client,
                result["prompt"]
            )

            if answer is None:
                return {
                    **result,
                    "answer": None,
                    "error": "Gemini API quota is exhausted."
                }

            return {
                **result,
                "answer": answer,
                "error": None
            }

        except Exception as error:
            return {
                **result,
                "answer": None,
                "error": str(error)
            }


if __name__ == "__main__":

    service = RAGService()

    query = input("\nEnter your question: ")

    result = service.prepare_query(query)

    print("\n===== RETRIEVED SOURCES =====")

    for index, chunk in enumerate(
        result["retrieved_chunks"],
        start=1
    ):
        print(f"\nSource {index}")
        print(f"Page: {chunk['page_number']}")
        print(f"Section: {chunk['section']}")
        print(f"Score: {chunk['score']:.4f}")

    print("\n===== PROMPT READY =====")
    print("The grounded prompt has been prepared.")
    print("Gemini API was NOT called.")