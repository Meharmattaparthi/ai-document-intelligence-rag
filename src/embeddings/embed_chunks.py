import json
import numpy as np
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = "data/processed/metformin_chunks.json"
EMBEDDINGS_PATH = "data/processed/metformin_embeddings.npy"
METADATA_PATH = "data/processed/metformin_metadata.json"


def load_chunks(file_path):
    """Load document chunks from JSON."""

    with open(file_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return chunks


def create_embeddings(chunks):
    """Convert text chunks into numerical embeddings."""

    print("Loading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk["text"] for chunk in chunks]

    print(f"Creating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings


def save_embeddings(embeddings, file_path):
    """Save embeddings as a NumPy array."""

    np.save(file_path, embeddings)


def save_metadata(chunks, file_path):
    """Save chunk metadata separately."""

    metadata = []

    for chunk in chunks:
        metadata.append({
            "document": chunk["document"],
            "page_number": chunk["page_number"],
            "section": chunk.get("section", "GENERAL"),
            "text": chunk["text"]
        })

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":

    chunks = load_chunks(CHUNKS_PATH)

    embeddings = create_embeddings(chunks)

    save_embeddings(
        embeddings,
        EMBEDDINGS_PATH
    )

    save_metadata(
        chunks,
        METADATA_PATH
    )

    print("\nEmbedding process completed!")

    print(f"Number of chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embeddings saved to: {EMBEDDINGS_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")