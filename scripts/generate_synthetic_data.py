"""Generate synthetic data for testing and benchmarks."""

import random
from pathlib import Path

import numpy as np


def generate_synthetic_documents(num_docs: int = 1000, output_file: Path = Path("data/documents.txt")):
    """Generate synthetic documents for indexing."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "cat", "mouse", "elephant", "tiger", "lion", "bear", "wolf",
        "rabbit", "deer", "bird", "fish", "snake", "monkey", "panda",
        "computer", "science", "machine", "learning", "artificial", "intelligence",
        "neural", "network", "deep", "learning", "transformer", "attention",
        "language", "model", "natural", "processing", "text", "generation",
    ]

    with open(output_file, "w") as f:
        for i in range(num_docs):
            doc_length = random.randint(20, 200)
            doc_words = random.choices(words, k=doc_length)
            doc_text = " ".join(doc_words)
            f.write(f"{i}\t{doc_text}\n")

    print(f"Generated {num_docs} documents in {output_file}")


def generate_synthetic_embeddings(
    num_vectors: int = 1000,
    dim: int = 384,
    output_file: Path = Path("data/embeddings.npy"),
):
    """Generate synthetic embedding vectors."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    embeddings = np.random.randn(num_vectors, dim).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    np.save(output_file, embeddings)
    print(f"Generated {num_vectors} embeddings in {output_file}")


if __name__ == "__main__":
    generate_synthetic_documents(num_docs=1000)
    generate_synthetic_embeddings(num_vectors=1000, dim=384)

