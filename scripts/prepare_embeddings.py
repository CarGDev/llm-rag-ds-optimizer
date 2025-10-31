"""Prepare embeddings for datasets."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_deterministic_embeddings(
    corpus_file: Path,
    output_file: Path,
    dim: int = 384,
    seed: int = 42,
    limit: int | None = None,
) -> None:
    """
    Generate deterministic embeddings for a corpus.

    Args:
        corpus_file: Path to corpus JSONL file
        output_file: Output .npy file for embeddings
        dim: Embedding dimension
        seed: Random seed for reproducibility
        limit: Optional limit on number of documents
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.RandomState(seed)
    
    embeddings = []
    count = 0
    
    print(f"Generating deterministic embeddings (dim={dim}, seed={seed})...")
    
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if limit and count >= limit:
                break
            
            if line.strip():
                doc = json.loads(line)
                # Generate deterministic embedding based on document ID
                doc_hash = hash(doc["id"]) % (2**31)
                rng_local = np.random.RandomState(seed + doc_hash)
                
                # Generate normalized random vector
                emb = rng_local.randn(dim).astype(np.float32)
                emb = emb / np.linalg.norm(emb)
                
                embeddings.append(emb)
                count += 1
                
                if count % 10000 == 0:
                    print(f"Processed {count} documents...")
    
    embeddings_array = np.stack(embeddings)
    np.save(output_file, embeddings_array)
    print(f"Saved {len(embeddings)} embeddings to {output_file}")


def load_embeddings(emb_file: Path) -> np.ndarray:
    """Load embeddings from .npy file."""
    return np.load(emb_file)


def main():
    parser = argparse.ArgumentParser(description="Prepare embeddings for corpus")
    parser.add_argument("--input", type=Path, required=True, help="Corpus JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output .npy file")
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--limit", type=int, help="Limit number of documents")
    
    args = parser.parse_args()
    
    generate_deterministic_embeddings(
        args.input,
        args.output,
        dim=args.dim,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

