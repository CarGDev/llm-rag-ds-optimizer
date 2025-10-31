"""Build indices (BM25 + HNSW) for a corpus."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from llmds.hnsw import HNSW
from llmds.inverted_index import InvertedIndex
from llmds.tokenizer import Tokenizer


def build_indices(
    corpus_file: Path,
    emb_file: Path | None,
    index_dir: Path,
    bm25: bool = True,
    hnsw: bool = True,
    ef_construction: int = 200,
    M: int = 16,
    embedding_dim: int = 384,
) -> dict:
    """
    Build inverted index and/or HNSW for a corpus.

    Args:
        corpus_file: Path to corpus JSONL file
        emb_file: Optional path to embeddings .npy file
        index_dir: Directory to save indices
        bm25: Whether to build BM25 inverted index
        hnsw: Whether to build HNSW index
        ef_construction: HNSW efConstruction parameter
        M: HNSW M parameter
        embedding_dim: Embedding dimension

    Returns:
        Dictionary with build statistics
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer = Tokenizer()
    stats = {}
    
    # Load embeddings if available
    embeddings = None
    if emb_file and emb_file.exists():
        print(f"Loading embeddings from {emb_file}...")
        embeddings = np.load(emb_file)
        print(f"Loaded {len(embeddings)} embeddings")
    
    # Build BM25 index
    if bm25:
        print("Building BM25 inverted index...")
        start_time = time.time()
        
        index = InvertedIndex(tokenizer=tokenizer)
        doc_count = 0
        
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    index.add_document(doc_id=int(doc["id"].split("_")[-1]) if doc["id"].split("_")[-1].isdigit() else doc_count, text=doc["text"])
                    doc_count += 1
                    
                    if doc_count % 10000 == 0:
                        print(f"Indexed {doc_count} documents...")
        
        # Save index metadata
        index_stats = index.stats()
        stats["bm25"] = {
            "build_time_sec": time.time() - start_time,
            "total_documents": index_stats["total_documents"],
            "total_terms": index_stats["total_terms"],
        }
        
        print(f"✓ BM25 index built: {stats['bm25']['total_documents']} documents, {stats['bm25']['build_time_sec']:.2f}s")
    
    # Build HNSW index
    if hnsw:
        if embeddings is None:
            print("Warning: No embeddings provided. Generating deterministic embeddings...")
            # Generate on-the-fly
            embeddings = []
            doc_count = 0
            rng = np.random.RandomState(42)
            with open(corpus_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        emb = rng.randn(embedding_dim).astype(np.float32)
                        emb = emb / np.linalg.norm(emb)
                        embeddings.append(emb)
                        doc_count += 1
            embeddings = np.stack(embeddings)
        
        print(f"Building HNSW index (M={M}, efConstruction={ef_construction})...")
        start_time = time.time()
        
        hnsw = HNSW(dim=embedding_dim, M=M, ef_construction=ef_construction, ef_search=50)
        
        for i, emb in enumerate(embeddings):
            hnsw.add(emb, i)
            if (i + 1) % 10000 == 0:
                print(f"Added {i + 1} vectors...")
        
        hnsw_stats = hnsw.stats()
        stats["hnsw"] = {
            "build_time_sec": time.time() - start_time,
            "num_vectors": hnsw_stats["num_vectors"],
            "num_layers": hnsw_stats["num_layers"],
        }
        
        print(f"✓ HNSW index built: {stats['hnsw']['num_vectors']} vectors, {stats['hnsw']['build_time_sec']:.2f}s")
    
    # Save statistics
    stats_file = index_dir / "build_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Indices built and saved to {index_dir}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build indices for corpus")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus JSONL file")
    parser.add_argument("--emb", type=Path, help="Embeddings .npy file")
    parser.add_argument("--index-dir", type=Path, required=True, help="Index output directory")
    parser.add_argument("--bm25", action="store_true", help="Build BM25 index")
    parser.add_argument("--hnsw", action="store_true", help="Build HNSW index")
    parser.add_argument("--ef", type=int, default=200, help="HNSW efConstruction")
    parser.add_argument("--M", type=int, default=16, help="HNSW M parameter")
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension")
    
    args = parser.parse_args()
    
    if not args.bm25 and not args.hnsw:
        print("Error: Must specify --bm25 and/or --hnsw")
        sys.exit(1)
    
    build_indices(
        corpus_file=args.corpus,
        emb_file=args.emb,
        index_dir=args.index_dir,
        bm25=args.bm25,
        hnsw=args.hnsw,
        ef_construction=args.ef,
        M=args.M,
        embedding_dim=args.dim,
    )


if __name__ == "__main__":
    main()
